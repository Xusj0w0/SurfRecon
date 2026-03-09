import os
import os.path as osp
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from torch_scatter import scatter_add
from tqdm import tqdm

from internal.cameras import Camera, Cameras
from internal.density_controllers.density_controller import Utils
from internal.models.mip_splatting import MipSplattingModelMixin
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.general_utils import build_rotation
from internal.utils.ssim import create_window

from ...utils.graphic_utils import depth_to_pointmap
from ...utils.mesh import Meshes
from ...utils.partitionable_scene import MinMaxBoundingBox
from ...utils.sparse_voxel_grid import SparseVoxelFeatureAccumulator
from ..mesh_gaussian import MeshGaussianUtils
from ..renderers.importance import rasterize_importance
from .mesh_gaussian_densifier import (MeshGaussianDensityController,
                                      MeshGaussianDensityControllerImpl)


@dataclass
class DepthGuidedDensityController(MeshGaussianDensityController):
    depth_guided_densification_interval: int = 3000

    gt_depth_guided_until_iter: int = 6000

    guided_densify_ratio: float = 0.2

    guided_prune_ratio: float = 0.0

    def instantiate(self, *args, **kwargs):
        return DepthGuidedDensityControllerImpl(self)


class DepthGuidedDensityControllerImpl(MeshGaussianDensityControllerImpl):
    config: DepthGuidedDensityController

    def setup(self, stage: str, pl_module) -> None:
        super().setup(stage, pl_module)

        self._enable_guided_densify = False
        self._cached_data: Optional[list] = None
        self._transform_matrix: Optional[torch.Tensor] = None
        self._bounding_box: Optional[MinMaxBoundingBox] = None
        self._scene_extent: float = -1.0

        self._guided_densify_step = -32768
        self._global_step = -1
        self._reset_skipped = False

    def after_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module):
        if global_step >= self.config.densify_until_iter:
            return

        _optimizers = self._exclude_occupancy_optimizer(optimizers)
        self._global_step = global_step
        self._n_gaussians_prev = gaussian_model.n_gaussians
        with torch.no_grad():
            self.update_states(outputs)
            gaussian_changed = False

            # guided densify
            if (
                self._enable_guided_densify
                and global_step > self.config.densify_from_iter
                and global_step % self.config.depth_guided_densification_interval == 0
            ):
                cameras = pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras
                if global_step > self.config.gt_depth_guided_until_iter:
                    self._ssim_aware_densify(gaussian_model, pl_module.renderer, cameras, _optimizers)
                else:
                    self._depth_guided_densify_and_prune(gaussian_model, pl_module.renderer, cameras, _optimizers)
                gaussian_changed = True
                self._guided_densify_step = global_step

            # trim before resetting opacity
            if (
                global_step > self.config.trim_from_iter
                and global_step % self.config.trim_interval == 0
                and (global_step >= self._guided_densify_step + 2 * self.config.densification_interval)
            ):
                cameras = pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras
                prune_mask = self._get_trimming_prune_mask(
                    cameras=cameras,
                    gaussian_model=gaussian_model,
                    top_k=self.config.top_k,
                    ratio=self.config.trim_ratio,
                )
                self._prune_points(prune_mask, gaussian_model, _optimizers)
                gaussian_changed = True

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=_optimizers,
                )
                gaussian_changed = True

            if (
                global_step % self.config.opacity_reset_interval == 0
                or (torch.all(pl_module.background_color == 1.0) and global_step == self.config.densify_from_iter)
                # or (self._reset_skipped and global_step >= self._guided_densify_step + 1000)
            ):
                self._reset_opacities(gaussian_model, _optimizers)
                self.opacity_reset_at = global_step

            if gaussian_changed:
                self._recompute_3d_filter(gaussian_model)

    def _prune_points(self, mask, gaussian_model, optimizers):
        if self._global_step < self.opacity_reset_at + 2 * self.config.densification_interval:
            return
        return super()._prune_points(mask, gaussian_model, optimizers)

    @torch.no_grad()
    def _depth_guided_densify_and_prune(self, gaussian_model: VanillaGaussianModel, renderer, cameras: Cameras, optimizers):
        if self.config.guided_densify_ratio <= 0.0 and self.config.guided_prune_ratio <= 0.0:
            return

        indices = torch.randperm(len(cameras))
        # Compute densification scores
        device = gaussian_model.get_xyz.device
        bg_color = torch.zeros((3,), device=device)
        voxel_grid = SparseVoxelFeatureAccumulator(voxel_size=2e-3 * self._scene_extent, feat_dim=4)  # rgb+diff_depth
        prune_scores = torch.zeros((gaussian_model.n_gaussians,), device=device)
        for idx in tqdm(range(len(cameras)), desc="Computing depth-guided densification scores", leave=False):
            # load camera, gt_image, and gt_depth
            camera = cameras[indices[idx]].to_device(device)
            _, (image_name, gt_image, _), gt_depth_data = self._cached_data[indices[idx]]
            if gt_depth_data is None:
                continue
            gt_image = gt_image.to(device)
            gt_depth_ds: torch.Tensor = gt_depth_data.get(device=device)
            depth_mean, depth_std = gt_depth_ds.mean(), gt_depth_ds.std()
            clamp_min, clamp_max = depth_mean - 2 * depth_std, depth_mean + 2 * depth_std
            # clamp_min, clamp_max = torch.quantile(gt_depth_ds, 0.05), torch.quantile(gt_depth_ds, 0.95)

            # render
            outputs = renderer(camera, gaussian_model, bg_color, render_types=["rgb", "depth"])
            render = outputs["render"]
            gaussian_depth = outputs["median_depth"].squeeze()
            gt_depth_shape = (int(gt_depth_data.camera.height), int(gt_depth_data.camera.width))
            if gaussian_depth.shape[-2:] != gt_depth_shape:
                gt_depth = F.interpolate(
                    gt_depth_ds[None, None, ...], size=gaussian_depth.shape[-2:], mode="bilinear", align_corners=True
                ).squeeze()
            else:
                gt_depth = gt_depth_ds
            valid_pixels = torch.logical_and(gt_depth > clamp_min, gt_depth < clamp_max)
            loss = self._compute_loss(render, gt_image)

            # compute depth diff
            gaussian_depth_clamp, gt_depth_clamp = gaussian_depth.clamp_min(min=clamp_min), gt_depth.clamp_min(min=clamp_min)
            # diff_depth = (gaussian_depth_clamp - gt_depth_clamp) / gt_depth_clamp
            diff_depth = gaussian_depth_clamp - gt_depth_clamp
            # diff < 0: prune points
            # diff > 0: densify points

            # accum prune score
            if self.config.guided_prune_ratio > 0.0:
                prune_pixels = torch.logical_and(valid_pixels, diff_depth < -2e-3 * self._scene_extent)
                prune_weights = loss * prune_pixels
                importances = rasterize_importance(camera, gaussian_model, weight_map=prune_weights)
                scores = importances["accum_scaled_weights"] / (importances["num_hit_pixels"] + 1e-5)
                prune_scores += scores

            # gather valid coords
            if self.config.guided_densify_ratio > 0.0:
                camera_ds: Camera = gt_depth_data.camera.to_device(device)
                densify_pixels = torch.logical_and(valid_pixels, diff_depth > 0.0 * self._scene_extent)
                densify_weights = loss * (diff_depth.abs()) * densify_pixels
                coords = self._gather_valid_coords(camera_ds, densify_weights, gt_image, gt_depth_ds)
                if coords is not None:
                    voxel_grid.update(coords[:, :3], coords[:, 3:])

        if self.config.guided_densify_ratio > 0.0:
            # build new Gaussians from voxels
            n_gaussians = gaussian_model.n_gaussians
            xyz, vals, cnt = voxel_grid.finalize(return_count=True)
            rgb = (vals[:, :3] / (cnt + 1e-8)).clamp(min=0.0, max=1.0)
            intensity = vals[:, 3] / (cnt.squeeze() + 1e-8)
            n_samples = min(int(self.config.guided_densify_ratio * n_gaussians), (intensity > 0).sum().item())
            new_properties, prune_mask = self._build_gaussians_from_voxels(
                xyz=xyz, rgb=rgb, intensity=intensity, n_samples=n_samples, gaussian_model=gaussian_model
            )
            self._prune_points(prune_mask, gaussian_model, optimizers)
            self._n_gaussians_prev = gaussian_model.n_gaussians  # update prev num after pruning

        # prune with scores
        if self.config.guided_prune_ratio > 0.0:
            prune_mask = prune_scores > torch.quantile(prune_scores, 1 - self.config.guided_prune_ratio)
            self._prune_points(prune_mask, gaussian_model, optimizers)
            self._n_gaussians_prev = gaussian_model.n_gaussians  # update prev num after pruning

        # densify
        if self.config.guided_densify_ratio > 0.0:
            new_parameters = Utils.cat_tensors_to_properties(new_properties, gaussian_model, optimizers)
            gaussian_model.properties = new_parameters
            self._extend_densification_states(len(new_properties["means"]))
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _ssim_aware_densify(self, gaussian_model: VanillaGaussianModel, renderer, cameras: Cameras, optimizers):
        if self.config.guided_densify_ratio <= 0.0:
            return

        indices = torch.randperm(len(cameras))
        # Compute densification scores
        device = gaussian_model.get_xyz.device
        bg_color = torch.zeros((3,), device=device)
        voxel_grid = SparseVoxelFeatureAccumulator(voxel_size=2e-3 * self._scene_extent, feat_dim=4)  # rgb+diff_depth
        for idx in tqdm(range(len(cameras)), desc="Computing depth-guided densification scores", leave=False):
            # load camera, gt_image, and gt_depth
            camera = cameras[indices[idx]].to_device(device)
            _, (image_name, gt_image, _), gt_depth_data = self._cached_data[indices[idx]]
            if gt_depth_data is None:
                continue
            gt_image = gt_image.to(device)
            gt_depth_ds: torch.Tensor = gt_depth_data.get(device=device)
            depth_mean, depth_std = gt_depth_ds.mean(), gt_depth_ds.std()
            clamp_min, clamp_max = depth_mean - 2 * depth_std, depth_mean + 2 * depth_std

            # render
            outputs = renderer(camera, gaussian_model, bg_color, render_types=["rgb", "depth"])
            render = outputs["render"]
            conf = outputs["mask"].squeeze()
            gaussian_depth = outputs["median_depth"].squeeze()
            gt_depth_shape = (int(gt_depth_data.camera.height), int(gt_depth_data.camera.width))
            if gaussian_depth.shape[-2:] != gt_depth_shape:
                gt_depth = F.interpolate(
                    gt_depth_ds[None, None, ...], size=gaussian_depth.shape[-2:], mode="bilinear", align_corners=True
                ).squeeze()
            else:
                gt_depth = gt_depth_ds
            valid_pixels = torch.logical_and(gt_depth > clamp_min, gt_depth < clamp_max)
            loss = self._compute_loss(render, gt_image)

            # gather valid coords
            camera_ds: Camera = gt_depth_data.camera.to_device(device)
            densify_weights = loss * conf * valid_pixels
            coords = self._gather_valid_coords(camera_ds, densify_weights, gt_image, gt_depth_ds)
            if coords is not None:
                voxel_grid.update(coords[:, :3], coords[:, 3:])

        # build new Gaussians from voxels
        n_gaussians = gaussian_model.n_gaussians
        xyz, vals, cnt = voxel_grid.finalize(return_count=True)
        rgb = (vals[:, :3] / (cnt + 1e-8)).clamp(min=0.0, max=1.0)
        intensity = vals[:, 3] / (cnt.squeeze() + 1e-8)
        n_samples = min(int(self.config.guided_densify_ratio * n_gaussians), (intensity > 0).sum().item())
        new_properties, prune_mask = self._build_gaussians_from_voxels(
            xyz=xyz, rgb=rgb, intensity=intensity, n_samples=n_samples, gaussian_model=gaussian_model
        )

        # prune points
        self._prune_points(prune_mask, gaussian_model, optimizers)
        self._n_gaussians_prev = gaussian_model.n_gaussians  # update prev num after pruning

        # densify
        new_parameters = Utils.cat_tensors_to_properties(new_properties, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters
        self._extend_densification_states(len(new_properties["means"]))
        torch.cuda.empty_cache()

    def _extend_densification_states(self, n_samples: int):
        device = self.max_radii2D.device
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((n_samples,), device=device)], dim=0)
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((n_samples, 1), device=device)], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros((n_samples, 1), device=device)], dim=0)
        self.xyz_gradient_accum_abs = torch.cat([self.xyz_gradient_accum_abs, torch.zeros((n_samples, 1), device=device)], dim=0)
        self.xyz_gradient_accum_abs_max = torch.cat([self.xyz_gradient_accum_abs_max, torch.zeros((n_samples, 1), device=device)], dim=0)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _build_gaussians_from_voxels(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        intensity: torch.Tensor,
        n_samples: int,
        gaussian_model: VanillaGaussianModel,
        num_neighbors: int = 4,
    ):
        from internal.utils.sh_utils import RGB2SH

        device = xyz.device

        prob = intensity / (intensity.sum() + 1e-12)
        indices = np.random.choice(len(prob), size=n_samples, p=prob.cpu().numpy(), replace=True)
        indices = torch.from_numpy(indices).to(device=device, dtype=torch.long)
        uniq_ids, inv, counts = torch.unique(indices, return_inverse=True, return_counts=True)
        vox, vox_rgb = xyz[uniq_ids], rgb[uniq_ids]
        n_vox = len(vox)

        # find voxels' knn in gaussian model
        knn = knn_points(vox[None], gaussian_model.get_xyz[None], K=num_neighbors, return_nn=True)
        dists = knn.dists[0]  # (n_samples, num_neighbors)
        nearest_points = knn.knn[0]  # (n_samples, num_neighbors, 3)
        # compute interpolation weights with inverse distance
        weights = 1.0 / (torch.sqrt(dists) + 1e-12)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (n_samples, num_neighbors)
        # sample neighbors according to weights
        row_id = torch.repeat_interleave(torch.arange(n_vox).to(device), counts, dim=0)
        col_id = torch.multinomial(weights[row_id], num_samples=1, replacement=True).squeeze()
        lin = row_id * num_neighbors + col_id
        counts_per_neighbor = torch.zeros(n_vox * num_neighbors, dtype=torch.long, device=device)
        counts_per_neighbor.scatter_add_(0, lin, torch.ones_like(lin, dtype=torch.long, device=device))
        # interpolate new points
        starts = torch.cumsum(counts_per_neighbor, dim=0) - counts_per_neighbor
        seg_id = torch.repeat_interleave(torch.arange(counts_per_neighbor.numel()).to(device), counts_per_neighbor, dim=0)
        pos = torch.arange(len(seg_id)).to(device) - starts[seg_id]
        interp = (pos + 1) / (counts_per_neighbor[seg_id] + 1)  # (n_samples,)
        interp = interp.unsqueeze(-1)
        i, j = seg_id // num_neighbors, seg_id % num_neighbors
        p1_sel, nn_sel = vox[i], nearest_points[i, j]
        p_interp = (1.0 - interp) * p1_sel + interp * nn_sel  # (n_samples, 3)

        # build Gaussian properties
        means = p_interp
        _all_points = torch.cat([gaussian_model.get_xyz, means], dim=0)
        knn_all = knn_points(means[None], _all_points[None], K=2)
        dists_all = knn_all.dists[0, :, 1]  # (n_samples,)
        scales = torch.sqrt(dists_all.clamp_min(1e-6))[..., None].repeat(1, 3)  # (n_samples, 3)
        scales = gaussian_model.scale_inverse_activation(scales)
        rots = torch.zeros((n_samples, 4)).to(means)
        rots[:, 0] = 1.0
        opacities = gaussian_model.opacity_inverse_activation(0.1 * torch.ones((n_samples, 1)).to(means))
        shs_dc = RGB2SH(torch.repeat_interleave(vox_rgb, counts, dim=0))[:, None]
        shs_rest = torch.zeros((n_samples, (gaussian_model.max_sh_degree + 1) ** 2 - 1, 3)).to(means)

        # shrink sampled Gaussians
        ids = knn.idx[0]  # (n_samples, num_neighbors)
        referred_ids, referred_counts = torch.unique(ids.flatten(), return_counts=True)

        _means = gaussian_model.get_xyz[referred_ids]
        _scales = gaussian_model.get_scaling[referred_ids]  #  / (referred_counts.float()[:, None] ** (1 / 3))
        _scales = gaussian_model.scale_inverse_activation(_scales.clamp_min(1e-6))
        _rots = gaussian_model.get_rotation[referred_ids]
        _opacities = torch.ones_like(gaussian_model.get_opacity[referred_ids]) * 0.1
        _opacities = gaussian_model.opacity_inverse_activation(_opacities)
        _shs_dc = gaussian_model.shs_dc[referred_ids]
        _shs_rest = gaussian_model.shs_rest[referred_ids]

        # build properties
        means = torch.cat([means, _means], dim=0)
        scales = torch.cat([scales, _scales], dim=0)
        rots = torch.cat([rots, _rots], dim=0)
        opacities = torch.cat([opacities, _opacities], dim=0)
        shs_dc = torch.cat([shs_dc, _shs_dc], dim=0)
        shs_rest = torch.cat([shs_rest, _shs_rest], dim=0)
        new_properties = {
            "means": nn.Parameter(means, requires_grad=True),
            "scales": nn.Parameter(scales, requires_grad=True),
            "rotations": nn.Parameter(rots, requires_grad=True),
            "opacities": nn.Parameter(opacities, requires_grad=True),
            "shs_dc": nn.Parameter(shs_dc, requires_grad=True),
            "shs_rest": nn.Parameter(shs_rest, requires_grad=True),
        }
        if MipSplattingModelMixin._filter_3d_name in gaussian_model.get_property_names():
            new_properties[MipSplattingModelMixin._filter_3d_name] = nn.Parameter(
                torch.zeros((len(means), 1)).to(means), requires_grad=False
            )

        prune_mask = torch.zeros((gaussian_model.n_gaussians,), dtype=torch.bool, device=means.device)
        prune_mask[referred_ids] = True

        torch.cuda.empty_cache()

        return new_properties, prune_mask

    @torch.no_grad()
    def _build_gaussians_from_pcd(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        intensity: torch.Tensor,
        n_samples: int,
        gaussian_model: VanillaGaussianModel,
    ):
        from internal.utils.sh_utils import RGB2SH

        device = xyz.device

        prob = intensity / (intensity.sum() + 1e-12)
        indices = np.random.choice(len(prob), size=n_samples, p=prob.cpu().numpy(), replace=False)
        indices = torch.from_numpy(indices).to(device=device, dtype=torch.long)

        means, shs_dc = xyz[indices], RGB2SH(rgb[indices]).unsqueeze(1)
        all_points = torch.cat([gaussian_model.get_xyz, means], dim=0)
        knn = knn_points(means[None], all_points[None], K=2)
        dists = knn.dists[0, :, 1]  # (n_samples,)
        scales = torch.sqrt(dists.clamp_min(1e-6))[..., None].repeat(1, 3)  # (n_samples, 3)
        scales = gaussian_model.scale_inverse_activation(scales)
        rots = torch.zeros((n_samples, 4)).to(means)
        rots[:, 0] = 1.0
        opacities = gaussian_model.opacity_inverse_activation(0.1 * torch.ones((n_samples, 1)).to(means))
        shs_rest = torch.zeros((n_samples, (gaussian_model.max_sh_degree + 1) ** 2 - 1, 3)).to(means)  # (n_samples, sh_dim, 3)

        new_properties = {
            "means": nn.Parameter(means, requires_grad=True),
            "scales": nn.Parameter(scales, requires_grad=True),
            "rotations": nn.Parameter(rots, requires_grad=True),
            "opacities": nn.Parameter(opacities, requires_grad=True),
            "shs_dc": nn.Parameter(shs_dc, requires_grad=True),
            "shs_rest": nn.Parameter(shs_rest, requires_grad=True),
        }
        if MipSplattingModelMixin._filter_3d_name in gaussian_model.get_property_names():
            new_properties[MipSplattingModelMixin._filter_3d_name] = nn.Parameter(
                torch.zeros((len(means), 1)).to(means), requires_grad=False
            )

        return new_properties

    def _gather_valid_coords(self, camera: Cameras, densify_weights: torch.Tensor, gt_image: torch.Tensor, gt_depth_ds: torch.Tensor):
        need_to_ds = torch.cat([densify_weights[None], gt_image], dim=0)
        ds = F.interpolate(need_to_ds[None], size=gt_depth_ds.shape[-2:], mode="bilinear", align_corners=True)[0]
        densify_weights_ds, gt_image_ds = ds[0:1], ds[1:]
        pointmap = depth_to_pointmap(gt_depth_ds.unsqueeze(0), camera)
        c2w = torch.linalg.inv(camera.world_to_camera.T)
        pointmap = torch.einsum("ij, jhw -> ihw", c2w, torch.cat([pointmap, torch.ones_like(pointmap[:1])], dim=0))[:3]
        coords = torch.cat([pointmap, gt_image_ds, densify_weights_ds], dim=0).permute(1, 2, 0)[densify_weights_ds.squeeze() > 0]  # (N, 7)
        xyz_transformed = coords[:, :3] @ self._transform_matrix[:3, :3].T + self._transform_matrix[:3, -1]
        is_in_bbox = torch.logical_and(
            (xyz_transformed[:, :2] > self._bounding_box.min[:2]).all(dim=-1),
            (xyz_transformed[:, :2] < self._bounding_box.max[:2]).all(dim=-1),
        )
        if not is_in_bbox.any():
            return None
        coords = coords[is_in_bbox]
        return coords

    def _on_train_start(self, gaussian_model, module):
        _enabled = True
        loader = module.trainer.train_dataloader
        if getattr(loader, "max_cache_num", 1) >= 0:
            print("[MeshGuidedDensityController] training dataloader does not cache all images, skip mesh-guided densification")
            _enabled = False
        if not _enabled:
            return

        try:
            xyz = gaussian_model.get_xyz
            self._cached_data = loader.cached
            transforms, bounding_box = module.trainer.datamodule.dataparser_outputs._bounding_box
            transforms = torch.tensor(transforms).to(xyz)
            self._transform_matrix = torch.eye(4)[:3].to(xyz)
            self._transform_matrix[:3, :3] = build_rotation(transforms[:4][None, :])[0].to(xyz)
            self._transform_matrix[:3, 3] = transforms[4:]
            xmin, ymin, xmax, ymax = bounding_box
            min_tensor, max_tensor = torch.tensor([xmin, ymin, 0.0]).to(xyz), torch.tensor([xmax, ymax, 0.0]).to(xyz)
            self._bounding_box = MinMaxBoundingBox(
                min=min_tensor - 0.1 * (max_tensor - min_tensor),
                max=max_tensor + 0.1 * (max_tensor - min_tensor),
            )
            self._scene_extent = module.trainer.datamodule.dataparser_outputs.camera_extent
            self._enable_guided_densify = True
        except:
            self._enable_guided_densify = False

        initialize_from = module.hparams.get("initialize_from", None)
        if initialize_from is None or not osp.exists(initialize_from):
            return
        if not self._enable_guided_densify:
            return

        device = gaussian_model.get_xyz.device
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        optimizers = self._exclude_occupancy_optimizer(module.trainer.optimizers)
        indices = torch.randperm(len(cameras))
        prune_mask = self._get_trimming_prune_mask(cameras, gaussian_model, top_k=self.config.top_k, ratio=self.config.start_trim_ratio)

        renderer = module.renderer
        bg_color = torch.zeros((3,), device=device)
        voxel_grid = SparseVoxelFeatureAccumulator(voxel_size=2e-3 * self._scene_extent, feat_dim=4)
        # if self.config.guided_densify_ratio > 0.0:
        #     for idx in tqdm(range(len(cameras)), desc="Computing depth-guided densification scores", leave=False):
        #         # load camera, gt_image, and gt_depth
        #         camera = cameras[indices[idx]].to_device(device)
        #         _, (image_name, gt_image, _), gt_depth_data = self._cached_data[indices[idx]]
        #         if gt_depth_data is None:
        #             continue
        #         gt_image = gt_image.to(device)
        #         gt_depth_ds: torch.Tensor = gt_depth_data.get(device=device)
        #         depth_mean, depth_std = gt_depth_ds.mean(), gt_depth_ds.std()
        #         clamp_min, clamp_max = depth_mean - 2 * depth_std, depth_mean + 2 * depth_std

        #         # render
        #         outputs = renderer(camera, gaussian_model, bg_color, render_types=["rgb", "depth"])
        #         render = outputs["render"]
        #         gaussian_depth = outputs["median_depth"].squeeze()
        #         gt_depth_shape = (int(gt_depth_data.camera.height), int(gt_depth_data.camera.width))
        #         if gaussian_depth.shape[-2:] != gt_depth_shape:
        #             gt_depth = F.interpolate(
        #                 gt_depth_ds[None, None, ...], size=gaussian_depth.shape[-2:], mode="bilinear", align_corners=True
        #             ).squeeze()
        #         else:
        #             gt_depth = gt_depth_ds
        #         valid_pixels = torch.logical_and(gt_depth > clamp_min, gt_depth < clamp_max)

        #         # compute depth diff
        #         gaussian_depth_clamp, gt_depth_clamp = gaussian_depth.clamp_min(min=clamp_min), gt_depth.clamp_min(min=clamp_min)
        #         diff_depth = gaussian_depth_clamp - gt_depth_clamp
        #         diff_depth_weight = 1 + torch.log(1 + diff_depth.clamp_min(min=0.0) / (self._scene_extent + 1e-8))

        #         # gather coords
        #         camera_ds: Camera = gt_depth_data.camera.to_device(device)
        #         densify_pixels = torch.logical_and(valid_pixels, diff_depth > 2e-3 * self._scene_extent)
        #         densify_weights = diff_depth_weight * densify_pixels
        #         coords = self._gather_valid_coords(camera_ds, densify_weights, gt_image, gt_depth_ds)
        #         if coords is not None:
        #             voxel_grid.update(coords[:, :3], coords[:, 3:])

        #     # build new Gaussians from voxels before trimming at start
        #     n_gaussians = gaussian_model.n_gaussians
        #     xyz, vals, cnt = voxel_grid.finalize(return_count=True)
        #     rgb = (vals[:, :3] / (cnt + 1e-8)).clamp(min=0.0, max=1.0)
        #     intensity = vals[:, 3] / (cnt.squeeze() + 1e-8)
        #     n_samples = min(int(self.config.guided_densify_ratio * n_gaussians), (intensity > 0).sum().item())
        #     new_properties = self._build_gaussians_from_pcd(
        #         xyz=xyz, rgb=rgb, intensity=intensity, n_samples=n_samples, gaussian_model=gaussian_model
        #     )

        # prune points
        self._prune_points(prune_mask, gaussian_model, optimizers)
        self._n_gaussians_prev = gaussian_model.n_gaussians
        # densify
        # if self.config.guided_densify_ratio > 0.0:
        #     new_parameters = Utils.cat_tensors_to_properties(new_properties, gaussian_model, optimizers)
        #     gaussian_model.properties = new_parameters
        #     self._extend_densification_states(len(new_properties["means"]))
        #     torch.cuda.empty_cache()

    @classmethod
    def _compute_loss(cls, a: torch.Tensor, b: torch.Tensor):
        if a.ndim == 3:
            a = a.unsqueeze(0)
        if b.ndim == 3:
            b = b.unsqueeze(0)

        ssim = cls._ssim(a, b).squeeze(0).mean(0)
        return 1.0 - ssim

    @staticmethod
    def _ssim(a: torch.Tensor, b: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        channel = a.size(-3)
        window = create_window(window_size, channel)
        if a.is_cuda:
            window = window.cuda(a.get_device())
        window = window.type_as(a)

        mu1 = F.conv2d(a, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(b, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(a * a, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(b * b, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(a * b, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map

    def _dump_mesh(self, mesh, global_step, output_path):
        import open3d as o3d

        # snapshot path
        snapshot_path = osp.join(output_path, "snapshots")
        os.makedirs(snapshot_path, exist_ok=True)
        # dump mesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.verts.detach().cpu().numpy())
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces.detach().cpu().numpy())
        ply_path = osp.join(snapshot_path, "mesh_step_{:06d}.ply".format(global_step))
        o3d.io.write_triangle_mesh(ply_path, mesh_o3d)

    def _dump_snapshot(self, points, global_step, output_path, scores=None):
        from plyfile import PlyData, PlyElement

        if scores is None:
            scores = torch.ones((points.shape[0])).to(points)

        # snapshot path
        snapshot_path = osp.join(output_path, "snapshots")
        os.makedirs(snapshot_path, exist_ok=True)
        # dump densification scores
        xyz = points.detach().cpu().numpy()
        densification_scores = scores.detach().cpu().numpy()
        vertex_dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("scalar", "f4")])
        vertices = np.empty(xyz.shape[0], dtype=vertex_dtype)
        vertices["x"] = xyz[:, 0]
        vertices["y"] = xyz[:, 1]
        vertices["z"] = xyz[:, 2]
        vertices["scalar"] = densification_scores
        ply_path = osp.join(snapshot_path, "scores_step_{:06d}.ply".format(global_step))
        el = PlyElement.describe(vertices, "vertex")
        PlyData([el]).write(ply_path)

    def _dump_images(self, images: torch.Tensor, global_step: int, output_path: str):
        import torchvision

        # snapshot path
        snapshot_path = osp.join(output_path, "snapshots")
        os.makedirs(snapshot_path, exist_ok=True)
        torchvision.utils.save_image(images, osp.join(snapshot_path, "images_step_{:06d}.png".format(global_step)))

    def _dump_depth(self, depth: torch.Tensor, global_step: int, output_path: str):
        import torchvision

        # snapshot path
        snapshot_path = osp.join(output_path, "snapshots")
        os.makedirs(snapshot_path, exist_ok=True)
        torchvision.utils.save_image(self.depth2invdepth(depth), osp.join(snapshot_path, "depth_step_{:06d}.png".format(global_step)))

    @staticmethod
    def depth2invdepth(depth: torch.Tensor):
        from internal.utils.visualizers import Visualizers

        invdepth = 1.0 / depth.clamp_min(1e-6)
        invdepth = (invdepth - invdepth.min()) / (invdepth.max() - invdepth.min())
        invdepth = Visualizers.float_colormap(invdepth, "inferno")
        return invdepth


def _unit(x: torch.Tensor, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _orth_jitter(dir_vec: torch.Tensor, scale: torch.Tensor):
    """
    dir_vec: (M,3) unit direction
    scale:   (M,1) jitter magnitude
    returns: (M,3) orthogonal noise (approximately) with given magnitude
    """
    noise = torch.randn_like(dir_vec)
    # remove projection onto dir
    proj = (noise * dir_vec).sum(dim=-1, keepdim=True) * dir_vec
    ortho = noise - proj
    ortho = _unit(ortho)
    return ortho * scale
