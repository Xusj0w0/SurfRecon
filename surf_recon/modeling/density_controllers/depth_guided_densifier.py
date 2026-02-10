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

from ...utils.general_utils import init_cdf_mask
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

    gt_depth_guided_until_iter: int = 9000

    guided_densify_ratio: float = 0.2

    guided_prune_ratio: float = 0.05

    depth_tolerance_ratio: float = 0.1

    n_delaunay_gaussians_coarse: int = 300_000

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

        def _load_mesh_and_cached_on_train_start(gaussian_model, module):
            _enabled = True
            if not hasattr(module.renderer, "render_mesh"):
                print("[MeshGuidedDensityController] renderer does not support mesh rendering, skip mesh-guided densification")
                _enabled = False
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

        pl_module.on_train_start_hooks.append(_load_mesh_and_cached_on_train_start)

        self._guided_densify_step = -100

    def after_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module):
        if global_step >= self.config.densify_until_iter:
            return

        _optimizers = self._exclude_occupancy_optimizer(optimizers)
        self._n_gaussians_prev = gaussian_model.n_gaussians
        with torch.no_grad():
            self.update_states(outputs)
            gaussian_changed = False

            # guided densify
            if self._enable_guided_densify and global_step % self.config.depth_guided_densification_interval:
                cameras = pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras
                if global_step > self.config.gt_depth_guided_until_iter:
                    self._depth_guided_densify(gaussian_model, pl_module.renderer, cameras, _optimizers, global_step, pl_module)
                else:
                    self._depth_guided_densify_and_prune(gaussian_model, pl_module.renderer, cameras, _optimizers, global_step, pl_module)
                gaussian_changed = True
                self._guided_densify_step = global_step

            # trim before resetting opacity
            if (
                global_step > self.config.trim_from_iter
                and global_step % self.config.trim_interval == 0
                and (global_step >= self._guided_densify_step + 1000)
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

            if global_step % self.config.opacity_reset_interval == 0 or (
                torch.all(pl_module.background_color == 1.0) and global_step == self.config.densify_from_iter
            ):
                self._reset_opacities(gaussian_model, _optimizers)
                self.opacity_reset_at = global_step

            if gaussian_changed:
                self._recompute_3d_filter(gaussian_model)

    @torch.no_grad()
    def _extract_mesh(self, gaussian_model: VanillaGaussianModel, renderer, cameras: Cameras):
        # Extract mesh
        xyz = gaussian_model.get_xyz
        xyz_transformed = xyz @ self._transform_matrix[:3, :3].T + self._transform_matrix[:3, 3]
        mask = torch.logical_and(
            (xyz_transformed[..., :2] > self._bounding_box.min[:2]).all(dim=-1),
            (xyz_transformed[..., :2] < self._bounding_box.max[:2]).all(dim=-1),
        )
        properties = gaussian_model.properties
        masked_properties = {k: v[mask] for k, v in properties.items()}
        gaussian_model.properties = masked_properties
        mesh = MeshGaussianUtils.post_extract_mesh(
            gaussian_model=gaussian_model,
            renderer=renderer,
            cameras=cameras,
            max_num_delaunay_gaussians=self.config.n_delaunay_gaussians_coarse,
            sdf_n_binary_steps=4,
            without_color=True,
            skip_filtering=True,
        )
        gaussian_model.properties = properties
        return mesh

    @torch.no_grad()
    def _depth_guided_densify_and_prune(
        self,
        gaussian_model: VanillaGaussianModel,
        renderer,
        cameras: Cameras,
        optimizers,
        global_step: int,
        pl_module=None,
    ):
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
            prune_pixels = torch.logical_and(valid_pixels, diff_depth < -2e-3 * self._scene_extent)
            prune_weights = torch.abs(loss * prune_pixels)
            # prune_weights[prune_weights < torch.quantile(prune_weights, 0.5)] = 0.0
            importances = rasterize_importance(camera, gaussian_model, weight_map=prune_weights)
            scores = importances["accum_scaled_weights"] / (importances["num_hit_pixels"] + 1e-5)
            prune_scores += scores

            # gather valid coords
            camera_ds: Camera = gt_depth_data.camera.to_device(device)
            densify_pixels = torch.logical_and(valid_pixels, diff_depth > 2e-3 * self._scene_extent)
            densify_weights = torch.abs(loss * densify_pixels)
            # densify_weights[densify_weights < torch.quantile(densify_weights, 0.5)] = 0.0
            need_to_ds = torch.cat([densify_pixels[None].float(), densify_weights[None], gt_image], dim=0)
            ds = F.interpolate(need_to_ds[None], size=gt_depth_ds.shape[-2:], mode="bilinear", align_corners=True)[0]
            densify_pixels_ds, densify_weights_ds, gt_image_ds = ds[0] > 0.5, ds[1:2], ds[2:]
            pointmap = depth_to_pointmap(gt_depth_ds.unsqueeze(0), camera_ds)
            c2w = torch.linalg.inv(camera_ds.world_to_camera.T)
            pointmap = torch.einsum("ij, jhw -> ihw", c2w, torch.cat([pointmap, torch.ones_like(pointmap[:1])], dim=0))[:3]
            coords = torch.cat([pointmap, gt_image_ds, densify_weights_ds], dim=0).permute(1, 2, 0)[densify_pixels_ds]  # (N, 7)
            xyz_transformed = coords[:, :3] @ self._transform_matrix[:3, :3].T + self._transform_matrix[:3, -1]
            is_in_bbox = torch.logical_and(
                (xyz_transformed[:, :2] > self._bounding_box.min[:2]).all(dim=-1),
                (xyz_transformed[:, :2] < self._bounding_box.max[:2]).all(dim=-1),
            )
            if not is_in_bbox.any():
                continue
            coords = coords[is_in_bbox]
            voxel_grid.update(coords[:, :3], coords[:, 3:])

        # prune with scores
        prune_mask = prune_scores > torch.quantile(prune_scores, 1 - self.config.guided_prune_ratio)
        self._prune_points(prune_mask, gaussian_model, optimizers)
        self._n_gaussians_prev = gaussian_model.n_gaussians  # update previous gaussian num for following grad-based densification

        # densify with coords
        n_gaussians = gaussian_model.n_gaussians
        xyz, vals, cnt = voxel_grid.finalize(return_count=True)
        rgb = (vals[:, :3] / (cnt + 1e-8)).clamp(min=0.0, max=1.0)
        intensity = vals[:, 3]
        n_samples = min(int(self.config.guided_densify_ratio * n_gaussians), (intensity > 0).sum().item())
        self._densify_with_coords(
            xyz=xyz, rgb=rgb, intensity=intensity, n_samples=n_samples, gaussian_model=gaussian_model, optimizers=optimizers
        )

        # Extend optimizer states
        self._extend_densification_states(n_samples)

    @torch.no_grad()
    def _depth_guided_densify(
        self,
        gaussian_model: VanillaGaussianModel,
        renderer,
        cameras: Cameras,
        optimizers,
        global_step: int,
        pl_module=None,
    ):
        indices = torch.randperm(len(cameras))
        # Compute densification scores
        device = gaussian_model.get_xyz.device
        bg_color = torch.zeros((3,), device=device)
        voxel_grid = SparseVoxelFeatureAccumulator(voxel_size=2e-3 * self._scene_extent, feat_dim=4)  # rgb+diff_depth
        prune_scores = torch.zeros((gaussian_model.n_gaussians,), device=device)
        for idx in tqdm(range(len(cameras)), desc="Computing depth-guided densification scores", leave=False):
            # load camera, gt_image, and gt_depth
            camera = cameras[indices[idx]].to_device(device)
            _, (image_name, gt_image, _), _ = self._cached_data[indices[idx]]
            gt_image = gt_image.to(device)

            # render
            outputs = renderer(camera, gaussian_model, bg_color, render_types=["rgb", "depth"])
            render = outputs["render"]
            conf = outputs["mask"]
            gaussian_depth = outputs["median_depth"].squeeze()
            depth_mean, depth_std = gaussian_depth.mean(), gaussian_depth.std()
            clamp_min, clamp_max = depth_mean - 2 * depth_std, depth_mean + 2 * depth_std
            valid_pixels = torch.logical_and(gaussian_depth > clamp_min, gaussian_depth < clamp_max)
            loss = self._compute_loss(render, gt_image)

            # gather valid coords
            densify_weights = torch.abs(loss * conf * valid_pixels)
            pointmap = depth_to_pointmap(gaussian_depth.unsqueeze(0), camera)
            c2w = torch.linalg.inv(camera.world_to_camera.T)
            pointmap = torch.einsum("ij, jhw -> ihw", c2w, torch.cat([pointmap, torch.ones_like(pointmap[:1])], dim=0))[:3]
            coords = torch.cat([pointmap, gt_image, densify_weights[None]], dim=0)
            xyz_transformed = coords[:, :3] @ self._transform_matrix[:3, :3].T + self._transform_matrix[:3, -1]
            is_in_bbox = torch.logical_and(
                (xyz_transformed[:, :2] > self._bounding_box.min[:2]).all(dim=-1),
                (xyz_transformed[:, :2] < self._bounding_box.max[:2]).all(dim=-1),
            )
            if not is_in_bbox.any():
                continue
            coords = coords[is_in_bbox]
            voxel_grid.update(coords[:, :3], coords[:, 3:])

        # densify with coords
        n_gaussians = gaussian_model.n_gaussians
        xyz, vals, cnt = voxel_grid.finalize(return_count=True)
        rgb = (vals[:, :3] / (cnt + 1e-8)).clamp(min=0.0, max=1.0)
        intensity = vals[:, 3]
        n_samples = min(int(self.config.guided_densify_ratio * n_gaussians), (intensity > 0).sum().item())
        self._densify_with_coords(
            xyz=xyz, rgb=rgb, intensity=intensity, n_samples=n_samples, gaussian_model=gaussian_model, optimizers=optimizers
        )

        # Extend optimizer states
        self._extend_densification_states(n_samples)

    def _extend_densification_states(self, n_samples: int):
        device = self.max_radii2D.device
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((n_samples,), device=device)], dim=0)
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((n_samples, 1), device=device)], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros((n_samples, 1), device=device)], dim=0)
        self.xyz_gradient_accum_abs = torch.cat([self.xyz_gradient_accum_abs, torch.zeros((n_samples, 1), device=device)], dim=0)
        self.xyz_gradient_accum_abs_max = torch.cat([self.xyz_gradient_accum_abs_max, torch.zeros((n_samples, 1), device=device)], dim=0)
        torch.cuda.empty_cache()

    @classmethod
    def scatter_to_vertices(
        cls,
        val: torch.Tensor,
        pix_to_face: torch.Tensor,
        bary_coords: torch.Tensor,
        mesh: Meshes,
    ):
        val = val.squeeze()
        assert val.ndim == 2, "loss_hw should be (H, W)"
        pix_to_face = pix_to_face.squeeze()
        assert pix_to_face.ndim == 2, "pix_to_face should be (H, W)"
        bary_coords = bary_coords.squeeze()
        assert bary_coords.ndim == 3 and bary_coords.size(-1) == 3, "bary_coords should be (H, W, 3)"

        device = val.device
        n_verts = mesh.verts.shape[0]

        valid = torch.logical_and(pix_to_face >= 0, (bary_coords > 0).all(dim=-1))
        if valid.sum() == 0:
            vertex_val_sum = torch.zeros((n_verts,), device=device, dtype=val.dtype)
            vertex_w_sum = torch.zeros((n_verts,), device=device, dtype=val.dtype)
            return vertex_val_sum, vertex_w_sum

        face_id_valid = pix_to_face[valid]  # (N,)
        bary_valid = bary_coords[valid]  # (N, 3)
        loss_valid = val[valid]  # (N,)

        tri_vids = mesh.faces[face_id_valid].long().reshape(-1)  # (N*3,)
        contrib = (loss_valid.unsqueeze(-1) * bary_valid).reshape(-1)  # (N*3,)
        vertex_val_sum = scatter_add(contrib, tri_vids, dim=0, dim_size=n_verts)
        vertex_w_sum = scatter_add(bary_valid.reshape(-1), tri_vids, dim=0, dim_size=n_verts)
        return vertex_val_sum, vertex_w_sum

    def _densify_with_vertices(
        self,
        verts: torch.Tensor,
        scores: torch.Tensor,
        n_samples: int,
        gaussian_model: VanillaGaussianModel,
        optimizers: list,
    ):
        device = verts.device

        prob = scores / scores.sum()
        indices = np.random.choice(len(prob), size=n_samples, p=prob.detach().cpu().numpy(), replace=False)
        indices = torch.from_numpy(indices).to(device)
        new_means = verts[indices]

        knn = knn_points(new_means[None], gaussian_model.get_xyz[None], K=1)
        gaussian_ids = knn.idx[0, :, 0]  # (n_samples,)
        new_properties = {}
        for key, value in gaussian_model.properties.items():
            if key == "means":
                new_properties[key] = new_means
            else:
                new_properties[key] = value[gaussian_ids]

        new_parameters = Utils.cat_tensors_to_properties(new_properties, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

    def _densify_with_coords(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        intensity: torch.Tensor,
        n_samples: int,
        gaussian_model: VanillaGaussianModel,
        optimizers,
    ):
        from internal.utils.general_utils import inverse_sigmoid
        from internal.utils.sh_utils import RGB2SH

        device = xyz.device
        prob = intensity / intensity.sum()
        indices = np.random.choice(len(prob), size=n_samples, p=prob.detach().cpu().numpy(), replace=False)
        indices = torch.from_numpy(indices).to(device)

        means, color = xyz[indices], rgb[indices]
        _points = means.unsqueeze(0)
        knn = knn_points(_points, _points, K=2)
        dist = torch.sqrt(knn.dists[0, :, 1]).clamp(min=1e-7, max=1e-4 * self._scene_extent)
        scales = torch.log(dist)[..., None].repeat(1, 3).to(means)
        rots = torch.zeros((n_samples, 4)).to(means)
        rots[:, 0] = 1.0
        opacities = inverse_sigmoid(0.9 * torch.ones((n_samples, 1))).to(means)
        shs_dc = RGB2SH(color).unsqueeze(1).to(means)
        shs_rest = torch.zeros((n_samples, (gaussian_model.max_sh_degree + 1) ** 2 - 1, 3)).to(means)

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
                torch.zeros((n_samples, 1)).to(means), requires_grad=False
            )

        new_parameters = Utils.cat_tensors_to_properties(new_properties, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

    @classmethod
    def _compute_loss(cls, a: torch.Tensor, b: torch.Tensor):
        if a.ndim == 3:
            a = a.unsqueeze(0)
        if b.ndim == 3:
            b = b.unsqueeze(0)

        ssim = cls._ssim(a, b).squeeze(0).mean(0)
        return 1.0 - ssim
        rgb_diff = F.l1_loss(a, b, reduction="none").squeeze(0).mean(0)

        lambda_dssim = 0.2
        return lambda_dssim * (1.0 - ssim) + (1.0 - lambda_dssim) * rgb_diff

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
