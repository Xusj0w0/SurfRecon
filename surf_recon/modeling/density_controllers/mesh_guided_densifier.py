import os
import os.path as osp
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.ops import knn_points
from torch_scatter import scatter_add
from tqdm import tqdm

from internal.cameras import Camera, Cameras
from internal.models.mip_splatting import MipSplattingModelMixin
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.ssim import create_window

from ...utils.mesh import Meshes
from ..renderers.nvdr import NVDRRendererMixin
from .mesh_gaussian_densifier import (MeshGaussianDensityController,
                                      MeshGaussianDensityControllerImpl)

DEBUG = True


@dataclass
class MeshGuidedDensityController(MeshGaussianDensityController):
    mesh_path: str = ""

    guided_densify_from_iter: int = 5000

    guided_densification_interval: int = 3000

    guided_densify_ratio: float = 0.2

    def instantiate(self, *args, **kwargs):
        return MeshGuidedDensityControllerImpl(self)


class MeshGuidedDensityControllerImpl(MeshGaussianDensityControllerImpl):
    config: MeshGuidedDensityController

    def setup(self, stage: str, pl_module) -> None:
        super().setup(stage, pl_module)

        self._enable_guided_densify = False
        self._reference_mesh: Optional[Meshes] = None
        self._cached_data: Optional[list] = None

        def _load_mesh_and_cached_on_train_start(gaussian_model, module):
            _enabled = True
            if not osp.exists(self.config.mesh_path):
                print("[MeshGuidedDensityController] mesh_path does not exist, skip mesh-guided densification")
                _enabled = False
            if not hasattr(pl_module.renderer, "render_mesh"):
                print("[MeshGuidedDensityController] renderer does not support mesh rendering, skip mesh-guided densification")
                _enabled = False
            loader = pl_module.trainer.train_dataloader
            if getattr(loader, "max_cache_num", 1) >= 0:
                print("[MeshGuidedDensityController] training dataloader does not cache all images, skip mesh-guided densification")
                _enabled = False
            if not _enabled:
                return

            device = gaussian_model.get_xyz.device
            try:
                m: trimesh.Trimesh = trimesh.load(self.config.mesh_path, process=False)
                verts = torch.from_numpy(np.asarray(m.vertices, dtype=np.float32)).to(device)
                faces = torch.from_numpy(np.asarray(m.faces, dtype=np.int32)).to(device)
                self._reference_mesh = Meshes(verts=verts, faces=faces)
                self._cached_data = loader.cached
                self._enable_guided_densify = True
            except:
                self._enable_guided_densify = False

        pl_module.on_train_start_hooks.append(_load_mesh_and_cached_on_train_start)

    def after_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module):
        if global_step >= self.config.densify_until_iter:
            return

        _optimizers = self._exclude_occupancy_optimizer(optimizers)
        with torch.no_grad():
            self.update_states(outputs)

            # guided densify
            if self._enable_guided_densify:
                cameras = pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras
                self._mesh_guided_densify(gaussian_model, pl_module.renderer, cameras, _optimizers, global_step)

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=_optimizers,
                )

            if global_step % self.config.opacity_reset_interval == 0 or (
                torch.all(pl_module.background_color == 1.0) and global_step == self.config.densify_from_iter
            ):
                self._reset_opacities(gaussian_model, _optimizers)
                self.opacity_reset_at = global_step

        self._recompute_3d_filter(gaussian_model, optimizers, global_step, pl_module)

    @torch.no_grad()
    def _mesh_guided_densify(
        self, gaussian_model: VanillaGaussianModel, renderer: NVDRRendererMixin, cameras: Cameras, optimizers, global_step: int
    ):
        if not self._is_guided_densify_step(global_step):
            return

        device = gaussian_model.get_xyz.device
        n_verts, n_faces = self._reference_mesh.verts.shape[0], self._reference_mesh.faces.shape[0]
        scores = torch.zeros((n_verts,), device=device)
        weights = torch.zeros_like(scores)
        bg_color = torch.zeros((3,), device=device)
        for idx in tqdm(range(len(cameras)), desc="Computing mesh-guided densification scores", leave=False):
            camera = cameras[idx].to_device(device)
            gt_image = self._cached_data[idx][1][1]
            outputs = renderer(camera, gaussian_model, bg_color, render_types=["rgb"])
            _mesh = renderer.cull_mesh(self._reference_mesh, camera)
            rast_out, _ = renderer.nvdiff_rasterization(camera, _mesh.verts, _mesh.faces)

            render = outputs["render"]
            loss = self._compute_loss(render, gt_image.to(render))  # (H, W)
            # TODO: use outputs["mask"] for sky remove (especially in street scene)
            _scores, _weights = self.scatter_to_vertices(loss, rast_out.pix_to_face, rast_out.bary_coords, _mesh)
            scores += _scores
            weights += _weights

        scores = scores / (weights + 1e-8)
        mask = scores < torch.quantile(scores, 0.1)
        scores[mask] = 0.0
        prob = scores / scores.sum()

        n_samples = int(self.config.guided_densify_ratio * gaussian_model.n_gaussians)
        indices = np.random.choice(len(prob), size=n_samples, p=prob.detach().cpu().numpy(), replace=False)
        indices = torch.from_numpy(indices).to(device)
        new_means = self._reference_mesh.verts[indices]

        knn = knn_points(new_means[None], gaussian_model.get_xyz[None], K=1)
        gaussian_ids = knn.idx[0, :, 0]  # (n_samples,)
        new_properties = {}
        for key, value in gaussian_model.properties.items():
            if key == "means":
                new_properties[key] = new_means
            else:
                new_properties[key] = value[gaussian_ids]
        self._densification_postfix(new_properties, gaussian_model, optimizers)

        if DEBUG:
            from plyfile import PlyData, PlyElement

            xyz = self._reference_mesh.verts.detach().cpu().numpy()
            intensity = scores.detach().cpu().numpy()
            vertex_dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("scalar", "f4")])
            vertices = np.empty(xyz.shape[0], dtype=vertex_dtype)
            vertices["x"] = xyz[:, 0]
            vertices["y"] = xyz[:, 1]
            vertices["z"] = xyz[:, 2]
            vertices["scalar"] = intensity

            ply_path = "tmp/mesh_guided_intensity/step_{:05d}.ply".format(global_step)
            os.makedirs(osp.dirname(ply_path), exist_ok=True)
            el = PlyElement.describe(vertices, "vertex")
            PlyData([el]).write(ply_path)

    def _recompute_3d_filter(self, gaussian_model, optimizers, global_step, pl_module):
        if not MipSplattingModelMixin._filter_3d_name in gaussian_model.get_property_names():
            return

        # densify step
        is_densify_step = (
            global_step < self.config.densify_until_iter
            and global_step > self.config.densify_from_iter
            and global_step % self.config.densification_interval == 0
        )
        # guided densify step
        is_guided_densify_step = self._is_guided_densify_step(global_step)
        if is_densify_step or is_guided_densify_step:
            gaussian_model.compute_3d_filter()
            torch.cuda.empty_cache()

    def _is_guided_densify_step(self, global_step: int) -> bool:
        if (
            global_step < self.config.densify_until_iter
            and global_step >= self.config.guided_densify_from_iter
            and global_step % self.config.guided_densification_interval == 0
        ):
            return True
        return False

    @classmethod
    def scatter_to_vertices(cls, val: torch.Tensor, pix_to_face: torch.Tensor, bary_coords: torch.Tensor, mesh: Meshes):
        val = val.squeeze()
        assert val.ndim == 2, "loss_hw should be (H, W)"
        pix_to_face = pix_to_face.squeeze()
        assert pix_to_face.ndim == 2, "pix_to_face should be (H, W)"
        bary_coords = bary_coords.squeeze()
        assert bary_coords.ndim == 3 and bary_coords.size(-1) == 3, "bary_coords should be (H, W, 3)"

        device = val.device
        n_verts = mesh.verts.shape[0]

        valid = pix_to_face >= 0
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

    @classmethod
    def _compute_loss(cls, a: torch.Tensor, b: torch.Tensor):
        if a.ndim == 3:
            a = a.unsqueeze(0)
        if b.ndim == 3:
            b = b.unsqueeze(0)

        ssim = cls._ssim(a, b).squeeze(0).mean(0)
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
