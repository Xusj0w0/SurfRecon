from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from internal.cameras import Camera, Cameras
from internal.dataset import CacheDataLoader
from internal.density_controllers.vanilla_density_controller import (
    VanillaDensityController, VanillaDensityControllerImpl)
from internal.models.mip_splatting import MipSplattingModelMixin
from internal.models.vanilla_gaussian import VanillaGaussianModel

from ...utils.general_utils import TopKTracker
from ..renderers.importance import rasterize_importance


@dataclass
class TrimDensityControllerMixin:
    top_k: int = 5

    trim_ratio: float = 0.05

    trim_from_iter: int = 3000

    trim_interval: int = 500


class TrimDensityControllerImplMixin:
    def _get_trimming_prune_mask(
        self,
        cameras: Cameras,
        gaussian_model: VanillaGaussianModel,
        top_k: int = 5,
        ratio: float = 0.1,
    ):
        device = gaussian_model.get_xyz.device

        top_k_tracker = TopKTracker(n_samples=gaussian_model.n_gaussians, k=top_k, device=device)

        for idx in tqdm(range(len(cameras)), desc="Computing trimming scores", leave=False):
            camera = cameras[idx].to_device(device)
            importances = rasterize_importance(viewpoint_camera=camera, pc=gaussian_model)
            scores = importances["accum_weights"] / (importances["num_hit_pixels"] + 1e-5)
            top_k_tracker.update(scores)

        # Determine the threshold to trim gaussians
        avg_scores = top_k_tracker.means
        thresh = torch.quantile(avg_scores, ratio)
        prune_mask = avg_scores <= thresh

        return prune_mask

    def _depth_guided_trimming(self, gaussian_model: VanillaGaussianModel, renderer, ratio: float = 0.1):
        device = gaussian_model.get_xyz.device
        bg_color = torch.zeros((3,), device=device)
        prune_scores = torch.zeros((gaussian_model.n_gaussians,), device=device)
        for idx in tqdm(range(len(self._cached_data)), total=len(self._cached_data), desc="Depth-guided trimming", leave=False):
            camera, (image_name, gt_image, _), gt_depth_data = self._cached_data[idx]
            if gt_depth_data is None:
                continue

            camera = camera.to_device(device)
            gt_depth_ds: torch.Tensor = gt_depth_data.get(device)
            depth_mean, depth_std = gt_depth_ds.mean(), gt_depth_ds.std()
            clamp_min, clamp_max = torch.quantile(gt_depth_ds, 0.5), torch.quantile(gt_depth_ds, 0.9)
            clamp_min = max(clamp_min.item(), 0.01)
            # render
            outputs = renderer(camera, gaussian_model, bg_color, render_types=["rgb", "depth"])
            gaussian_depth = outputs["expected_depth"].squeeze()
            gt_depth_shape = (int(gt_depth_data.camera.height), int(gt_depth_data.camera.width))
            if gaussian_depth.shape[-2:] != gt_depth_shape:
                gt_depth = F.interpolate(
                    gt_depth_ds[None, None, ...], size=gaussian_depth.shape[-2:], mode="bilinear", align_corners=True
                ).squeeze()
            else:
                gt_depth = gt_depth_ds
            valid_pixels = gt_depth > clamp_min  # & (gt_depth < clamp_max)

            # compute depth diff
            gaussian_depth_clamp, gt_depth_clamp = gaussian_depth.clamp_min(min=clamp_min), gt_depth.clamp_min(min=clamp_min)
            diff_depth = gaussian_depth_clamp - gt_depth_clamp
            valid_pixels = valid_pixels & (diff_depth < 0.0)
            prune_weights = diff_depth.abs() * valid_pixels
            importances = rasterize_importance(camera, gaussian_model, weight_map=prune_weights)
            scores = importances["accum_weights"]  # / (importances["num_hit_pixels"] + 1e-5)
            prune_scores = torch.max(prune_scores, scores)

        # Determine the threshold to trim gaussians
        thresh = torch.quantile(prune_scores, 1 - ratio)
        prune_mask = prune_scores > thresh
        return prune_mask


@dataclass
class TrimDensityController(VanillaDensityController, TrimDensityControllerMixin):
    def instantiate(self, *args, **kwargs):
        return TrimDensityControllerImpl(self)


class TrimDensityControllerImpl(VanillaDensityControllerImpl, TrimDensityControllerImplMixin):
    config: TrimDensityController

    def after_backward(
        self,
        outputs: dict,
        batch,
        gaussian_model: VanillaGaussianModel,
        optimizers: List,
        global_step: int,
        pl_module,
    ) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        with torch.no_grad():
            self.update_states(outputs)

            # trim before resetting opacity
            if global_step > self.config.trim_from_iter and global_step % self.config.trim_interval == 0:
                cameras = pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras
                prune_mask = self._get_trimming_prune_mask(
                    cameras=cameras,
                    gaussian_model=gaussian_model,
                    top_k=self.config.top_k,
                    ratio=self.config.trim_ratio,
                )
                self._prune_points(prune_mask, gaussian_model, optimizers)
                torch.cuda.empty_cache()

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=optimizers,
                )

            if global_step % self.config.opacity_reset_interval == 0 or (
                torch.all(pl_module.background_color == 1.0) and global_step == self.config.densify_from_iter
            ):
                self._reset_opacities(gaussian_model, optimizers)
                self.opacity_reset_at = global_step


@dataclass
class DepthGuidedTrimDensityController(TrimDensityController):
    def instantiate(self, *args, **kwargs):
        return DepthGuidedTrimDensityControllerImpl(self)


class DepthGuidedTrimDensityControllerImpl(TrimDensityControllerImpl):
    def setup(self, stage: str, pl_module=None):
        super().setup(stage, pl_module)

        self._enable_guided_trim = False
        pl_module.on_train_start_hooks.append(self._on_train_start)

    def after_backward(
        self,
        outputs: dict,
        batch,
        gaussian_model: VanillaGaussianModel,
        optimizers: List,
        global_step: int,
        pl_module,
    ) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        with torch.no_grad():
            self.update_states(outputs)

            # trim before resetting opacity
            if (
                self._enable_guided_trim
                and global_step > self.config.trim_from_iter
                and global_step % self.config.trim_interval == 0
                and global_step >= self.opacity_reset_at + self.config.densification_interval
            ):
                prune_mask = self._depth_guided_trimming(gaussian_model, pl_module.renderer, ratio=self.config.trim_ratio)
                self._prune_points(prune_mask, gaussian_model, optimizers)
                torch.cuda.empty_cache()

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=optimizers,
                )

            if global_step % self.config.opacity_reset_interval == 0 or (
                torch.all(pl_module.background_color == 1.0) and global_step == self.config.densify_from_iter
            ):
                self._reset_opacities(gaussian_model, optimizers)
                self.opacity_reset_at = global_step

    def _on_train_start(self, gaussian_model, module):
        loader = module.trainer.train_dataloader
        if getattr(loader, "max_cache_num", 1) >= 0:
            print("[MeshGuidedDensityController] training dataloader does not cache all images, skip mesh-guided densification")
            return

        self._cached_data = loader.cached
        self._scene_extent = module.trainer.datamodule.dataparser_outputs.camera_extent
        self._enable_guided_trim = True
