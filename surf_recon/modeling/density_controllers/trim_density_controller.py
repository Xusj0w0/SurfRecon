from dataclasses import dataclass, field
from typing import List

import torch
from tqdm import tqdm

from internal.cameras import Camera, Cameras
from internal.dataset import CacheDataLoader
from internal.density_controllers.vanilla_density_controller import (
    VanillaDensityController, VanillaDensityControllerImpl)
from internal.models.mip_splatting import MipSplattingModelMixin
from internal.models.vanilla_gaussian import VanillaGaussianModel

from ..renderers.importance import rasterize_importance


@dataclass
class TrimDensityControllerMixin:
    top_k: int = 5

    trim_ratio: float = 0.1

    trim_from_iter: int = 1000

    trim_interval: int = 500


def _get_trimming_prune_mask(cameras: Cameras, gaussian_model: VanillaGaussianModel, top_k: int = 5, ratio: float = 0.1):
    device = gaussian_model.get_xyz.device
    # Create a score matrix to store top-k scores for each gaussian
    top_k_scores = torch.zeros((gaussian_model.n_gaussians, top_k), device=device)
    min_indices = top_k_scores.argmin(dim=-1, keepdim=True)  # [N_gaussians, 1]

    for idx in tqdm(range(len(cameras)), desc="Computing trimming scores", leave=False):
        camera = cameras[idx].to_device(device)
        importances = rasterize_importance(viewpoint_camera=camera, pc=gaussian_model)
        scores = importances["accum_weights"] / (importances["num_hit_pixels"] + 1e-5)
        min_scores = top_k_scores.gather(dim=-1, index=min_indices).squeeze()  # [N_gaussians]
        update_mask = scores > min_scores  # [N_gaussians]
        # Update top-k scores and min_indices
        if update_mask.sum() > 0:
            rows = update_mask.nonzero(as_tuple=True)[0]
            cols = min_indices[update_mask, 0]
            top_k_scores[rows, cols] = scores[rows]
            min_indices[rows, 0] = top_k_scores[rows].argmin(dim=-1)

    # Determine the threshold to trim gaussians
    avg_scores = top_k_scores.mean(dim=-1)
    thresh = torch.quantile(avg_scores, ratio)
    prune_mask = avg_scores < thresh

    return prune_mask


@dataclass
class TrimDensityController(VanillaDensityController, TrimDensityControllerMixin):
    def instantiate(self, *args, **kwargs):
        return TrimDensityControllerImpl(self)


class TrimDensityControllerImpl(VanillaDensityControllerImpl):
    config: TrimDensityController

    def after_backward(
        self, outputs: dict, batch, gaussian_model: VanillaGaussianModel, optimizers: List, global_step: int, pl_module
    ) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        with torch.no_grad():
            self.update_states(outputs)

            # trim before resetting opacity
            if global_step > self.config.trim_from_iter and global_step % self.config.trim_interval == 0:
                cameras = pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras
                prune_mask = _get_trimming_prune_mask(
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
