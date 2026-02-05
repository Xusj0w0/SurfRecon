from dataclasses import dataclass, field
from typing import List

import torch

from internal.density_controllers.density_controller import Utils
from internal.density_controllers.vanilla_density_controller import (
    VanillaDensityController, VanillaDensityControllerImpl)
from internal.models.mip_splatting import (MipSplattingModelMixin,
                                           MipSplattingUtils)
from internal.models.vanilla_gaussian import VanillaGaussianModel

from .trim_density_controller import _get_trimming_prune_mask


@dataclass
class MeshGaussianDensityController(VanillaDensityController):
    start_trim_top_k: int = 5

    start_trim_ratio: float = 0.5

    cull_opacity_threshold: float = field(default=0.05)

    def instantiate(self, *args, **kwargs):
        return MeshGaussianDensityControllerImpl(self)


class MeshGaussianDensityControllerImpl(VanillaDensityControllerImpl):
    config: MeshGaussianDensityController

    def setup(self, stage: str, pl_module) -> None:
        super().setup(stage, pl_module)

        def _trim_on_train_start(gaussian_model, module):
            cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
            optimizers = self._exclude_occupancy_optimizer(module.trainer.optimizers)
            prune_mask = _get_trimming_prune_mask(
                cameras, gaussian_model, top_k=self.config.start_trim_top_k, ratio=self.config.start_trim_ratio
            )
            self._prune_points(prune_mask, gaussian_model, optimizers)
            torch.cuda.empty_cache()

        pl_module.on_train_start_hooks.append(_trim_on_train_start)

    def after_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module):
        _optimizers = self._exclude_occupancy_optimizer(optimizers)
        super().after_backward(outputs, batch, gaussian_model, _optimizers, global_step, pl_module)

        self._recompute_3d_filter(gaussian_model, _optimizers, global_step, pl_module)

    def _recompute_3d_filter(self, gaussian_model, optimizers, global_step, pl_module):
        if not MipSplattingModelMixin._filter_3d_name in gaussian_model.get_property_names():
            return

        # densify step
        if (
            global_step < self.config.densify_until_iter
            and global_step > self.config.densify_from_iter
            and global_step % self.config.densification_interval == 0
        ):
            gaussian_model.compute_3d_filter()
            torch.cuda.empty_cache()

    def _init_state(self, n_gaussians: int, device):
        super()._init_state(n_gaussians=n_gaussians, device=device)

        xyz_gradient_accum_abs = torch.zeros((n_gaussians, 1), device=device)
        xyz_gradient_accum_abs_max = torch.zeros((n_gaussians, 1), device=device)

        self.xyz_gradient_accum_abs: torch.Tensor
        self.xyz_gradient_accum_abs_max: torch.Tensor
        self.register_buffer("xyz_gradient_accum_abs", xyz_gradient_accum_abs, persistent=True)
        self.register_buffer("xyz_gradient_accum_abs_max", xyz_gradient_accum_abs_max, persistent=True)

    def _add_densification_stats(self, grad, update_filter, scale):
        scaled_grad = grad[update_filter, :2]
        scaled_grad_abs = grad[update_filter, 2:]
        if scale is not None:
            scaled_grad = scaled_grad * scale
            scaled_grad_abs = scaled_grad_abs * scale
        grad_norm = torch.norm(scaled_grad, dim=-1, keepdim=True)
        grad_norm_abs = torch.norm(scaled_grad_abs, dim=-1, keepdim=True)

        self.xyz_gradient_accum[update_filter] += grad_norm
        self.xyz_gradient_accum_abs[update_filter] += grad_norm_abs
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(
            self.xyz_gradient_accum_abs_max[update_filter],
            grad_norm_abs,
        )
        self.denom[update_filter] += 1

    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers):
        min_opacity = self.config.cull_opacity_threshold
        prune_extent = self.prune_extent

        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # calculate mean grads abs
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        # densify
        self._densify_and_clone(grads, grads_abs, gaussian_model, optimizers)
        self._densify_and_split(grads, grads_abs, gaussian_model, optimizers)

        # prune
        if self.config.cull_by_max_opacity:
            prune_mask = torch.logical_and(
                gaussian_model.get_opacity_max() >= 0.0,
                gaussian_model.get_opacity_max() < min_opacity,
            )
            gaussian_model.reset_opacity_max()
        else:
            prune_mask = (gaussian_model.get_opacities() < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * prune_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self._prune_points(prune_mask, gaussian_model, optimizers)
        torch.cuda.empty_cache()

    def _densify_and_clone(
        self,
        grads: torch.Tensor,
        grads_abs: torch.Tensor,
        gaussian_model: VanillaGaussianModel,
        optimizers: List[torch.optim.Optimizer],
    ):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        ratio = (torch.norm(grads, dim=-1) >= self.config.densify_grad_threshold).float().mean()
        grad_abs_threshold = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussian_model.get_scales(), dim=1).values <= percent_dense * scene_extent,
        )

        # Copy selected Gaussians
        new_properties = {}
        for key, value in gaussian_model.properties.items():
            new_properties[key] = value[selected_pts_mask]

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

    def _densify_and_split(
        self,
        grads: torch.Tensor,
        grads_abs: torch.Tensor,
        gaussian_model: VanillaGaussianModel,
        optimizers: List[torch.optim.Optimizer],
        N: int = 2,
    ):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        ratio = (torch.norm(grads, dim=-1) >= self.config.densify_grad_threshold).float().mean()
        grad_abs_threshold = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        device = gaussian_model.get_property("means").device
        n_init_points = gaussian_model.n_gaussians
        scales = gaussian_model.get_scales()

        # The number of Gaussians and `grads` is different after cloning, so padding is required
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        padded_grad_abs = torch.zeros((n_init_points,), device=device)
        padded_grad_abs[: grads_abs.shape[0]] = grads_abs.squeeze()

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scales, dim=1).values > percent_dense * scene_extent,
        )

        # Split
        new_properties = self._split_properties(gaussian_model, selected_pts_mask, N)

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

        # Prune selected Gaussians, since they are already split
        prune_filter = torch.cat(
            [
                selected_pts_mask,
                selected_pts_mask.new_zeros(N * selected_pts_mask.sum()),
            ]
        )
        self._prune_points(prune_filter, gaussian_model, optimizers)

    def _prune_points(self, mask, gaussian_model, optimizers):
        super()._prune_points(mask, gaussian_model, optimizers)

        valid_points_mask = ~mask
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]

    def _reset_opacities(self, gaussian_model: VanillaGaussianModel, optimizers: List[torch.optim.Optimizer]):
        if MipSplattingModelMixin._filter_3d_name in gaussian_model.get_property_names():
            current_opacity_with_filter, _ = gaussian_model.get_3d_filtered_scales_and_opacities()
            opacities_new = torch.min(
                current_opacity_with_filter, torch.ones_like(current_opacity_with_filter) * self.config.opacity_reset_value
            )
            _, compensation = MipSplattingUtils.apply_3d_filter_on_scales(
                filter_3d=gaussian_model.get_3d_filter(),
                scales=gaussian_model.get_scales(),
                compute_opacity_compensation=True,
            )
            opacities_new = gaussian_model.opacity_inverse_activation(opacities_new / compensation[..., None])
            new_parameters = Utils.replace_tensors_to_properties(
                tensors={"opacities": opacities_new},
                optimizers=optimizers,
            )
            gaussian_model.update_properties(new_parameters)
        else:
            super()._reset_opacities(gaussian_model=gaussian_model, optimizers=optimizers)

    @staticmethod
    def _exclude_occupancy_optimizer(optimizers):
        _optimizers = []
        for opt in optimizers:
            is_gaussian_params = True
            for group in opt.param_groups:
                if group["name"] == "occupancy":
                    is_gaussian_params = False
                    break
            if is_gaussian_params:
                _optimizers.append(opt)
        return _optimizers
