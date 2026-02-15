import math
from dataclasses import dataclass, field
from typing import Literal, Union

import lightning
import torch
import torch.nn.functional as F

from internal.cameras import Camera, Cameras
from internal.metrics.metric import Metric
from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl
from internal.utils.visualizers import Visualizers

from ...utils.graphic_utils import depth_to_normal
from ...utils.weight_scheduler import WeightScheduler


@dataclass
class InvDepthRegularizedMetricMixin:
    dn_from_iter: int = 3_000
    "Depth-Normal consistency regularization start iteration"

    lambda_dn: float = 0.05

    depth_loss_type: Literal["l1", "l1+ssim", "l2", "kl"] = "l1"

    depth_loss_ssim_weight: float = 0.2

    depth_loss_weight: WeightScheduler = field(default_factory=lambda: WeightScheduler())

    depth_normalized: bool = False

    depth_map_key: str = "median_depth"


class InvDepthRegularizedMetricMixinImpl:
    config: InvDepthRegularizedMetricMixin

    def get_inverse_depth_metric(self, batch, outputs):
        # TODO: apply mask

        camera, _, gt_inverse_depth_data = batch
        device = camera.device

        if gt_inverse_depth_data is None:
            return torch.tensor(0.0, device=camera.device)

        gt_inverse_depth = gt_inverse_depth_data.get(device=device)

        predicted_inverse_depth = 1.0 / outputs[self.config.depth_map_key].clamp_min(0.01).squeeze()  # znear
        if self.config.depth_normalized:
            clamp_val = (predicted_inverse_depth.mean() + 2 * predicted_inverse_depth.std()).item()
            predicted_inverse_depth = predicted_inverse_depth.clamp(max=clamp_val) / clamp_val
            gt_inverse_depth = gt_inverse_depth.clamp(max=clamp_val) / clamp_val

        gt_shape = (
            int(gt_inverse_depth_data.camera.height),
            int(gt_inverse_depth_data.camera.width),
        )
        pred_shape = predicted_inverse_depth.shape[-2:]
        if pred_shape != gt_shape:
            gt_inverse_depth = F.interpolate(
                gt_inverse_depth[None, None, ...],
                size=pred_shape,
                mode="bilinear",
                align_corners=True,
            ).squeeze()

        return self._get_inverse_depth_loss(predicted_inverse_depth, gt_inverse_depth)

    def get_dreg_weight(self, step: int):
        return self.config.depth_loss_weight.init * (
            self.config.depth_loss_weight.final_factor ** min(step / self.config.depth_loss_weight.max_steps, 1)
        )

    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)

        if self.config.depth_loss_type == "l1":
            self._get_inverse_depth_loss = self._depth_l1_loss
        elif self.config.depth_loss_type == "l1+ssim":
            # self.depth_ssim = StructuralSimilarityIndexMeasure()
            self.depth_ssim = self._depth_ssim
            self._get_inverse_depth_loss = self._depth_l1_and_ssim_loss
        elif self.config.depth_loss_type == "l2":
            self._get_inverse_depth_loss = self._depth_l2_loss
        # elif self.config.depth_loss_type == "kl":
        #     self._get_inverse_depth_loss = self._depth_kl_loss
        else:
            raise NotImplementedError()
        if pl_module is not None:
            self._scene_extent = pl_module.trainer.datamodule.dataparser_outputs.camera_extent
            if stage == "fit":
                if self.config.depth_loss_weight.max_steps is None:
                    self.config.depth_loss_weight.max_steps = pl_module.trainer.max_steps


    def _depth_l1_loss(self, a, b):
        return torch.abs(a - b).mean()

    def _depth_l1_and_ssim_loss(self, a, b):
        l1_loss = self._depth_l1_loss(a, b)
        # ssim_metric = self.depth_ssim(a[None, None, ...], b[None, None, ...])
        ssim_metric = self.depth_ssim(a, b)

        return (1 - self.config.depth_loss_ssim_weight) * l1_loss + self.config.depth_loss_ssim_weight * (1 - ssim_metric)

    def _depth_l2_loss(self, a, b):
        return ((a - b) ** 2).mean()

    def _depth_kl_loss(self, a, b):
        pass

    def _depth_ssim(self, a, b):
        from internal.utils.ssim import ssim

        return ssim(a[None], b[None])


@dataclass
class InvDepthRegularizedMetrics(InvDepthRegularizedMetricMixin, VanillaMetrics):
    def instantiate(self, *args, **kwargs):
        return InvDepthRegularizedMetricsImpl(self)


class InvDepthRegularizedMetricsImpl(InvDepthRegularizedMetricMixinImpl, VanillaMetricsImpl):
    config: InvDepthRegularizedMetrics

    def get_train_metrics(self, pl_module, gaussian_model, step, batch, outputs):
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)

        d_reg_weight = self.get_dreg_weight(step)
        dreg = self.get_inverse_depth_metric(batch, outputs)
        metrics["inv_depth_reg"] = dreg
        pbar["inv_depth_reg"] = True
        metrics["loss"] += d_reg_weight * dreg

        loss_dn = torch.tensor(0.0).to(metrics["loss"])
        if step > self.config.dn_from_iter and self.config.lambda_dn > 0.0:
            loss_dn = self._compute_depth_normal_loss(batch, outputs)
        metrics["loss_dn"] = loss_dn
        pbar["loss_dn"] = False
        metrics["loss"] += self.config.lambda_dn * loss_dn

        return metrics, pbar

    def _compute_depth_normal_loss(self, batch, outputs):
        depth_median = outputs.get("median_depth", None)
        depth_expected = outputs.get("expected_depth", None)
        normal = outputs.get("normal", None)  # (3, H, W) in camera coordinate

        camera, *_ = batch
        loss = torch.tensor(0.0, device=camera.device)
        if all(m is not None for m in [depth_median, depth_expected, normal]):
            normal_from_median_depth = depth_to_normal(depth_median, camera)
            error_map_median = 1.0 - (normal * normal_from_median_depth).sum(0)
            normal_from_expected_depth = depth_to_normal(depth_expected, camera)
            error_map_expected = 1.0 - (normal * normal_from_expected_depth).sum(0)
            ratio = 0.6
            loss = (1.0 - ratio) * error_map_median.mean() + ratio * error_map_expected.mean()
        return loss

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs):
        metrics, pbar = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)

        d_reg_weight = self.get_dreg_weight(pl_module.trainer.global_step)
        dreg = self.get_inverse_depth_metric(batch, outputs)

        metrics["inv_depth_reg"] = dreg
        pbar["inv_depth_reg"] = True
        metrics["loss"] += d_reg_weight * dreg

        return metrics, pbar
