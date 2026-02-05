import math
from dataclasses import dataclass, field

import lightning
import torch
import torch.nn.functional as F

from internal.cameras import Camera, Cameras
from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl
from internal.utils.visualizers import Visualizers


@dataclass
class MeshRegularizationSchedule:
    start_iter: int = 20_000
    "Mesh regularization start iteration"

    mesh_update_interval: int = 1

    reset_delaunay_interval: int = -1
    "Reset delaunay gaussian samples every N iters; set to -1 to disable resetting"

    reset_occupancy_interval: int = 500
    "Reset base occupancies of tetrahedra vertices every N iters"

    reset_occupancy_stop_iter: int = 25_000
    "Stop resetting occupancy from this iter"

    reset_occupancy_label_interval: int = 200
    "Reset occupancy labels of tetrahedra vertices every N iters"

    reset_tetrahedralization_interval: int = 500
    "Reset tetrahedralization every N iters"


@dataclass
class MeshRegularizedMetrics(VanillaMetrics):
    fused_ssim: bool = field(default=True)

    dn_from_iter: int = 3_000
    "Depth-Normal consistency regularization start iteration"

    lambda_dn: float = 0.05

    median_fusing_ratio: float = 1.0
    """
    If both median_depth and expected_depth are rendered,
    normal errors from median_depth are scaled by alpha,
    while that of expected_depth are scaled by (1-alpha)
    """

    mesh_regularization_schedule: MeshRegularizationSchedule = field(default_factory=lambda: MeshRegularizationSchedule())

    use_occupancy_label_loss: bool = True
    "Whether to compute bce loss between occupancy and its label (evaluated by mesh)"

    lambda_mesh_depth: float = 0.05
    "Weighting coefficient of loss between gaussian depth and mesh depth"

    lambda_mesh_normal: float = 0.05
    "Weighting coefficient of loss between gaussian normal and mesh normal"

    lambda_occupancy_label: float = 0.0

    lambda_center_isosurface: float = 0.0

    def instantiate(self, *args, **kwargs):
        return MeshRegularizedMetricsImpl(self)


class MeshRegularizedMetricsImpl(VanillaMetricsImpl):
    config: MeshRegularizedMetrics

    def get_train_metrics(self, pl_module, gaussian_model, step, batch, outputs):
        # basic metrics
        rgb_diff_loss, ssim_metric, rgb_diff_loss_aug = self._compute_basic_metrics(batch, outputs)
        if rgb_diff_loss_aug is not None:
            loss = (1.0 - self.lambda_dssim) * rgb_diff_loss_aug + self.lambda_dssim * (1.0 - ssim_metric)
        else:
            loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1.0 - ssim_metric)
            rgb_diff_loss_aug = torch.tensor(0.0).to(loss)
        metrics = {"loss": loss, "rgb_diff": rgb_diff_loss, "ssim": ssim_metric, "rgb_diff_aug": rgb_diff_loss_aug}
        pbar = {"loss": True, "rgb_diff": True, "ssim": True, "rgb_diff_aug": True}

        # mesh metrics
        _metrics, _pbar, loss = self._compute_extra_metrics(pl_module, gaussian_model, batch, outputs)
        metrics.update(_metrics)
        pbar.update(_pbar)
        metrics["loss"] += loss

        return metrics, pbar

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs):
        # basic metrics
        rgb_diff_loss, ssim_metric, rgb_diff_loss_aug = self._compute_basic_metrics(batch, outputs)
        loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1.0 - ssim_metric)
        rgb_diff_loss_aug = torch.tensor(0.0).to(loss)
        metrics = {"loss": loss, "rgb_diff": rgb_diff_loss, "ssim": ssim_metric, "rgb_diff_aug": rgb_diff_loss_aug}
        pbar = {"loss": True, "rgb_diff": True, "ssim": True, "rgb_diff_aug": False}

        # mesh metrics
        _metrics, _pbar, loss = self._compute_extra_metrics(pl_module, gaussian_model, batch, outputs)
        metrics.update(_metrics)
        pbar.update(_pbar)
        metrics["loss"] += loss

        # PSNR and LPIPS
        camera, image_info, _ = batch
        image_name, gt_image, _ = image_info
        metrics["psnr"] = self.psnr(outputs["render"], gt_image)
        pbar["psnr"] = True
        metrics["lpips"] = self.no_state_dict_models["lpips"](outputs["render"].clamp(0.0, 1.0).unsqueeze(0), gt_image.unsqueeze(0))
        pbar["lpips"] = True

        return metrics, pbar

    def _compute_basic_metrics(self, batch, outputs):
        # Get batch info
        camera, image_info, _ = batch
        image_name, gt_image, masked_pixels = image_info
        render, render_aug = outputs["render"], outputs["render_aug"]
        if masked_pixels is not None:
            masked_pixels = masked_pixels.to(torch.uint8)
            gt_image = gt_image * masked_pixels
            render = render * masked_pixels
            if render_aug is not None:
                render_aug = render_aug * masked_pixels
        rgb_diff_loss = self.rgb_diff_loss_fn(render, gt_image)
        rgb_diff_loss_aug = None
        if render_aug is not None:
            rgb_diff_loss_aug = self.rgb_diff_loss_fn(render_aug, gt_image)
        ssim_metric = self.ssim(render, gt_image)
        return rgb_diff_loss, ssim_metric, rgb_diff_loss_aug

    def _compute_extra_metrics(self, pl_module, gaussian_model, batch, outputs):
        metrics, pbar = {}, {}
        loss = torch.tensor(0.0).to(gaussian_model.get_xyz)

        # Get batch info
        camera, image_info, _ = batch
        image_name, gt_image, masked_pixels = image_info
        global_step: int = pl_module.trainer.global_step + 1  # start from 1
        spatial_scale = pl_module.trainer.datamodule.dataparser_outputs.camera_extent

        # Set intermediates
        normal_from_median_depth = None  # (3, H, W) in camera coordinate
        normal_from_expected_depth = None  # (3, H, W) in camera coordinate
        depth_gaussian = None  # (H, W)
        normal_gaussian = None  # (3, H, W) in camera coordinate
        mesh_valid_mask = None  # (H, W)

        # Depth-Normal consistency regularization
        depth_median = outputs.get("median_depth", None)
        depth_expected = outputs.get("expected_depth", None)
        normal = outputs.get("normal", None)  # (3, H, W) in camera coordinate
        loss_dn = torch.tensor(0.0).to(loss)
        if (
            global_step > self.config.dn_from_iter
            and self.config.lambda_dn > 0.0
            and all(m is not None for m in [depth_median, depth_expected, normal])
        ):
            if normal_from_median_depth is None:
                normal_from_median_depth = self.depth_to_normal(depth_median, camera)
            if normal_from_expected_depth is None:
                normal_from_expected_depth = self.depth_to_normal(depth_expected, camera)
            error_map_median = 1.0 - (normal * normal_from_median_depth).sum(0)
            error_map_expected = 1.0 - (normal * normal_from_expected_depth).sum(0)
            ratio = 0.6
            loss_dn = (1.0 - ratio) * error_map_median.mean() + ratio * error_map_expected.mean()

        metrics["loss_dn"] = loss_dn
        pbar["loss_dn"] = False
        loss += self.config.lambda_dn * loss_dn

        # Mesh regularization
        depth_median = outputs.get("median_depth", None)
        depth_expected = outputs.get("expected_depth", None)
        normal = outputs.get("normal", None)  # (3, H, W) in camera coordinate
        if global_step > self.config.mesh_regularization_schedule.start_iter and all(
            m is not None for m in [depth_median, depth_expected, normal]
        ):
            # Mesh depth regularization
            depth_mesh = outputs.get("mesh_depth", None)
            loss_mesh_depth = torch.tensor(0.0).to(loss)
            if self.config.lambda_mesh_depth > 0.0 and depth_mesh is not None:
                depth_mesh = depth_mesh.squeeze()
                if depth_gaussian is None:
                    depth_gaussian = (
                        self.config.median_fusing_ratio * depth_median + (1.0 - self.config.median_fusing_ratio) * depth_expected
                    ).squeeze()

                error_map = torch.log(1.0 + (depth_mesh - depth_gaussian).abs() / spatial_scale)
                if mesh_valid_mask is None:
                    mesh_valid_mask = (depth_mesh > 0.0).squeeze()
                loss_mesh_depth = (error_map * mesh_valid_mask).mean()

            metrics["loss_mesh_depth"] = loss_mesh_depth
            pbar["loss_mesh_depth"] = False
            loss += self.config.lambda_mesh_depth * loss_mesh_depth

            # Mesh normal regularization
            normal_mesh = outputs.get("mesh_normal", None)  # (3, H, W) in world coordinate
            depth_mesh = outputs.get("mesh_depth", None)
            loss_mesh_normal = torch.tensor(0.0).to(loss)
            if self.config.lambda_mesh_normal > 0.0 and all(m is not None for m in [normal_mesh, depth_mesh]):
                normal_mesh = torch.einsum(
                    "i j, i h w -> j h w", camera.world_to_camera[:3, :3], normal_mesh
                )  # (3, H, W) in camera coordinate
                # Compute normal_gaussian by fusing normal_from_median_depth and normal_from_expected_depth
                if normal_from_median_depth is None:
                    normal_from_median_depth = self.depth_to_normal(depth_median, camera)
                if normal_from_expected_depth is None:
                    normal_from_expected_depth = self.depth_to_normal(depth_expected, camera)
                if normal_gaussian is None:
                    normal_gaussian = (
                        self.config.median_fusing_ratio * normal_from_median_depth
                        + (1.0 - self.config.median_fusing_ratio) * normal_from_expected_depth
                    )
                # # Compute normal_gaussian using GS renderer
                # normal_gaussian = normal

                error_map = 1.0 - (normal_gaussian * normal_mesh).sum(0).abs()
                if mesh_valid_mask is None:
                    mesh_valid_mask = (depth_mesh > 0.0).squeeze()
                loss_mesh_normal = (error_map * mesh_valid_mask).mean()

            metrics["loss_mesh_normal"] = loss_mesh_normal
            pbar["loss_mesh_normal"] = False
            loss += self.config.lambda_mesh_normal * loss_mesh_normal

            # Occupancy label loss
            loss_occupancy_label = torch.tensor(0.0).to(loss)
            if self.config.use_occupancy_label_loss and self.config.lambda_occupancy_label > 0.0:
                occupancy_logits = getattr(gaussian_model, "get_delaunay_occupancy_logit", None)
                occupancy_labels = getattr(gaussian_model, "get_delaunay_occupancy_label", None)
                assert occupancy_logits is not None and occupancy_labels is not None

                occupancy_logits, occupancy_labels = occupancy_logits.reshape(-1), occupancy_labels.reshape(-1)
                # loss_occupancy_label = F.binary_cross_entropy_with_logits(occupancy_logits, occupancy_labels, reduction="none")
                # loss_occupancy_label = (loss_occupancy_label * (occupancy_labels > 0.5)).mean()
                loss_occupancy_label = F.binary_cross_entropy_with_logits(occupancy_logits, occupancy_labels)
                loss_occupancy_label = (loss_occupancy_label * (occupancy_labels > 0.5).float()).mean()

            metrics["loss_occupancy_label"] = loss_occupancy_label
            pbar["loss_occupancy_label"] = False
            loss += self.config.lambda_occupancy_label * loss_occupancy_label

            # Center isosurface loss
            loss_center_isosurface = torch.tensor(0.0).to(loss)
            if self.config.lambda_center_isosurface > 0.0:
                isosurface = getattr(gaussian_model.config, "sdf_isosurface", 0.5)
                occupancy = getattr(gaussian_model, "get_delaunay_occupancy", None)
                assert occupancy is not None

                center_occupancy = occupancy[:, -1]
                loss_center_isosurface = (isosurface - center_occupancy).clamp(min=0.0).mean()  # gaussian center should be inside surface

            metrics["loss_center_isosurface"] = loss_center_isosurface
            pbar["loss_center_isosurface"] = False
            loss += self.config.lambda_center_isosurface * loss_center_isosurface

        return metrics, pbar, loss

    def training_setup(self, pl_module):
        pl_module.extra_train_metrics.append(self._log_rendered_results)
        return super().training_setup(pl_module)

    @classmethod
    @torch.no_grad()
    def _log_rendered_results(cls, outputs, batch, gaussian_model, global_step, pl_module: lightning.LightningModule, metrics, prog_bar):
        if global_step % 1000 == 0:
            camera, image_info, _ = batch
            image_name, gt_image, masked_pixels = image_info
            median_depth = outputs.get("median_depth", None)
            if median_depth is None:
                return

            render = outputs["render"]
            median_depth = outputs["median_depth"]
            normal = (1.0 - outputs["normal"]) / 2.0
            mesh_depth = outputs.get("mesh_depth", None)
            if mesh_depth is not None:
                mesh_depth = torch.where(mesh_depth > median_depth.min(), mesh_depth, median_depth.min())
                median_depth = cls.depth2invdepth(median_depth)
                mesh_depth = cls.depth2invdepth(mesh_depth)
                mesh_normal = torch.einsum(
                    "i j, i h w -> j h w", camera.world_to_camera[:3, :3], outputs["mesh_normal"]
                )  # (3, H, W) in camera coordinate
                mesh_normal = (1.0 - cls.fix_normal_map(camera, mesh_normal)) / 2.0
                img = torch.cat(
                    [
                        torch.cat([render, median_depth, normal], dim=-1),
                        torch.cat([gt_image, mesh_depth, mesh_normal], dim=-1),
                    ],
                    dim=-2,
                )
            else:
                median_depth = cls.depth2invdepth(median_depth)
                img = torch.cat([render, median_depth, normal], dim=-1)

            pl_module.log_image("Rendered", img)

    @classmethod
    def depth_to_normal(cls, depth: torch.Tensor, camera: Camera):
        """
        Convert depth map to normal map in camera coordinate.

        Args:
            depth (Tensor (1, H, W)): depth map
            camera: Camera

        Returns:
            normal (Tensor, (3, H, W)): normal map
        """
        pointmap = cls.depth_to_pointmap(depth=depth, camera=camera)
        normal = cls.pointmap_to_normal(pointmap=pointmap)
        return normal

    @classmethod
    def depth_to_pointmap(cls, depth: torch.Tensor, camera: Camera):
        """
        Convert depth map to point map in camera coordinate.

        Args:
            depth (Tensor (1, H, W)): depth map
            camera: Camera

        Returns:
            pointmap (Tensor, (3, H, W)): point map
        """
        H, W = depth.shape[-2:]
        assert camera.height.item() == H and camera.width.item() == W

        intrins_inv = torch.tensor(
            [
                [1 / camera.fx, 0.0, -camera.width / (2 * camera.fx)],
                [0.0, 1 / camera.fy, -camera.height / (2 * camera.fy)],
                [0.0, 0.0, 1.0],
            ]
        ).to(depth)
        grid_x, grid_y = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).to(intrins_inv).reshape(3, -1)
        rays_d = intrins_inv @ points
        pointmap = depth.reshape(1, -1) * rays_d
        return pointmap.reshape(3, H, W)

    @classmethod
    def pointmap_to_normal(cls, pointmap: torch.Tensor):
        """
        Convert point map to normal map in camera coordinate.

        Args:
            pointmap (Tensor (3, H, W)): point map
            camera: Camera

        Returns:
            normal (Tensor, (3, H, W)): normal
        """
        normal = pointmap.new_zeros(pointmap.shape)
        dx = pointmap[..., 2:, 1:-1] - pointmap[..., :-2, 1:-1]
        dy = pointmap[..., 1:-1, 2:] - pointmap[..., 1:-1, :-2]
        normal[..., 1:-1, 1:-1] = F.normalize(torch.cross(dx, dy, dim=0), dim=0)
        return normal

    @staticmethod
    def depth2invdepth(depth: torch.Tensor):
        invdepth = 1.0 / depth.clamp_min(1e-6)
        invdepth = (invdepth - invdepth.min()) / (invdepth.max() - invdepth.min())
        invdepth = Visualizers.float_colormap(invdepth, "inferno")
        return invdepth

    @staticmethod
    def fix_normal_map(view: Camera, normal: torch.Tensor, normal_in_view_space=True):
        W, H = view.width.item(), view.height.item()
        intrins_inv = torch.linalg.inv(view.get_K()[:3, :3])
        grid_x, grid_y = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).to(normal)
        rays_d = (intrins_inv @ points).reshape(3, H, W)

        if normal_in_view_space:
            normal_view = normal
        else:
            normal_view = normal.clone()
            if normal.shape[0] == 3:
                normal_view = normal_view.permute(1, 2, 0)
            normal_view = normal_view @ view.world_to_camera[:3, :3]
            if normal.shape[0] == 3:
                normal_view = normal_view.permute(2, 0, 1)

        if normal_view.shape[0] != 3:
            rays_d = rays_d.permute(1, 2, 0)
            dim_to_sum = -1
        else:
            dim_to_sum = 0

        return torch.sign((-rays_d * normal_view).sum(dim=dim_to_sum, keepdim=True)) * normal_view
