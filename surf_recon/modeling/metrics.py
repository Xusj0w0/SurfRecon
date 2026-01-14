from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from internal.cameras import Camera, Cameras
from internal.metrics.vanilla_metrics import VanillaMetrics, VanillaMetricsImpl

from .utils.sdf import SDFUtils


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

    reset_tetrahedralization_interval: int = 200
    "Reset tetrahedralization every N iters"


@dataclass
class MeshRegularizedMetrics(VanillaMetrics):
    dn_start_iter: int = 7_000
    "Depth-Normal consistency regularization start iteration"

    lambda_dn: float = 0.05

    median_fusing_ratio: float = 0.6
    """
    If both median_depth and expected_depth are rendered,
    normal errors from median_depth are scaled by alpha,
    while that of expected_depth are scaled by (1-alpha)
    """

    mesh_regularization_schedule: MeshRegularizationSchedule = field(default_factory=lambda: MeshRegularizationSchedule())

    use_occupancy_label_loss: bool = True
    "Whether to compute bce loss between occupancy and its label (evaluated by mesh)"

    lambda_mesh_depth: float = 0.01
    "Weighting coefficient of loss between gaussian depth and mesh depth"

    lambda_mesh_normal: float = 0.01
    "Weighting coefficient of loss between gaussian normal and mesh normal"

    lambda_occupancy_label: float = 0.001

    lambda_center_isosurface: float = 0.001

    def instantiate(self, *args, **kwargs):
        return MeshRegularizedMetricsImpl(self)


class MeshRegularizedMetricsImpl(VanillaMetricsImpl):
    config: MeshRegularizedMetrics

    def _get_basic_metrics(self, pl_module, gaussian_model, batch, outputs):
        metrics, pbar = super()._get_basic_metrics(pl_module, gaussian_model, batch, outputs)

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
        loss_dn = torch.tensor(0.0).to(metrics["loss"])
        if global_step > self.config.dn_start_iter and self.config.lambda_dn > 0.0:
            depth_median = outputs.get("median_depth", None)
            depth_expected = outputs.get("expected_depth", None)
            normal = outputs.get("normal", None)  # (3, H, W) in camera coordinate
            assert all(m is not None for m in [depth_median, depth_expected, normal])

            normal_from_median_depth = normal_from_median_depth or self.depth_to_normal(depth_median, camera)
            normal_from_expected_depth = normal_from_expected_depth or self.depth_to_normal(depth_expected, camera)
            error_map_median = 1.0 - (normal * normal_from_median_depth).sum(0)
            error_map_expected = 1.0 - (normal * normal_from_expected_depth).sum(0)
            loss_dn = (
                self.config.median_fusing_ratio * error_map_median.mean()
                + (1.0 - self.config.median_fusing_ratio) * error_map_expected.mean()
            )

        metrics["loss_dn"] = loss_dn
        pbar["loss_dn"] = False
        metrics["loss"] += self.config.lambda_dn * loss_dn

        # Mesh regularization
        if global_step > self.config.mesh_regularization_schedule.start_iter:
            depth_median = outputs.get("median_depth", None)
            depth_expected = outputs.get("expected_depth", None)
            normal = outputs.get("normal", None)  # (3, H, W) in camera coordinate
            assert all(m is not None for m in [depth_median, depth_expected, normal])

            # Mesh depth regularization
            loss_mesh_depth = torch.tensor(0.0).to(metrics["loss"])
            if self.config.lambda_mesh_depth > 0.0:
                depth_mesh = outputs.get("mesh_depth", None)
                assert depth_mesh is not None

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
            metrics["loss"] += self.config.lambda_mesh_depth * loss_mesh_depth

            # Mesh normal regularization
            loss_mesh_normal = torch.tensor(0.0).to(metrics["loss"])
            if self.config.lambda_mesh_normal > 0.0:
                normal_mesh = outputs.get("mesh_normal", None)  # (H, W, 3) in world coordinate
                assert normal_mesh is not None

                normal_mesh = torch.einsum(
                    "i j, h w i -> j h w", camera.world_to_camera[:3, :3], normal_mesh
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
            metrics["loss"] += self.config.lambda_mesh_normal * loss_mesh_normal

            # Occupancy label loss
            loss_occupancy_label = torch.tensor(0.0).to(metrics["loss"])
            if self.config.lambda_occupancy_label > 0.0:
                occupancy_logits = getattr(gaussian_model, "get_delaunay_occupancy_logit", None)
                occupancy_labels = getattr(gaussian_model, "get_delaunay_occupancy_label", None)
                assert occupancy_logits is not None and occupancy_labels is not None

                occupancy_logits, occupancy_labels = occupancy_logits.reshape(-1), occupancy_labels.reshape(-1)
                loss_occupancy_label = F.binary_cross_entropy_with_logits(occupancy_logits, occupancy_labels, reduction="none")
                loss_occupancy_label = (loss_occupancy_label * (occupancy_labels > 0.5)).mean()

            metrics["loss_occupancy_label"] = loss_occupancy_label
            pbar["loss_occupancy_label"] = False
            metrics["loss"] += self.config.lambda_occupancy_label * loss_occupancy_label

            # Center isosurface loss
            loss_center_isosurface = torch.tensor(0.0).to(metrics["loss"])
            if self.config.lambda_center_isosurface > 0.0:
                isosurface = getattr(gaussian_model.config, "sdf_isosurface", 0.5)
                occupancy = getattr(gaussian_model, "get_delaunay_occupancy", None)
                assert occupancy is not None

                center_occupancy = occupancy.reshape(-1, 9)[:, -1]
                loss_center_isosurface = (isosurface - center_occupancy).clamp(min=0.0).mean()  # gaussian center should be inside surface

            metrics["loss_center_isosurface"] = loss_center_isosurface
            pbar["loss_center_isosurface"] = False
            metrics["loss"] += self.config.lambda_center_isosurface * loss_center_isosurface

        return metrics, pbar

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

        intrins_inv = torch.linalg.inv(camera.get_K()[..., :3, :3])
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
