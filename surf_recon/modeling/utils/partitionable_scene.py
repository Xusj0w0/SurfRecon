import json
import math
import os
import os.path as osp
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import cm
from tqdm.auto import tqdm

from internal.cameras import Cameras
from internal.configs.instantiate_config import InstantiatableConfig
from internal.dataparsers import DataParserOutputs, ImageSet
from internal.dataparsers.colmap_dataparser import Colmap
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.sh_utils import SH2RGB
from surf_recon.modeling.renderers.importance import render_simp


@dataclass
class MinMaxBoundingBox:
    min: torch.Tensor  # [2]
    max: torch.Tensor  # [2]


@dataclass
class MinMaxBoundingBoxes:
    min: torch.Tensor  # [N, 2]
    max: torch.Tensor  # [N, 2]

    def __getitem__(self, item):
        return MinMaxBoundingBox(
            min=self.min[item],
            max=self.max[item],
        )

    def to(self, *args, **kwargs):
        self.min = self.min.to(*args, **kwargs)
        self.max = self.max.to(*args, **kwargs)
        return self


@dataclass
class PartitionCoordinates:
    id: torch.Tensor  # [N_partitions, 2]
    xy: torch.Tensor  # [N_partitions, 2]
    size: torch.Tensor  # [N_partitions, 2]

    def __len__(self):
        return self.id.shape[0]

    def __getitem__(self, item):
        return self.id[item], self.xy[item]

    def __iter__(self):
        for idx in range(len(self)):
            yield self.id[idx], self.xy[idx], self.size[idx]

    def get_bounding_boxes(self, enlarge: Union[float, torch.Tensor] = 0.0) -> MinMaxBoundingBoxes:
        xy_min = self.xy - (enlarge * self.size)  # [N_partitions, 2]
        xy_max = self.xy + self.size + (enlarge * self.size)  # [N_partitions, 2]
        return MinMaxBoundingBoxes(
            min=xy_min,
            max=xy_max,
        )


@dataclass
class SceneConfig(InstantiatableConfig):
    dataset_path: str = ""

    ckpt_path: str = ""

    transforms: List[float] = field(default_factory=lambda: [1.0] + [0.0] * 6)
    "Scene transformation, in [qw, qx, qy, qz, tx, ty, tz] format"

    partition_dim: List[int] = field(default_factory=lambda: [1, 1])

    scene_bbox_enlarge_by_pts: float = 0.0

    scene_bbox_outlier_by_pts: float = 0.005

    scene_bbox_enlarge_by_campos: float = 0.2

    bbox_enlarge_by_campos: float = 0.1
    "Enlarge block bbox for camera position based assignment"

    bbox_enlarge_by_camvis: float = 0.1
    "Enlarge block bbox for camera visibility computation"

    camera_visibility_threshold: float = 0.25

    bbox_enlarge_by_gaussian_pos: float = 0.1
    "Enlarge block bbox for gaussian position based assignment"

    gaussian_score_prune_ratio: float = 0.05

    def __post_init__(self):
        assert osp.exists(self.dataset_path) and osp.exists(self.ckpt_path), "Dataset path and checkpoint path must exist"
        assert len(self.partition_dim) == 2, "Partition dimension must be a list of two integers"
        assert len(self.transforms) == 7, "Transforms must be a list of seven floats [qw, qx, qy, qz, tx, ty, tz]"

    def instantiate(self, *args, **kwargs):
        return PartitionableScene(config=self, *args, **kwargs)


class PartitionableScene:
    def __init__(self, config: SceneConfig, device: Optional[torch.device] = None, **kwargs):
        self.config = config
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def run(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        # Save config
        with open(osp.join(output_path, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=4, separators=(", ", ": "))
        # Try loading intermediates
        intermediates = self.load_intermediates(output_path)

        # Load scene
        gaussian_model, renderer, ckpt, image_set = self.load_scene()

        # Apply transformation to camera positions
        campos = image_set.cameras.camera_center.to(self.device)
        campos_transformed = campos @ self.rotation.T + self.translation

        # Apply transformation to gaussian means
        gs_means = gaussian_model.get_xyz.detach().clone().to(self.device)
        gs_means_transformed = gs_means @ self.rotation.T + self.translation

        # Compute scene bounding box and division
        scene_bbox = self.get_bounding_box_by_campos(campos_transformed)
        partition_coords = self.balanced_camera_based_division(campos_transformed, scene_bbox)

        # Camera position based assignment
        campos_assign = self.is_in_bboxes(
            partition_coords.get_bounding_boxes(enlarge=self.config.bbox_enlarge_by_campos), campos_transformed
        )  # [N_partitions, N_cameras]
        # Camera visibility based assignment
        cam_vis = intermediates.get("camera_visibility", None)
        if cam_vis is None:
            cam_vis = self.compute_camera_visibility(
                partition_coords=partition_coords, cameras=image_set.cameras, gaussian_model=gaussian_model
            )
        visibility_assign = cam_vis > self.config.camera_visibility_threshold  # [N_partitions, N_cameras]
        camera_assign = torch.logical_or(campos_assign, visibility_assign)

        # Gaussian position based assignment
        gaussian_pos_assign = self.is_in_bboxes(
            partition_coords.get_bounding_boxes(enlarge=self.config.bbox_enlarge_by_gaussian_pos), gs_means_transformed
        )
        # Gaussian score based assignment
        gs_score = intermediates.get("gaussian_score", None)
        if gs_score is None:
            gs_score = self.compute_gaussian_score(
                partition_coords=partition_coords,
                camera_assign=camera_assign,
                cameras=image_set.cameras,
                gaussian_model=gaussian_model,
            )
        gs_score_assign = gaussian_pos_assign.new_zeros(gaussian_pos_assign.shape)
        for cell_id in range(len(partition_coords)):
            gs_score_assign[cell_id] = self.init_cdf_mask(gs_score[cell_id], 1.0 - self.config.gaussian_score_prune_ratio)
        gaussian_assign = torch.logical_or(gaussian_pos_assign, gs_score_assign)

        # Save intermediate results
        if not "camera_visibility" in intermediates:
            self.save_intermediates(output_path, scene_bbox, partition_coords, cam_vis, gs_score)

        # Save plots
        self.save_plots(
            output_path=output_path,
            scene_bbox=scene_bbox,
            partition_coords=partition_coords,
            cameras=image_set.cameras,
            gaussian_model=gaussian_model,
            camera_assign=camera_assign,
            gaussian_assign=gaussian_assign,
        )

        # save partitions
        self.save_partitions(
            output_path=output_path,
            partition_coords=partition_coords,
            gaussian_model=gaussian_model,
            image_set=image_set,
            camera_assign=camera_assign,
            gaussian_assign=gaussian_assign,
        )

    def load_scene(self) -> Tuple[VanillaGaussianModel, VanillaRenderer, Dict[str, Any], ImageSet]:
        gaussian_model, renderer, ckpt = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            self.config.ckpt_path, device=self.device, pre_activate=False
        )
        dataparser: Colmap = ckpt["datamodule_hyper_parameters"]["parser"]
        dataparser.points_from = "random"
        dataparser.split_mode = "reconstruction"
        dataparser_outputs = dataparser.instantiate(path=self.config.dataset_path, output_path=os.getcwd(), global_rank=0).get_outputs()
        return gaussian_model, renderer, ckpt, dataparser_outputs.train_set

    def get_bounding_box_by_points(self, points: torch.Tensor):
        xyz_min = torch.quantile(points, self.config.scene_bbox_outlier_by_pts, dim=0)
        xyz_max = torch.quantile(points, 1 - self.config.scene_bbox_outlier_by_pts, dim=0)
        if self.config.scene_bbox_enlarge_by_pts > 0.0:
            size = xyz_max - xyz_min
            enlarge_size = size * self.config.scene_bbox_enlarge_by_pts
            xyz_min = xyz_min - enlarge_size
            xyz_max = xyz_max + enlarge_size

        return MinMaxBoundingBox(min=xyz_min, max=xyz_max)

    def get_bounding_box_by_campos(self, campos: torch.Tensor):
        xy_min = torch.min(campos[..., :2], dim=0).values
        xy_max = torch.max(campos[..., :2], dim=0).values
        if self.config.scene_bbox_enlarge_by_campos > 0.0:
            size = xy_max - xy_min
            enlarge_size = size * self.config.scene_bbox_enlarge_by_campos
            xy_min = xy_min - enlarge_size
            xy_max = xy_max + enlarge_size

        return MinMaxBoundingBox(min=xy_min, max=xy_max)

    def balanced_camera_based_division(self, campos: torch.Tensor, scene_bbox: MinMaxBoundingBox):
        """ """
        num_cams = len(campos)
        x_dim, y_dim = self.config.partition_dim
        cells = [None for _ in range(x_dim * y_dim)]
        ij2idx = lambda i, j: i * y_dim + j
        idx2ij = lambda idx: (idx // y_dim, idx % y_dim)
        # Example 3x4 partition:
        # 3 7 11      (0,3) (1,3) (2,3)
        # 2 6 10      (0,2) (1,2) (2,2)
        # 1 5 9       (0,1) (1,1) (2,1)
        # 0 4 8       (0,0) (1,0) (2,0)

        # Divide cameras along x-axis
        num_cameras_per_column = math.ceil(num_cams / x_dim)
        _, x_sort_indices = torch.sort(campos[:, 0], dim=0)
        for i, x_st in enumerate(range(0, num_cams, num_cameras_per_column)):
            x_ed = min(x_st + num_cameras_per_column, num_cams)
            x_mid_cam_id = x_sort_indices[-1] if x_ed < num_cams else None
            cam_ids_in_col = x_sort_indices[x_st:x_ed]
            campos_in_col = campos[cam_ids_in_col]

            # Divide cameras along y-axis
            _, y_sort_indices = torch.sort(campos_in_col[:, 1], dim=0)
            num_cams_in_col = len(campos_in_col)
            num_cams_per_cell = math.ceil(num_cams_in_col / y_dim)
            for j, y_st in enumerate(range(0, num_cams_in_col, num_cams_per_cell)):
                y_ed = min(y_st + num_cams_per_cell, num_cams_in_col)
                cam_ids_in_cell = cam_ids_in_col[y_sort_indices[y_st:y_ed]]
                y_mid_cam_id = cam_ids_in_cell[-1] if y_ed < num_cams_in_col else None

                # Compute cell bbox by campos
                _campos = campos[cam_ids_in_cell]
                bbox_campos = MinMaxBoundingBox(
                    min=torch.min(_campos[:, :2], dim=0).values,
                    max=torch.max(_campos[:, :2], dim=0).values,
                )
                cells[ij2idx(i, j)] = {
                    "cam_ids": cam_ids_in_cell,
                    "bbox": bbox_campos,
                    "x_mid_cam_id": x_mid_cam_id,
                    "y_mid_cam_id": y_mid_cam_id,
                }

        # Extend cell bbox
        # Extend along y-axis
        for i in range(x_dim):
            for j in range(y_dim - 1):
                lower_cell = cells[idx2ij(i, j)]
                upper_cell = cells[idx2ij(i, j + 1)]
                y_mid = 0.5 * (lower_cell["bbox"].max[1] + upper_cell["bbox"].min[1])
                lower_cell["bbox"].max[1] = y_mid
                upper_cell["bbox"].min[1] = y_mid

                if j == 0:
                    lower_cell["bbox"].min[1] = scene_bbox.min[1]
                if j + 1 == y_dim - 1:
                    upper_cell["bbox"].max[1] = scene_bbox.max[1]
        # Extend along x-axis
        for j in range(y_dim):
            for i in range(x_dim - 1):
                left_cell = cells[idx2ij(i, j)]
                right_cell = cells[idx2ij(i + 1, j)]
                x_mid = 0.5 * (left_cell["bbox"].max[0] + right_cell["bbox"].min[0])
                left_cell["bbox"].max[0] = x_mid
                right_cell["bbox"].min[0] = x_mid

                if i == 0:
                    left_cell["bbox"].min[0] = scene_bbox.min[0]
                if i + 1 == x_dim - 1:
                    right_cell["bbox"].max[0] = scene_bbox.max[0]

        # Build PartitionCoordinates
        id_tensor, xy_tensor, size_tensor = (
            torch.empty((0, 2), device=self.device, dtype=torch.long),
            torch.empty((0, 2), device=self.device, dtype=torch.float),
            torch.empty((0, 2), device=self.device, dtype=torch.float),
        )
        for idx, cell in enumerate(cells):
            i, j = idx2ij(idx)
            bbox: MinMaxBoundingBox = cell["bbox"]
            id_tensor = torch.cat([id_tensor, torch.tensor([[i, j]], device=self.device, dtype=torch.long)], dim=0)
            xy_tensor = torch.cat([xy_tensor, torch.tensor([[bbox.min[0], bbox.min[1]]], device=self.device, dtype=torch.float)], dim=0)
            size_tensor = torch.cat([size_tensor, (bbox.max - bbox.min).unsqueeze(0)], dim=0)

        return PartitionCoordinates(id=id_tensor, xy=xy_tensor, size=size_tensor)

    def compute_camera_visibility(self, partition_coords: PartitionCoordinates, cameras: Cameras, gaussian_model: VanillaGaussianModel):
        cam_vis = torch.zeros((len(partition_coords), len(cameras)), device=self.device, dtype=torch.float32)

        bboxes = partition_coords.get_bounding_boxes(enlarge=self.config.bbox_enlarge_by_camvis)
        gs_means = gaussian_model.get_xyz.detach().clone().to(self.device)
        gs_means_transformed = gs_means @ self.rotation.T + self.translation
        is_gaussian_in_partition = self.is_in_bboxes(bboxes, gs_means_transformed)  # [N_partitions, N_gaussians]

        for cam_idx, camera in enumerate(tqdm(cameras, desc="Computing camera visibility")):
            camera = camera.to_device(self.device)
            with torch.no_grad():
                outputs = render_simp(
                    viewpoint_camera=camera,
                    pc=gaussian_model,
                    bg_color=torch.tensor([0.0, 0.0, 0.0], device=self.device),
                )
            accum_weights = outputs["accum_weights"]  # [N_gaussians]
            total_weights = accum_weights.sum()
            for cell_id in range(len(is_gaussian_in_partition)):
                gaussian_mask = is_gaussian_in_partition[cell_id]
                if gaussian_mask.sum() == 0:
                    cam_vis[cell_id, cam_idx] = 0.0
                else:
                    cam_vis[cell_id, cam_idx] = accum_weights[gaussian_mask].sum() / total_weights

        return cam_vis

    def compute_gaussian_score(
        self, partition_coords: PartitionCoordinates, camera_assign: torch.Tensor, cameras: Cameras, gaussian_model: VanillaGaussianModel
    ):
        n_gaussians = gaussian_model.get_xyz.shape[0]
        gaussian_score = torch.zeros((len(partition_coords), n_gaussians), device=self.device, dtype=torch.float32)

        for cell_id in range(len(partition_coords)):
            valid_cam_ids = camera_assign[cell_id].nonzero().squeeze().tolist()
            for cam_id in enumerate(tqdm(valid_cam_ids, desc="Computing gaussian score at partition #{}".format(cell_id))):
                camera = cameras[cam_id].to_device(self.device)
                with torch.no_grad():
                    outputs = render_simp(
                        viewpoint_camera=camera,
                        pc=gaussian_model,
                        bg_color=torch.tensor([0.0, 0.0, 0.0], device=self.device),
                    )
                accum_weights = outputs["accum_weights"]  # [N_gaussians]
                area_proj = outputs["area_proj"]  # [N_gaussians]
                score = accum_weights / area_proj.float().clamp_min(1e-6)
                gaussian_score[cell_id] += score
        gaussian_score = gaussian_score / camera_assign.sum(dim=1, keepdim=True)
        return gaussian_score

    def save_intermediates(
        self,
        output_path: str,
        scene_bbox: MinMaxBoundingBox,
        partition_coords: PartitionCoordinates,
        camera_visibility: torch.Tensor,
        gaussian_score: torch.Tensor,
    ):
        intermediates_path = self.get_intermediates_path(output_path)
        transforms = camera_visibility.new_zeros((3, 4), dtype=torch.float32)
        transforms[:3, :3].copy_(self.rotation)
        transforms[:3, 3].copy_(self.translation)
        torch.save(
            {
                "transforms": transforms,
                "scene_bbox": asdict(scene_bbox),
                "partition_coords": asdict(partition_coords),
                "camera_visibility": camera_visibility,
                "gaussian_score": gaussian_score,
            },
            intermediates_path,
        )

    def load_intermediates(self, output_path: str) -> Dict[str, Any]:
        try:
            intermediates_path = self.get_intermediates_path(output_path)
            data = torch.load(intermediates_path, map_location=self.device)
            return {
                "scene_bbox": MinMaxBoundingBox(**data["scene_bbox"]),
                "partition_coords": PartitionCoordinates(**data["partition_coords"]),
                "camera_visibility": data["camera_visibility"],
                "gaussian_score": data["gaussian_score"],
            }
        except:
            return {}

    def save_partitions(
        self,
        output_path: str,
        partition_coords: PartitionCoordinates,
        gaussian_model: VanillaGaussianModel,
        image_set: ImageSet,
        camera_assign: torch.Tensor,
        gaussian_assign: torch.Tensor,
    ):
        for cell_idx, (part_id, part_xy, part_size) in enumerate(tqdm(partition_coords, desc="Saving partitions")):
            cell_dir = osp.join(self.get_cells_dir(output_path), self.get_cell_name(cell_idx))
            os.makedirs(cell_dir, exist_ok=True)
            valid_cam_ids = camera_assign[cell_idx].nonzero().squeeze().tolist()

            # Save metadata
            metadata = {
                "cell_id": cell_idx,
                "partition_id": part_id.tolist(),
                "xy_min": part_xy.tolist(),
                "xy_max": (part_xy + part_size).tolist(),
                "transforms": self.config.transforms,
            }
            with open(osp.join(cell_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4, separators=(", ", ": "))

            # Save camera to json
            camera_list = []
            for cam_id in valid_cam_ids:
                camera = image_set.cameras[cam_id]
                c2w = torch.linalg.inv(camera.world_to_camera.T)
                camera_list.append(
                    {
                        "id": cam_id,
                        "img_name": image_set.image_names[cam_id],
                        "width": int(camera.width),
                        "height": int(camera.height),
                        "position": c2w[:3, -1].numpy().tolist(),
                        "rotation": c2w[:3, :3].numpy().tolist(),
                        "fx": float(camera.fx),
                        "fy": float(camera.fy),
                        "cx": camera.cx.item(),
                        "cy": camera.cy.item(),
                        "time": camera.time.item() if camera.time is not None else None,
                        "appearance_id": camera.appearance_id.item() if camera.appearance_id is not None else None,
                        "normalized_appearance_id": (
                            camera.normalized_appearance_id.item() if camera.normalized_appearance_id is not None else None
                        ),
                    }
                )
            with open(osp.join(cell_dir, "cameras.json"), "w") as f:
                json.dump(camera_list, f, indent=4, separators=(", ", ": "))

            # Write image list
            with open(osp.join(cell_dir, "image_list.txt"), "w") as f:
                for cam_id in valid_cam_ids:
                    f.write(f"{image_set.image_names[cam_id]}\n")

            # Save Gaussian model
            gs_mask = gaussian_assign[cell_idx]
            properties = gaussian_model.properties
            masked_properties = {k: v[gs_mask] for k, v in properties.items()}
            gaussian_model.properties = masked_properties
            GaussianPlyUtils.load_from_model(gaussian_model).to_ply_format().save_to_ply(osp.join(cell_dir, "gaussians.ply"))
            gaussian_model.properties = properties

    def save_plots(
        self,
        output_path: str,
        scene_bbox: MinMaxBoundingBox,
        partition_coords: PartitionCoordinates,
        cameras: Cameras,
        gaussian_model: VanillaGaussianModel,
        camera_assign: torch.Tensor,
        gaussian_assign: torch.Tensor,
    ):
        figures_dir = self.get_figures_dir(output_path)
        os.makedirs(figures_dir, exist_ok=True)

        # Apply transformation to camera positions
        campos = cameras.camera_center.to(self.device)
        campos = campos @ self.rotation.T + self.translation

        # Apply transformation to gaussian means
        gs_means = gaussian_model.get_xyz.to(self.device)
        gs_means = gs_means @ self.rotation.T + self.translation
        gs_rgb = SH2RGB(gaussian_model.get_shs().to(self.device))

        # Sparsify points
        STEP = 32
        COLORS = list(iter(cm.rainbow(np.linspace(0, 1, len(partition_coords)))))

        # Plot scene
        fig, ax = plt.subplots()
        self.set_plot_ax_limit(ax, scene_bbox)
        # plot gaussians
        ax.scatter(gs_means[::STEP, 0].cpu().numpy(), gs_means[::STEP, 1].cpu().numpy(), s=1.0, c=gs_rgb[::STEP].cpu().numpy(), marker=".")
        # plot scene bbox
        scene_bbox_min, scene_bbox_max = scene_bbox.min.cpu().numpy(), scene_bbox.max.cpu().numpy()
        scene_bbox_obj = ax.add_artist(
            mpatches.Rectangle(
                (scene_bbox_min[0], scene_bbox_min[1]),
                scene_bbox_max[0] - scene_bbox_min[0],
                scene_bbox_max[1] - scene_bbox_min[1],
                fill=False,
                edgecolor="green",
                linewidth=2.0,
                linestyle="--",
            )
        )
        fig.savefig(osp.join(figures_dir, "scene.png"), dpi=300)
        # plot division
        scene_bbox_obj.remove()
        for cell_idx, (part_id, part_xy, part_size) in partition_coords:
            cell_bbox_min, cell_bbox_max = part_xy.cpu().numpy(), (part_xy + part_size).cpu().numpy()
            ax.add_artist(
                mpatches.Rectangle(
                    (cell_bbox_min[0], cell_bbox_min[1]),
                    cell_bbox_max[0] - cell_bbox_min[0],
                    cell_bbox_max[1] - cell_bbox_min[1],
                    fill=False,
                    edgecolor=COLORS[cell_idx % len(COLORS)],
                    linewidth=2.0,
                    linestyle="-",
                )
            )
        fig.savefig(osp.join(figures_dir, "scene_division.png"), dpi=300)
        plt.close(fig)

        # Plot cells
        for cell_idx, (part_id, part_xy, part_size) in enumerate(tqdm(partition_coords, desc="Saving partition plots")):
            fig, ax = plt.subplots()
            self.set_plot_ax_limit(ax, scene_bbox)
            color = COLORS[cell_idx % len(COLORS)]
            _gs_means = gs_means[gaussian_assign[cell_idx]].cpu().numpy()
            _gs_rgb = gs_rgb[gaussian_assign[cell_idx]].cpu().numpy()
            # plot gaussians
            ax.scatter(_gs_means[::STEP, 0], _gs_means[::STEP, 1], s=1.0, c=_gs_rgb[::STEP], marker=".")
            # plot cell bbox
            cell_bbox_min, cell_bbox_max = part_xy.cpu().numpy(), (part_xy + part_size).cpu().numpy()
            ax.add_artist(
                mpatches.Rectangle(
                    (cell_bbox_min[0], cell_bbox_min[1]),
                    cell_bbox_max[0] - cell_bbox_min[0],
                    cell_bbox_max[1] - cell_bbox_min[1],
                    fill=False,
                    edgecolor=color,
                    linewidth=2.0,
                    linestyle="-",
                )
            )
            # Plot cameras
            _campos = campos[camera_assign[cell_idx]].cpu().numpy()
            ax.scatter(_campos[:, 0], _campos[:, 1], s=5.0, c="red", marker="o")
            # Annotate
            ax.annotate(
                "Cell #{}: ({}, {}), {} cameras".format(cell_idx, part_id[0].item(), part_id[1].item(), _campos.shape[0]),
                xy=(
                    cell_bbox_min[0] + 0.125 * (cell_bbox_max[0] - cell_bbox_min[0]),
                    cell_bbox_min[1] + 0.25 * (cell_bbox_max[1] - cell_bbox_min[1]),
                ),
                fontsize=5,
            )
            fig.savefig(osp.join(figures_dir, self.get_cell_name(cell_idx) + ".png"), dpi=300)
            plt.close(fig)

    def set_plot_ax_limit(self, ax, scene_bbox: MinMaxBoundingBox, enlarge: float = 0.2):
        x_enlarge = (scene_bbox.max[0] - scene_bbox.min[0]) * enlarge
        y_enlarge = (scene_bbox.max[1] - scene_bbox.min[1]) * enlarge

        ax.set_xlim([(scene_bbox.min[0] - x_enlarge).item(), (scene_bbox.max[0] + x_enlarge).item()])
        ax.set_ylim([(scene_bbox.min[1] - y_enlarge).item(), (scene_bbox.max[1] + y_enlarge).item()])

        ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def is_in_bboxes(bboxes: MinMaxBoundingBoxes, points: torch.Tensor) -> torch.Tensor:
        xy_min, xy_max = bboxes.min.unsqueeze(1), bboxes.max.unsqueeze(1)  # [N_partitions, 1, 2]
        points = points[..., :2].unsqueeze(0)  # [1, N, 2]
        is_in_partition = torch.logical_and(
            (points >= xy_min.to(points)).all(dim=-1), (points <= xy_max.to(points)).all(dim=-1)
        )  # [N_partitions, N]
        return is_in_partition

    @property
    def rotation(self) -> torch.Tensor:
        if not hasattr(self, "_rotation"):
            transforms = torch.tensor(self.config.transforms, device=self.device)
            rotation = transforms[:4]
            self._rotation = self.quat2mat(rotation)
        return self._rotation

    @property
    def translation(self) -> torch.Tensor:
        if not hasattr(self, "_translation"):
            transforms = torch.tensor(self.config.transforms, device=self.device)
            self._translation = transforms[4:]
        return self._translation

    @staticmethod
    def quat2mat(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation matrix.
        :param quat: [4] tensor
        :return: [3, 3] tensor
        """
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        rotmat = torch.tensor(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=quat.dtype,
            device=quat.device,
        )
        return rotmat

    @staticmethod
    def init_cdf_mask(importance: torch.Tensor, threshold: float):
        importance = importance.flatten()
        if threshold != 1.0:
            percent_sum = threshold
            vals, _ = torch.sort(importance + 1e-6)
            cumsum_val = torch.cumsum(vals, dim=0)
            split_index = ((cumsum_val / vals.sum()) > (1 - percent_sum)).nonzero().min()
            split_val_nonprune = vals[split_index]

            non_prune_mask = importance > split_val_nonprune
        else:
            non_prune_mask = torch.ones_like(importance).bool()

        return non_prune_mask

    def get_intermediates_path(self, output_path: str) -> str:
        intermediates_path = osp.join(output_path, "intermediates.pt")
        return intermediates_path

    def get_cells_dir(self, output_path: str, cell_idx: int) -> str:
        return osp.join(output_path, "cells")

    def get_figures_dir(self, output_path: str) -> str:
        return osp.join(output_path, "figures")

    def get_cell_name(self, cell_idx: int) -> str:
        return f"cell_{cell_idx:03d}"
