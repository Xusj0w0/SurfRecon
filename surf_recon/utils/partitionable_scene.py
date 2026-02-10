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
import yaml
from matplotlib.pyplot import cm
from tqdm.auto import tqdm

from internal.cameras import Cameras
from internal.configs.instantiate_config import InstantiatableConfig
from internal.dataparsers.colmap_dataparser import Colmap
from internal.dataparsers.dataparser import (DataParserOutputs, ImageSet,
                                             PointCloud)
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.general_utils import build_rotation
from internal.utils.sh_utils import SH2RGB, eval_sh
from surf_recon.modeling.renderers.importance import rasterize_importance
from surf_recon.utils.path_utils import get_cell_partition_info_dir


def torch2numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


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
        return self.id[item], self.xy[item], self.size[item]

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

    def extend_to_bbox(self, bbox: MinMaxBoundingBox):
        _xymin = self.xy.clone()
        _xymax = _xymin + self.size.clone()
        _id = self.id.clone()

        x_dim, y_dim = self.id[:, 0].max().item() + 1, self.id[:, 1].max().item() + 1
        for cell_idx in range(len(self)):
            if _id[cell_idx][0] == 0:
                _xymin[cell_idx][0] = bbox.min[0]
            if _id[cell_idx][0] == x_dim - 1:
                _xymax[cell_idx][0] = bbox.max[0]
            if _id[cell_idx][1] == 0:
                _xymin[cell_idx][1] = bbox.min[1]
            if _id[cell_idx][1] == y_dim - 1:
                _xymax[cell_idx][1] = bbox.max[1]

            return PartitionCoordinates(id=_id, xy=_xymin, size=_xymax - _xymin)


@dataclass
class SceneConfig(InstantiatableConfig):
    dataset_path: str = ""

    coarse_model_path: str = ""

    transforms: List[float] = field(default_factory=lambda: [1.0] + [0.0] * 6)
    "Scene transformation, in [qw, qx, qy, qz, tx, ty, tz] format"

    partition_dim: List[int] = field(default_factory=lambda: [1, 1])

    scene_bbox_enlarge_by_pts: float = 0.0

    scene_bbox_outlier_by_pts: float = 0.005

    scene_bbox_enlarge_by_campos: float = 0.2

    bbox_enlarge_by_campos: float = 0.2
    "Enlarge block bbox for camera position based assignment"

    bbox_enlarge_by_camvis: float = 0.2
    "Enlarge block bbox for camera visibility computation"

    camera_visibility_threshold: float = 0.25

    bbox_enlarge_by_gaussian_pos: float = 0.1
    "Enlarge block bbox for gaussian position based assignment"

    gaussian_score_prune_ratio: float = 0.03

    def __post_init__(self):
        assert osp.exists(self.dataset_path) and osp.exists(self.coarse_model_path), "Dataset path and checkpoint path must exist"
        assert len(self.partition_dim) == 2, "Partition dimension must be a list of two integers"
        assert len(self.transforms) == 7, "Transforms must be a list of seven floats [qw, qx, qy, qz, tx, ty, tz]"

    def instantiate(self, *args, **kwargs):
        return PartitionableScene(config=self, *args, **kwargs)


class PartitionableScene:
    def __init__(self, config: SceneConfig, **kwargs):
        self.config = config
        self.device = torch.device("cuda")

    def run(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        # Save config
        with open(osp.join(output_path, "config.yaml"), "w") as f:
            yaml.safe_dump(asdict(self.config), f, indent=4, sort_keys=False)
        # Try loading intermediates
        intermediates = self.load_intermediates(output_path)

        # Load scene
        gaussian_model, renderer, ckpt, dataparser_outputs = self.load_scene()
        image_set = dataparser_outputs.train_set
        pcd = dataparser_outputs.point_cloud

        # Apply transformation to camera positions
        campos = image_set.cameras.camera_center.to(self.device)
        campos_transformed = campos @ self.rotation.T + self.translation

        # Compute scene bounding box and division
        scene_bbox = self.get_bounding_box_by_campos(campos_transformed)
        campos_bbox = MinMaxBoundingBox(
            min=torch.min(campos_transformed[..., :2], dim=0).values,
            max=torch.max(campos_transformed[..., :2], dim=0).values,
        )
        partition_coords = self.balanced_camera_based_division(campos_transformed, campos_bbox)

        fig, ax = plt.subplots()
        self.set_plot_ax_limit(ax, scene_bbox)
        # Plot scene and division
        _, scene_bbox_obj = self.plot_scene(ax, pcd, scene_bbox)
        fig.savefig(osp.join(self.get_figures_dir(output_path), "scene.png"), dpi=300)
        scene_bbox_obj.remove()
        cell_bbox_objs = self.plot_scene_division(ax, partition_coords)
        fig.savefig(osp.join(self.get_figures_dir(output_path), "scene_division.png"), dpi=300)
        for obj in cell_bbox_objs:
            obj.remove()

        # Save intermediate results
        intermediates_path = osp.join(self.get_intermediates_path(output_path), "scene_division.pt")
        torch.save(
            {
                "scene_bbox": asdict(scene_bbox),
                "partition_coords": asdict(partition_coords),
            },
            intermediates_path,
        )

        # Camera position based assignment
        campos_assign = self.is_in_bboxes(
            partition_coords.get_bounding_boxes(enlarge=self.config.bbox_enlarge_by_campos),
            campos_transformed,
        )  # [N_partitions, N_cameras]
        # Camera visibility based assignment
        cam_vis = intermediates.get("camera_visibility", None)
        if cam_vis is None:
            cam_vis = self.compute_camera_visibility(
                partition_coords=partition_coords,
                cameras=image_set.cameras,
                gaussian_model=gaussian_model,
            )
            # Save intermediate results
            intermediates_path = osp.join(self.get_intermediates_path(output_path), "camera_visibility.pt")
            torch.save(cam_vis, intermediates_path)
        camvis_assign = cam_vis > self.config.camera_visibility_threshold  # [N_partitions, N_cameras]
        camera_assign = torch.logical_or(campos_assign, camvis_assign)
        print("Num cameras assigned to each cell: {}".format(camera_assign.sum(dim=1).tolist()))

        # Plot camera assignment
        for cell_idx in range(len(partition_coords)):
            cell_objs = self.plot_cell(ax, cell_idx, partition_coords, campos_transformed, camera_assign)
            fig.savefig(
                osp.join(
                    self.get_figures_dir(output_path),
                    "{}.png".format(self.get_cell_name(cell_idx)),
                ),
                dpi=300,
            )
            for obj in cell_objs:
                obj.remove()
        plt.close(fig)

        # Save partitions
        self.save_partitions(
            output_path=output_path,
            scene_bbox=scene_bbox,
            partition_coords=partition_coords,
            gaussian_model=gaussian_model,
            image_set=image_set,
            camera_assign=camera_assign,
        )
        # Save pt
        partition_info = {
            "scene_bbox": asdict(scene_bbox),
            "partition_coords": asdict(partition_coords),
            "campos_assign": campos_assign,
            "camvis_assign": camvis_assign,
            "camera_visibility": cam_vis,
        }
        torch.save(partition_info, osp.join(output_path, "partition_info.pt"))

    def load_scene(
        self,
    ) -> Tuple[VanillaGaussianModel, VanillaRenderer, Dict[str, Any], DataParserOutputs]:
        ckpt_path = GaussianModelLoader.search_load_file(self.config.coarse_model_path)
        gaussian_model, renderer, ckpt = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            ckpt_path, device=self.device, eval_mode=False, pre_activate=False
        )
        dataparser: Colmap = ckpt["datamodule_hyper_parameters"]["parser"]
        dataparser.split_mode = "reconstruction"
        dataparser.points_from = "sfm"
        dataparser_outputs = dataparser.instantiate(path=self.config.dataset_path, output_path=os.getcwd(), global_rank=0).get_outputs()
        return gaussian_model, renderer, ckpt, dataparser_outputs

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
        num_cams = len(campos)
        x_dim, y_dim = self.config.partition_dim
        # Example 3x4 partition:
        # 3 7 11      (0,3) (1,3) (2,3)
        # 2 6 10      (0,2) (1,2) (2,2)
        # 1 5 9       (0,1) (1,1) (2,1)
        # 0 4 8       (0,0) (1,0) (2,0)

        # Divide cameras along x-axis
        num_cameras_per_column = math.ceil(num_cams / x_dim)
        _, x_sort_indices = torch.sort(campos[:, 0], dim=0)
        x_splits = [0.0 for _ in range(x_dim - 1)]
        y_splits = [[0.0 for _ in range(y_dim - 1)] for _ in range(x_dim)]
        for i, x_st in enumerate(range(0, num_cams, num_cameras_per_column)):
            x_ed = min(x_st + num_cameras_per_column, num_cams)
            cam_ids_in_col = x_sort_indices[x_st:x_ed]
            campos_in_col = campos[cam_ids_in_col]

            # Determine x split
            if i != 0:
                x_splits[i - 1] = 0.5 * (campos_in_col[:, 0].min() + prev_col_max_x)
            prev_col_max_x = campos_in_col[:, 0].max()

            # Divide cameras along y-axis
            _, y_sort_indices = torch.sort(campos_in_col[:, 1], dim=0)
            num_cams_in_col = len(campos_in_col)
            num_cams_per_cell = math.ceil(num_cams_in_col / y_dim)
            for j, y_st in enumerate(range(0, num_cams_in_col, num_cams_per_cell)):
                y_ed = min(y_st + num_cams_per_cell, num_cams_in_col)
                cam_ids_in_cell = cam_ids_in_col[y_sort_indices[y_st:y_ed]]
                campos_in_cell = campos[cam_ids_in_cell]

                if j != 0:
                    y_splits[i][j - 1] = 0.5 * (campos_in_cell[:, 1].min() + prev_cell_max_y)
                prev_cell_max_y = campos_in_cell[:, 1].max()

        # Build partition coords
        id_tensor, xy_tensor, size_tensor = (
            torch.empty((0, 2), device=self.device, dtype=torch.long),
            torch.empty((0, 2), device=self.device, dtype=torch.float),
            torch.empty((0, 2), device=self.device, dtype=torch.float),
        )
        for i in range(x_dim):
            for j in range(y_dim):
                if i == 0:
                    x_min, x_max = scene_bbox.min[0], x_splits[0]
                elif i == x_dim - 1:
                    x_min, x_max = x_splits[i - 1], scene_bbox.max[0]
                else:
                    x_min, x_max = x_splits[i - 1], x_splits[i]
                if j == 0:
                    y_min, y_max = scene_bbox.min[1], y_splits[i][0]
                elif j == y_dim - 1:
                    y_min, y_max = y_splits[i][j - 1], scene_bbox.max[1]
                else:
                    y_min, y_max = y_splits[i][j - 1], y_splits[i][j]

                id_tensor = torch.cat([id_tensor, torch.tensor([[i, j]]).to(id_tensor)], dim=0)
                xy_tensor = torch.cat([xy_tensor, torch.tensor([[x_min, y_min]]).to(xy_tensor)], dim=0)
                size_tensor = torch.cat(
                    [
                        size_tensor,
                        torch.tensor([[x_max - x_min, y_max - y_min]]).to(size_tensor),
                    ],
                    dim=0,
                )

        return PartitionCoordinates(id=id_tensor, xy=xy_tensor, size=size_tensor)

    def balanced_camera_based_division_prev(self, campos: torch.Tensor, scene_bbox: MinMaxBoundingBox):
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
                lower_cell = cells[ij2idx(i, j)]
                upper_cell = cells[ij2idx(i, j + 1)]
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
                left_cell = cells[ij2idx(i, j)]
                right_cell = cells[ij2idx(i + 1, j)]
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
            id_tensor = torch.cat(
                [
                    id_tensor,
                    torch.tensor([[i, j]], device=self.device, dtype=torch.long),
                ],
                dim=0,
            )
            xy_tensor = torch.cat(
                [
                    xy_tensor,
                    torch.tensor(
                        [[bbox.min[0], bbox.min[1]]],
                        device=self.device,
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            )
            size_tensor = torch.cat([size_tensor, (bbox.max - bbox.min).unsqueeze(0)], dim=0)

        return PartitionCoordinates(id=id_tensor, xy=xy_tensor, size=size_tensor)

    @torch.no_grad()
    def compute_camera_visibility(
        self,
        partition_coords: PartitionCoordinates,
        cameras: Cameras,
        gaussian_model: VanillaGaussianModel,
    ):
        cam_vis = torch.zeros(
            (len(partition_coords), len(cameras)),
            device=self.device,
            dtype=torch.float32,
        )

        bboxes = partition_coords.get_bounding_boxes(enlarge=self.config.bbox_enlarge_by_camvis)
        gs_means = gaussian_model.get_xyz.detach().clone().to(self.device)
        gs_means_transformed = gs_means @ self.rotation.T + self.translation
        is_gaussian_in_partition = self.is_in_bboxes(bboxes, gs_means_transformed)  # [N_partitions, N_gaussians]

        for cam_idx, camera in enumerate(tqdm(cameras, desc="Computing camera visibility")):
            camera = camera.to_device(self.device)
            outputs = rasterize_importance(viewpoint_camera=camera, pc=gaussian_model)
            accum_weights = outputs["accum_weights"]  # [N_gaussians]
            total_weights = accum_weights.sum()
            for cell_id in range(len(is_gaussian_in_partition)):
                gaussian_mask = is_gaussian_in_partition[cell_id]
                if gaussian_mask.sum() == 0:
                    cam_vis[cell_id, cam_idx] = 0.0
                else:
                    cam_vis[cell_id, cam_idx] = accum_weights[gaussian_mask].sum() / total_weights

        return cam_vis

    def load_intermediates(self, output_path: str) -> Dict[str, Any]:
        intermediates_path = self.get_intermediates_path(output_path)
        intermediates = {}
        if osp.exists(osp.join(intermediates_path, "scene_division.pt")):
            data = torch.load(
                osp.join(intermediates_path, "scene_division.pt"),
                map_location=self.device,
            )
            intermediates["scene_bbox"] = MinMaxBoundingBox(**data["scene_bbox"])
            intermediates["partition_coords"] = PartitionCoordinates(**data["partition_coords"])
        if osp.exists(osp.join(intermediates_path, "camera_visibility.pt")):
            intermediates["camera_visibility"] = torch.load(
                osp.join(intermediates_path, "camera_visibility.pt"),
                map_location=self.device,
            )
        # if osp.exists(osp.join(intermediates_path, "gaussian_score.pt")):
        #     intermediates["gaussian_score"] = torch.load(osp.join(intermediates_path, "gaussian_score.pt"), map_location=self.device)
        return intermediates

    def save_partitions(
        self,
        output_path: str,
        scene_bbox: MinMaxBoundingBox,
        partition_coords: PartitionCoordinates,
        gaussian_model: VanillaGaussianModel,
        image_set: ImageSet,
        camera_assign: torch.Tensor,
    ):
        metadata = {}
        metadata["scene"] = {
            "transforms": self.config.transforms,
            "bbox": scene_bbox.min.tolist() + scene_bbox.max.tolist(),
        }
        metadata["cells"] = []

        for cell_idx, (part_id, part_xy, part_size) in enumerate(tqdm(partition_coords, desc="Saving partitions")):
            cell_dir = get_cell_partition_info_dir(output_path, self.get_cell_name(cell_idx))
            os.makedirs(cell_dir, exist_ok=True)
            valid_cam_ids = camera_assign[cell_idx].nonzero().squeeze().tolist()

            # Save metadata
            metadata["cells"].append(
                {
                    "id": cell_idx,
                    "name": self.get_cell_name(cell_idx),
                    "partition_id": part_id.tolist(),
                    "bbox": part_xy.tolist() + (part_xy + part_size).tolist(),
                    "n_cameras": len(valid_cam_ids),
                }
            )

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
                        "appearance_id": (camera.appearance_id.item() if camera.appearance_id is not None else None),
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
        GaussianPlyUtils.load_from_model(gaussian_model).to_ply_format().save_to_ply(osp.join(output_path, "gaussians.ply"))

        with open(osp.join(output_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4, separators=(", ", ": "))

    def plot_scene(self, ax: plt.Axes, point_cloud: PointCloud, scene_bbox: MinMaxBoundingBox):
        # Sparsify points
        STEP = 32

        # Apply transformation to point cloud
        pts_xyz = torch.from_numpy(point_cloud.xyz[::STEP]).to(self.translation)
        pts_xyz = pts_xyz @ self.rotation.T + self.translation
        pts_rgb = torch.from_numpy(point_cloud.rgb[::STEP]).to(self.translation) / 255.0

        # Plot scene
        _pts_xyz = torch2numpy(pts_xyz)
        _pts_rgb = torch2numpy(pts_rgb)
        pcd_obj = ax.scatter(_pts_xyz[:, 0], _pts_xyz[:, 1], s=0.1, c=_pts_rgb, marker=".")
        # Plot scene bbox
        scene_bbox_min, scene_bbox_max = torch2numpy(scene_bbox.min), torch2numpy(scene_bbox.max)
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
        return [pcd_obj, scene_bbox_obj]

    def plot_scene_division(self, ax: plt.Axes, partition_coords: PartitionCoordinates):
        # plot division
        cell_bbox_objs = []
        for cell_idx, (part_id, part_xy, part_size) in enumerate(partition_coords):
            cell_bbox_min, cell_bbox_max = torch2numpy(part_xy), torch2numpy(part_xy + part_size)
            cell_bbox_obj = ax.add_artist(
                mpatches.Rectangle(
                    (cell_bbox_min[0], cell_bbox_min[1]),
                    cell_bbox_max[0] - cell_bbox_min[0],
                    cell_bbox_max[1] - cell_bbox_min[1],
                    fill=False,
                    edgecolor=self.COLORLIST[cell_idx % self.N_COLORS],
                    linewidth=1.0,
                    linestyle="-",
                )
            )
            cell_bbox_objs.append(cell_bbox_obj)
        return cell_bbox_objs

    def plot_cell(
        self,
        ax: plt.Axes,
        cell_idx: int,
        partition_coords: PartitionCoordinates,
        campos: torch.Tensor,
        camera_assign: torch.Tensor,
    ):
        color = self.COLORLIST[cell_idx % self.N_COLORS]

        # plot cell bbox
        part_id, part_xy, part_size = partition_coords[cell_idx]
        cell_bbox_min, cell_bbox_max = torch2numpy(part_xy), torch2numpy(part_xy + part_size)
        cell_bbox_obj = ax.add_artist(
            mpatches.Rectangle(
                (cell_bbox_min[0], cell_bbox_min[1]),
                cell_bbox_max[0] - cell_bbox_min[0],
                cell_bbox_max[1] - cell_bbox_min[1],
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                linestyle="-",
            )
        )
        # Plot cameras
        _campos = torch2numpy(campos[camera_assign[cell_idx]])
        campos_obj = ax.scatter(_campos[:, 0], _campos[:, 1], s=0.8, c="red", marker="o")
        # Annotate
        annotation_obj = ax.annotate(
            "Cell #{}: ({}, {}), {} cameras".format(cell_idx, part_id[0].item(), part_id[1].item(), _campos.shape[0]),
            xy=(
                cell_bbox_min[0] + 0.125 * (cell_bbox_max[0] - cell_bbox_min[0]),
                cell_bbox_min[1] + 0.25 * (cell_bbox_max[1] - cell_bbox_min[1]),
            ),
            fontsize=8,
        )

        return [cell_bbox_obj, campos_obj, annotation_obj]

    def set_plot_ax_limit(self, ax, scene_bbox: MinMaxBoundingBox, enlarge: float = 0.08):
        x_enlarge = (scene_bbox.max[0] - scene_bbox.min[0]) * enlarge
        y_enlarge = (scene_bbox.max[1] - scene_bbox.min[1]) * enlarge

        ax.set_xlim(
            [
                (scene_bbox.min[0] - x_enlarge).item(),
                (scene_bbox.max[0] + x_enlarge).item(),
            ]
        )
        ax.set_ylim(
            [
                (scene_bbox.min[1] - y_enlarge).item(),
                (scene_bbox.max[1] + y_enlarge).item(),
            ]
        )

        ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def is_in_bboxes(bboxes: MinMaxBoundingBoxes, points: torch.Tensor) -> torch.Tensor:
        xy_min, xy_max = bboxes.min.unsqueeze(1), bboxes.max.unsqueeze(1)  # [N_partitions, 1, 2]
        points = points[..., :2].unsqueeze(0)  # [1, N, 2]
        is_in_partition = torch.logical_and(
            (points >= xy_min.to(points)).all(dim=-1),
            (points <= xy_max.to(points)).all(dim=-1),
        )  # [N_partitions, N]
        return is_in_partition

    @property
    def rotation(self) -> torch.Tensor:
        if not hasattr(self, "_rotation"):
            transforms = torch.tensor(self.config.transforms, device=self.device)
            rotation = transforms[:4]
            self._rotation = build_rotation(rotation.unsqueeze(0)).squeeze(0)
        return self._rotation

    @property
    def translation(self) -> torch.Tensor:
        if not hasattr(self, "_translation"):
            transforms = torch.tensor(self.config.transforms, device=self.device)
            self._translation = transforms[4:]
        return self._translation

    @staticmethod
    def get_intermediates_path(output_path: str) -> str:
        intermediates_path = osp.join(output_path, "intermediates")
        os.makedirs(intermediates_path, exist_ok=True)
        return intermediates_path

    @staticmethod
    def get_figures_dir(output_path: str) -> str:
        figures_dir = osp.join(output_path, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        return figures_dir

    @staticmethod
    def get_cell_name(cell_idx: int) -> str:
        return f"cell_{cell_idx:03d}"

    @property
    def COLORLIST(self):
        if not hasattr(self, "_colorlist"):
            self._colorlist = list(iter(cm.rainbow(np.linspace(0, 1, self.N_COLORS))))
        return self._colorlist

    @property
    def N_COLORS(self):
        return 7
