import argparse
import copy
import gc
import json
import os
import os.path as osp
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from internal.cameras import Camera, Cameras
from internal.dataparsers.colmap_dataparser import Colmap
from internal.models.mip_splatting import MipSplatting, MipSplattingModelMixin
from internal.models.vanilla_gaussian import (VanillaGaussian,
                                              VanillaGaussianModel)
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.general_utils import build_rotation
from surf_recon.modeling.mesh_gaussian import (MeshGaussianUtils, MeshMixin,
                                               MeshVanillaGaussian,
                                               MeshVanillaGaussianModel)
from surf_recon.utils.partitionable_scene import (MinMaxBoundingBox,
                                                  PartitionCoordinates)
from surf_recon.utils.path_utils import (get_cells_output_dir,
                                         get_partition_info_dir,
                                         get_project_ckpt_dir)

MERGABLE_PROPERTY_NAMES = [
    "means",
    "shs_dc",
    "shs_rest",
    "opacities",
    "scales",
    "rotations",
    MipSplattingModelMixin._filter_3d_name,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", type=str, required=True)
    return parser.parse_args()


def build_transform_matrix(transforms: torch.Tensor) -> torch.Tensor:
    matrix = torch.eye(4, device=transforms.device)
    matrix[:3, :3] = build_rotation(transforms[:4].unsqueeze(0)).squeeze(0)
    matrix[:3, 3] = transforms[4:]
    return matrix


def unbound_bbox(bbox: MinMaxBoundingBox, cell_id: Tuple[int], dims: Tuple[int]):
    x_id, y_id = cell_id
    x_dim, y_dim = dims
    bbox_min, bbox_max = bbox.min.clone(), bbox.max.clone()
    if x_id == 0:
        bbox_min[0] = -torch.inf
    if x_id == x_dim - 1:
        bbox_max[0] = torch.inf
    if y_id == 0:
        bbox_min[1] = -torch.inf
    if y_id == y_dim - 1:
        bbox_max[1] = torch.inf
    return MinMaxBoundingBox(bbox_min, bbox_max)


def filter_delaunay_gaussians(gaussian_model: MeshMixin, is_in_partition: torch.Tensor):
    n_gaussians, device = gaussian_model.n_gaussians, gaussian_model.get_xyz.device

    # delaunay gaussian ids
    delaunay_ids = gaussian_model.get_delaunay_gaussian_ids
    delaunay_mask = torch.zeros((n_gaussians,), dtype=torch.bool)
    delaunay_mask[delaunay_ids] = True
    new_delaunay_ids = torch.nonzero(delaunay_mask[is_in_partition])[:, 0]
    gaussian_model.set_delaunay_gaussian_ids(new_delaunay_ids)

    # delaunay tetrahedra ids
    gaussian_model.set_delaunay_tets(torch.zeros((0, 4), dtype=torch.long, device=device))

    # occupancies
    mask = is_in_partition[delaunay_ids]
    base_occupancies = gaussian_model.occupancy_activation(gaussian_model.get_base_occupancy)
    occupancies = gaussian_model.get_delaunay_occupancy
    gaussian_model.set_delaunay_occupancy(base_occupancies[mask], occupancies[mask])

    # occupancy labels
    occupancy_labels = gaussian_model.get_delaunay_occupancy_label
    if occupancy_labels is not None:
        gaussian_model.set_delaunay_occupancy_label(occupancy_labels[mask])


def update_ckpt(ckpt, gaussian_model: VanillaGaussianModel):
    n_gaussians = gaussian_model.n_gaussians

    # Update gaussian model's hyper parameters
    if isinstance(gaussian_model, MipSplattingModelMixin):
        new_opacities, new_scales = gaussian_model.get_3d_filtered_scales_and_opacities()
        gaussian_model.opacities = gaussian_model.opacity_inverse_activation(new_opacities)
        gaussian_model.scales = gaussian_model.scale_inverse_activation(new_scales)
        property_names = [i for i in list(gaussian_model.property_names) if i != MipSplattingModelMixin._filter_3d_name]
        gaussian_model._names = tuple(property_names)
        gaussian_model.gaussians.pop(MipSplattingModelMixin._filter_3d_name)
    if isinstance(gaussian_model, MeshMixin):
        ckpt["hyper_parameters"]["gaussian"] = MeshVanillaGaussian(sh_degree=gaussian_model.max_sh_degree)
    else:
        ckpt["hyper_parameters"]["gaussian"] = VanillaGaussian(sh_degree=gaussian_model.max_sh_degree)
    # Update gaussian model's state dict
    for i in list(ckpt["state_dict"].keys()):
        if i.startswith("gaussian_model.gaussians.") or i.startswith("frozen_gaussians."):
            del ckpt["state_dict"][i]
        if isinstance(gaussian_model, MeshMixin):
            if i.startswith("gaussian_model.tetrahedra."):
                del ckpt["state_dict"][i]
    for k, v in gaussian_model.gaussians.items():
        ckpt["state_dict"]["gaussian_model.gaussians.{}".format(k)] = v.data
    if isinstance(gaussian_model, MeshMixin):
        for k, v in gaussian_model.tetrahedra.items():
            ckpt["state_dict"]["gaussian_model.tetrahedra.{}".format(k)] = v.data

    # Update density controller's state dict
    for k in list(ckpt["state_dict"].keys()):
        if k.startswith("density_controller."):
            _states: torch.Tensor = ckpt["state_dict"][k]
            ckpt["state_dict"][k] = torch.zeros((n_gaussians, *_states.shape[1:]), dtype=_states.dtype)

    # Remove optimizer states
    ckpt["optimizer_states"] = []

    # Remove image_list in dataparser
    dataparser = ckpt["datamodule_hyper_parameters"]["parser"]
    new_dataparser = Colmap(**{k: getattr(dataparser, k, None) for k in Colmap.__dataclass_fields__})
    new_dataparser.image_list = None
    ckpt["datamodule_hyper_parameters"]["parser"] = new_dataparser


def main():
    args = parse_args()
    device = torch.device("cpu")
    torch.autograd.set_grad_enabled(False)

    # Load partition info
    part_info_dir = Path(get_partition_info_dir(args.project))
    metadata = json.load(open(str((part_info_dir / "metadata.json").absolute()), "r"))
    transform_matrix = build_transform_matrix(torch.tensor(metadata["scene"]["transforms"], dtype=torch.float32, device=device))
    scene_bbox = metadata["scene"]["bbox"]
    scene_bbox = MinMaxBoundingBox(
        min=torch.tensor(scene_bbox[:2]).to(transform_matrix),
        max=torch.tensor(scene_bbox[2:]).to(transform_matrix),
    )
    # Build partition coordinates
    cell_xys, cell_sizes, cell_ids = [], [], []
    for cell in metadata["cells"]:
        bbox = cell["bbox"]
        cell_xys.append(torch.tensor(bbox[:2]).to(transform_matrix))
        cell_sizes.append((torch.tensor(bbox[2:]) - torch.tensor(bbox[:2])).to(transform_matrix))
        cell_ids.append(torch.tensor(cell["partition_id"], dtype=torch.long, device=device))
    partition_coords = PartitionCoordinates(
        id=torch.stack(cell_ids, dim=0),
        xy=torch.stack(cell_xys, dim=0),
        size=torch.stack(cell_sizes, dim=0),
    )
    x_dim, y_dim = (
        partition_coords.id[:, 0].max().item() + 1,
        partition_coords.id[:, 1].max().item() + 1,
    )

    # Process each cell
    gaussians_to_merge = {}
    tetras_to_merge = {
        "gaussian_ids": [],
        "delaunay_tets": [],
        "base_occupancy": [],
        "occupancy_shift": [],
        "occupancy_label": [],
    }
    num_gaussians_merged = 0
    with tqdm(enumerate(partition_coords), desc="Merging cells") as pbar:
        for idx, (cell_id, cell_xy, cell_size) in pbar:
            name = metadata["cells"][idx]["name"]
            pbar.set_description("{}".format(name))
            bbox = MinMaxBoundingBox(min=cell_xy, max=cell_xy + cell_size)
            bbox_unbounded = unbound_bbox(bbox, cell_id.tolist(), (x_dim, y_dim))

            # Load checkpoint
            ckpt_path = GaussianModelLoader.search_load_file(osp.join(get_cells_output_dir(args.project), name))
            pbar.write("Loading checkpoint of {}".format(name))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # Initialize gaussian model
            gaussian_model: VanillaGaussianModel = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device=device)

            # Split gaussians
            pbar.write("Splitting Gaussians...")
            gpos = gaussian_model.get_xyz @ transform_matrix[:3, :3].T + transform_matrix[:3, -1]
            is_in_partition = torch.logical_and(
                (gpos[..., :2] >= bbox_unbounded.min).all(dim=-1),
                (gpos[..., :2] <= bbox_unbounded.max).all(dim=-1),
            )
            # Split tetrahedra properties
            if isinstance(gaussian_model, MeshMixin) and gaussian_model.n_delaunay_gaussians > 0:
                filter_delaunay_gaussians(gaussian_model, is_in_partition)
            # Split gaussian properties
            properties = {}
            for k, v in gaussian_model.properties.items():
                properties[k] = v[is_in_partition]
            gaussian_model.properties = properties

            # Concatenate properties
            pbar.write("Merging {} Gaussians...".format(gaussian_model.n_gaussians))
            for i in MERGABLE_PROPERTY_NAMES:
                if i in gaussian_model.property_names:
                    gaussians_to_merge.setdefault(i, []).append(gaussian_model.get_property(i))
            if isinstance(gaussian_model, MeshMixin):
                for i in tetras_to_merge.keys():
                    if i == "gaussian_ids":
                        tetras_to_merge[i].append(gaussian_model.tetrahedra[i].data + num_gaussians_merged)
                    else:
                        tetras_to_merge[i].append(gaussian_model.tetrahedra[i].data)
            num_gaussians_merged += gaussian_model.n_gaussians

    # Get merged gaussian model
    print("Merging cells...")
    merged_gaussians = {}
    for k, v in gaussians_to_merge.items():
        merged_gaussians[k] = torch.cat(v, dim=0)
        v.clear()
        gc.collect()
        torch.cuda.empty_cache()
    gaussian_model.properties = merged_gaussians
    merged_tetras = {}
    for k, v in tetras_to_merge.items():
        merged_tetras[k] = torch.cat(v, dim=0)
        v.clear()
        gc.collect()
        torch.cuda.empty_cache()
    if isinstance(gaussian_model, MeshMixin):
        for k, v in merged_tetras.items():
            gaussian_model.tetrahedra[k] = torch.nn.Parameter(v, requires_grad=gaussian_model.tetrahedra[k].requires_grad)
        # Run tetrahedralization
        # with torch.no_grad():
        #     voronoi_points, _ = gaussian_model.compute_tetra_vertices()
        # delaunay_tets = MeshGaussianUtils.compute_delaunay_tetrahedralization(voronoi_points.to(tetra_device))
        # gaussian_model.set_delaunay_tets(delaunay_tets.to(device))
        # gaussian_model.set_delaunay_tets(torch.load("tmp.pt"))

    # Write checkpoint
    print("Total Gaussians merged: {}".format(gaussian_model.n_gaussians))
    print("Writing merged checkpoint...")
    update_ckpt(ckpt, gaussian_model)
    merged_ckpt_path = osp.join(get_project_ckpt_dir(args.project), "merged.ckpt")
    os.makedirs(osp.dirname(merged_ckpt_path), exist_ok=True)
    torch.save(ckpt, merged_ckpt_path)


if __name__ == "__main__":
    main()
