import argparse
import gc
import os
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import trimesh

from internal.cameras import Camera, Cameras
from internal.dataparsers.colmap_dataparser import Colmap
from internal.utils.gaussian_model_loader import GaussianModelLoader
from surf_recon.modeling.mesh_gaussian import MeshGaussianUtils
from surf_recon.utils.sdf import PointIntegration, TSDFFusion
from surf_recon.utils.tetmesh import marching_tetrahedra


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", "-c", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--dataset_path", "-d", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to output ply mesh")
    parser.add_argument("--image_downsample_factor", type=float, default=1.0)
    parser.add_argument("--num_max_delaunay_gaussians", type=int, default=-1)
    parser.add_argument("--sdf_mode", choices=["depth_fusion", "integration"], default="integration")
    parser.add_argument("--trunc_margin", type=float, default=-1.0)
    parser.add_argument("--isosurface_value", type=float, default=0.5)
    parser.add_argument("--sdf_n_binary_steps", type=int, default=8)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--tetra_on_cpu", action="store_true")
    return parser


@torch.no_grad()
def extract(
    gaussian_model,
    renderer,
    train_cameras: Cameras,
    max_num_delaunay_gaussians: int = -1,
    sdf_mode: Literal["depth_fusion", "integration"] = "integration",
    trunc_margin: Optional[float] = None,
    isosurface_value: float = 0.5,
    sdf_n_binary_steps: int = 8,
    tetra_on_cpu: bool = False,
    device: torch.device = torch.device("cpu"),
) -> trimesh.Trimesh:
    if (gaussian_model.get_base_occupancy != 0.0).any() and max_num_delaunay_gaussians < 0:
        delaunay_gaussian_ids = gaussian_model.get_delaunay_gaussian_ids
    else:
        # Sample Delaunay gaussians
        delaunay_gaussian_ids = MeshGaussianUtils.sample_delaunay_gaussians(
            n_samples=max_num_delaunay_gaussians,
            gaussian_model=gaussian_model,
            renderer=renderer,
            train_cameras=train_cameras,
        )
    # Tetrahedralization
    voronoi_points, voronoi_scales = MeshGaussianUtils.compute_tetra_vertices(
        gaussian_model=gaussian_model, delaunay_gaussian_ids=delaunay_gaussian_ids
    )
    delaunay_tets = MeshGaussianUtils.compute_delaunay_tetrahedralization(voronoi_points)

    def _render(viewpoint: Camera):
        outputs = renderer.forward(viewpoint, gaussian_model, render_types=["rgb", "depth"])
        return {"rgb": outputs["render"], "depth": outputs["median_depth"]}

    def _integrate(points: torch.Tensor, viewpoint: Camera):
        outputs = renderer.integrate(points3D=points, viewpoint_camera=viewpoint, pc=gaussian_model)
        return {
            "alpha": outputs["alpha_integrated"],
            "point_coords": outputs["point_coordinate"],
            "color": outputs["color_integrated"],
            "mask": outputs["render"][7],
        }

    if sdf_mode == "integration":

        def _voronoi_sdf(points: torch.Tensor):
            voronoi_sdf, *_ = (
                PointIntegration(points=points, isosurface_value=isosurface_value)
                .run(cameras=train_cameras, integrate_fn=_integrate)
                .get_outputs()
            )
            return voronoi_sdf.squeeze()

    elif sdf_mode == "depth_fusion":

        def _voronoi_sdf(points: torch.Tensor):
            voronoi_sdf, *_ = (
                TSDFFusion(points=points, trunc_margin=trunc_margin, use_binary_opacity=False)
                .run(cameras=train_cameras, render_fn=_render)
                .get_outputs()
            )
            return voronoi_sdf.squeeze()

    else:
        raise RuntimeError("Unsupported sdf_mode: {}".format(sdf_mode))

    voronoi_sdf = _voronoi_sdf(voronoi_points)
    torch.cuda.empty_cache()
    gc.collect()

    # Marching tetrahedra
    if tetra_on_cpu:
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(
            voronoi_points.cpu()[None], delaunay_tets.cpu().long(), voronoi_sdf.cpu()[None], voronoi_scales.cpu()[None]
        )
    else:
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(
            voronoi_points[None], delaunay_tets.long(), voronoi_sdf[None], voronoi_scales[None]
        )
    end_points, end_sdf = verts_list[0]
    end_points, end_sdf = end_points.to(device), end_sdf.to(device)
    end_scales = scale_list[0].to(device)
    faces = faces_list[0].to(device)

    # Refine result of marching tetrahedra with binary search along intersected edges
    left_points, right_points = end_points[:, 0, :], end_points[:, 1, :]
    left_sdf, right_sdf = end_sdf[:, 0, :], end_sdf[:, 1, :]
    left_scale, right_scale = end_scales[:, 0, 0], end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    for _ in range(sdf_n_binary_steps):
        mid_points = (left_points + right_points) / 2

        mid_sdf = _voronoi_sdf(mid_points)
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        points = (left_points + right_points) / 2

    # Extract vertex colors
    tsdf, vertex_colors, _ = (
        TSDFFusion(points=points, trunc_margin=trunc_margin).run(cameras=train_cameras, render_fn=_render).get_outputs()
    )
    invisible_pts_mask = tsdf.squeeze() < -1.1
    _trunc_margin = 1.0 * TSDFFusion.get_cameras_center_and_diag(train_cameras)["diag"]
    extra_vertex_colors, *_ = (
        TSDFFusion(points=points[invisible_pts_mask], trunc_margin=_trunc_margin)
        .run(cameras=train_cameras, render_fn=_render)
        .get_outputs()
    )
    vertex_colors[invisible_pts_mask] = extra_vertex_colors

    # Build mesh
    mesh = trimesh.Trimesh(
        vertices=points.cpu().numpy(),
        faces=faces.cpu().numpy(),
        vertex_colors=(vertex_colors.cpu().numpy() * 255).clip(min=0, max=255).astype(np.uint8),
        process=False,
    )

    # Filter mesh
    vertex_mask = distance <= scale
    face_mask = vertex_mask[faces].all(dim=1)
    mesh.update_vertices(vertex_mask.cpu().numpy())
    mesh.update_faces(face_mask.cpu().numpy())

    return mesh


def main():
    args = make_parser().parse_args()
    assert args.output_path.endswith(".ply"), "Mesh should be saved in PLY format"
    if args.trunc_margin < 0.0:
        args.trunc_margin = None

    device = torch.device(f"cuda:{args.gpu_index}" if args.gpu_index >= 0 else "cpu")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(args.ckpt_path, device=device)
    dataparser_outputs = (
        Colmap(split_mode="reconstruction", down_sample_factor=args.image_downsample_factor)
        .instantiate(path=args.dataset_path, output_path=os.getcwd(), global_rank=0)
        .get_outputs()
    )

    mesh = extract(
        gaussian_model=model,
        renderer=renderer,
        train_cameras=dataparser_outputs.train_set.cameras,
        max_num_delaunay_gaussians=args.num_max_delaunay_gaussians,
        sdf_mode=args.sdf_mode,
        trunc_margin=args.trunc_margin,
        isosurface_value=args.isosurface_value,
        sdf_n_binary_steps=args.sdf_n_binary_steps,
        tetra_on_cpu=args.tetra_on_cpu,
        device=device,
    )
    mesh.export(os.path.abspath(args.output_path))


if __name__ == "__main__":
    main()
