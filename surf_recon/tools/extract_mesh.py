import argparse
import os
from typing import List, Literal, Optional, Tuple

import open3d as o3d
import torch

from internal.cameras import Camera, Cameras
from internal.dataparsers.colmap_dataparser import Colmap
from internal.models.mip_splatting import (MipSplattingModelMixin,
                                           MipSplattingUtils)
from internal.models.vanilla_gaussian import (VanillaGaussian,
                                              VanillaGaussianModel)
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gs2d_mesh_utils import post_process_mesh
from surf_recon.modeling.mesh_gaussian import MeshGaussianUtils
from surf_recon.modeling.renderers import (MeshBindedRenderer,
                                           MeshBindedRendererModule)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", "-c", type=str, required=True, help="Path to ckpt file")
    parser.add_argument("--dataset_path", "-d", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to output ply mesh")
    parser.add_argument("--num_max_delaunay_gaussians", type=int, default=600_000)
    parser.add_argument("--opacity_threshold", type=float, default=0.0)
    parser.add_argument("--trunc_margin", type=float, default=-1.0)
    parser.add_argument("--sdf_n_binary_steps", type=int, default=8)
    parser.add_argument("--without_color", action="store_false")
    parser.add_argument("--tetra_on_cpu", action="store_true")

    args = parser.parse_args()
    assert args.output_path.endswith(".ply"), "Mesh should be saved in PLY format"
    if args.trunc_margin < 0.0:
        args.trunc_margin = None
    return args


def load_model_and_renderer(ckpt: dict, device):
    sh_degree = int(ckpt["state_dict"]["gaussian_model._active_sh_degree"])
    model_state_dict = GaussianModelLoader.filter_state_dict_by_prefix(ckpt["state_dict"], "gaussian_model.")
    keys_to_remove = []
    for key in model_state_dict.keys():
        if key.startswith("tetrahedra."):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del model_state_dict[key]
    filter_3d_key = "gaussians.{}".format(MipSplattingModelMixin._filter_3d_name)
    if filter_3d_key in model_state_dict:
        from internal.utils.general_utils import inverse_sigmoid

        opacities = torch.sigmoid(model_state_dict["gaussians.opacities"])
        scales = torch.exp(model_state_dict["gaussians.scales"])
        opacities, scales = MipSplattingUtils.apply_3d_filter(model_state_dict[filter_3d_key], opacities, scales)
        model_state_dict["gaussians.scales"] = torch.log(scales)
        model_state_dict["gaussians.opacities"] = inverse_sigmoid(opacities)
        del model_state_dict[filter_3d_key]
    model = VanillaGaussian(sh_degree=sh_degree).instantiate()
    model.setup_from_number(model_state_dict["gaussians.means"].shape[0])
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    renderer = MeshBindedRenderer().instantiate()
    renderer.setup(stage="validation")
    renderer = renderer.to(device)

    return model, renderer


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    # load model and renderer
    model, renderer = load_model_and_renderer(ckpt, device)

    # load dataset
    dataparser: Colmap = ckpt["datamodule_hyper_parameters"]["parser"]
    dataparser.points_from = "random"
    # dataparser.split_mode = "reconstruction"
    dataparser_outputs = dataparser.instantiate(path=args.dataset_path, output_path=os.getcwd(), global_rank=0).get_outputs()

    # extract mesh from gaussian model
    mesh = MeshGaussianUtils.post_extract_mesh(
        gaussian_model=model,
        renderer=renderer,
        cameras=dataparser_outputs.train_set.cameras,
        max_num_delaunay_gaussians=args.num_max_delaunay_gaussians,
        opacity_threshold=args.opacity_threshold,
        trunc_margin=args.trunc_margin,
        sdf_n_binary_steps=args.sdf_n_binary_steps,
        without_color=args.without_color,
        device=device,
        tetra_on_cpu=args.tetra_on_cpu,
    )
    # build o3d mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.verts.cpu().numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces.cpu().numpy())
    if not args.without_color:
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(mesh.verts_colors.cpu().numpy()[:, :3] / 255.0)
    # post-process mesh
    mesh_o3d = post_process_mesh(mesh_o3d, 50)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(args.output_path, mesh_o3d, write_triangle_uvs=False, write_vertex_normals=False)


if __name__ == "__main__":
    main()
