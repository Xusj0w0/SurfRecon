import argparse
import os
from typing import List, Literal, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from internal.cameras import Camera, Cameras
from internal.dataparsers.colmap_dataparser import Colmap
from internal.models.mip_splatting import (MipSplattingModelMixin,
                                           MipSplattingUtils)
from internal.models.vanilla_gaussian import (VanillaGaussian,
                                              VanillaGaussianModel)
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gs2d_mesh_utils import GS2DMeshUtils, post_process_mesh
from surf_recon.modeling.mesh_gaussian import MeshGaussianUtils
from surf_recon.modeling.renderers import (MeshBindedRenderer,
                                           MeshBindedRendererModule)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", "-c", type=str, required=True, help="Path to ckpt file")
    parser.add_argument("--dataset_path", "-d", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to output ply mesh")
    parser.add_argument("--voxel_size", type=float, default=0.004)
    parser.add_argument("--sdf_trunc", type=float, default=0.02)
    parser.add_argument("--depth_trunc", type=float, default=3.0)

    args = parser.parse_args()
    assert args.output_path.endswith(".ply"), "Mesh should be saved in PLY format"
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


def convert_to_o3d_camera(camera: Camera) -> o3d.camera.PinholeCameraParameters:
    camera_o3d = o3d.camera.PinholeCameraParameters()
    camera_o3d.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(camera.width),
        height=int(camera.height),
        fx=float(camera.fx),
        fy=float(camera.fy),
        cx=float(camera.cx),
        cy=float(camera.cy),
    )
    camera_o3d.extrinsic = np.asarray(camera.world_to_camera.T.cpu().numpy())
    return camera_o3d


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

    # create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    cameras = dataparser_outputs.train_set.cameras
    for idx in tqdm(range(len(cameras)), total=len(cameras), desc="Integrating frames into TSDF volume"):
        camera = cameras[idx].to_device(device)
        outputs = renderer(camera, model, render_types=["rgb", "depth"])
        rgb, depth = outputs["render"].squeeze(), outputs["median_depth"].squeeze()
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255.0, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(depth[..., None].cpu().numpy(), order="C")),
            depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        camera_o3d = convert_to_o3d_camera(cameras[idx])
        volume.integrate(rgbd, intrinsic=camera_o3d.intrinsic, extrinsic=camera_o3d.extrinsic)

    # extract mesh
    mesh = volume.extract_triangle_mesh()

    # post-process mesh
    mesh_o3d = post_process_mesh(mesh, 50)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(args.output_path, mesh_o3d, write_triangle_uvs=False, write_vertex_normals=True)


if __name__ == "__main__":
    main()
