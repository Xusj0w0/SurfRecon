import argparse
import os
import os.path as osp
import subprocess

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from tqdm import tqdm

from internal.utils.colmap import read_cameras_binary, read_images_binary
from internal.utils.visualizers import Visualizers
from utils.common import AsyncImageSaver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--input_size", "-s", type=int, default=504)
    parser.add_argument("--name", "-n", default="da3")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--preview", "-p", action="store_true", default=False)
    parser.add_argument("--keep_resolution", action="store_true", default=False)
    return parser.parse_args()


def get_image_data(image_id_list, colmap_images, colmap_cameras, image_dir):
    image_names = []
    image_files = []
    extrinsics = []
    intrinsics = []

    for image_id in image_id_list:
        image_data = colmap_images[image_id]
        image_name = image_data.name
        image_path = osp.join(image_dir, image_name)

        image_names.append(image_name)
        image_files.append(image_path)

        # Get camera parameters
        camera = colmap_cameras[image_data.camera_id]

        # Convert quaternion to rotation matrix
        R = image_data.qvec2rotmat()
        t = image_data.tvec

        # Create extrinsic matrix (world to camera)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        extrinsics.append(extrinsic)

        # Create intrinsic matrix
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            fx = fy = f
        else:
            # For other models, use basic pinhole approximation
            fx = fy = camera.params[0] if len(camera.params) > 0 else 1000
            cx = camera.width / 2
            cy = camera.height / 2

        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    return image_names, image_files, extrinsics, intrinsics


def apply_color_map(normalized_depth):
    colored_depth = Visualizers.float_colormap(torch.from_numpy(normalized_depth).unsqueeze(0))
    colored_depth = (colored_depth.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    return colored_depth


def store_depth_and_set_scale(depth, image_name, output_dir, image_saver, depth_colored_saver=None):
    max_depth = depth.max()

    depth_uint16 = np.clip((depth / max_depth) * 65535.0, a_min=0, a_max=65535).astype(np.uint16)

    scale_output_path = osp.join(output_dir, "{}.scale.npy".format(image_name))
    os.makedirs(osp.dirname(scale_output_path), exist_ok=True)
    np.save(scale_output_path, max_depth)
    image_saver.save(depth_uint16, osp.join(output_dir, "{}.uint16.png".format(image_name)))

    if depth_colored_saver is not None:
        # apply colormap
        min_depth = depth.min()
        normalized_depth = (depth - min_depth) / (max_depth - min_depth)
        depth_colored_saver.save(
            normalized_depth,
            osp.join(output_dir, "{}.preview.png".format(image_name)),
            processor=apply_color_map,
        )


def main():
    args = parse_args()

    # set output dir
    base_output_dir_prefix = osp.join(args.dataset_dir, args.name)
    depth_output_dir = "{}_depths".format(base_output_dir_prefix)
    inverse_depth_output_dir = "{}_inverse_depths".format(base_output_dir_prefix)
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(inverse_depth_output_dir, exist_ok=True)

    # load colmap sparse
    print("loading colmap sparse model...")
    colmap_cameras = read_cameras_binary(osp.join(args.dataset_dir, "sparse", "cameras.bin"))
    colmap_images = read_images_binary(osp.join(args.dataset_dir, "sparse", "images.bin"))
    image_id_list = sorted(list(colmap_images.keys()))
    image_name_to_id = {colmap_images[i].name: i for i in image_id_list}
    n_images = len(image_id_list)

    # Group images by camera_id
    image_id_list_group_by_camera_id = {}
    for image_id in image_id_list:
        cam_id = colmap_images[image_id].camera_id
        image_id_list_group_by_camera_id.setdefault(cam_id, []).append(image_id)
    print(f"Total images: {n_images}, cameras: {len(image_id_list_group_by_camera_id)}")

    # Build per-camera data
    image_data_group_by_camera_id = {}
    for cam_id, cam_image_ids in image_id_list_group_by_camera_id.items():
        image_data_group_by_camera_id[cam_id] = get_image_data(
            cam_image_ids,
            colmap_images=colmap_images,
            colmap_cameras=colmap_cameras,
            image_dir=osp.join(args.dataset_dir, args.image_dir),
        )

    # load Depth Anything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")  # "depth-anything/DA3NESTED-GIANT-LARGE"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # start inferring
    batch_size = args.batch_size
    depth_saver = AsyncImageSaver()
    depth_colored_saver = AsyncImageSaver(is_rgb=True) if args.preview else None

    t = tqdm(total=n_images)
    try:
        for (
            image_names,
            image_files,
            extrinsics,
            intrinsics,
        ) in image_data_group_by_camera_id.values():
            for i in range(0, len(image_names), batch_size):
                slice_end = i + batch_size
                prediction = None

                image_file_slice = image_files[i:slice_end]
                try:
                    prediction = model.inference(
                        image=image_file_slice,
                        extrinsics=extrinsics[i:slice_end],  # (N, 4, 4)
                        intrinsics=intrinsics[i:slice_end],  # (N, 3, 3)
                        align_to_input_ext_scale=True,
                        infer_gs=False,
                        process_res=args.input_size,
                    )
                except:
                    import traceback

                    traceback.print_exc()
                    print("batch skipped")
                    continue
                finally:
                    t.update(len(image_file_slice))

                for idx, image_name in zip(range(prediction.depth.shape[0]), image_names[i:slice_end]):
                    depth = prediction.depth[idx]
                    if args.keep_resolution:
                        colmap_camera = colmap_cameras[colmap_images[image_name_to_id[image_name]].camera_id]
                        depth = torch.from_numpy(depth)
                        depth = torch.nn.functional.interpolate(
                            depth[None, None],
                            size=(colmap_camera.height, colmap_camera.width),
                            mode="bilinear",
                            align_corners=True,
                        )[0, 0]
                        depth = depth.cpu().numpy()
                    inverse_depth = 1.0 / depth
                    inverse_depth = np.nan_to_num(inverse_depth, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                    store_depth_and_set_scale(
                        depth=depth,
                        image_name=image_name,
                        output_dir=depth_output_dir,
                        image_saver=depth_saver,
                        depth_colored_saver=depth_colored_saver,
                    )
                    store_depth_and_set_scale(
                        depth=inverse_depth,
                        image_name=image_name,
                        output_dir=inverse_depth_output_dir,
                        image_saver=depth_saver,
                        depth_colored_saver=depth_colored_saver,
                    )
    finally:
        t.close()
        depth_saver.stop()
        depth_colored_saver.stop() if depth_colored_saver is not None else None

    subprocess.call(
        [
            "python",
            osp.join(osp.dirname(__file__), "get_da3_depth_scales.py"),
            args.dataset_dir,
            "--name={}".format(osp.basename(args.name)),
        ]
    )


if __name__ == "__main__":
    main()
