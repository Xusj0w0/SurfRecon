import argparse
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

from internal.utils.colmap import (Camera, Image, Point3D, read_model,
                                   write_model)


def make_parser():
    parser = argparse.ArgumentParser(description="Convert CityGS-format dataset to Gaussian Splatting Lightning")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--down_sample_factor", type=int, default=1)
    parser.add_argument("--rescale_width", type=int, default=-1)
    parser.add_argument("--ext", nargs="+", default=["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"])
    parser.add_argument("--skip_image", default=False, action="store_true")
    parser.add_argument("--skip_sparse", default=False, action="store_true")
    return parser


def load_sparse_model(sparse_model_path):
    sparse_model = read_model(sparse_model_path)
    if sparse_model is None:
        sparse_model = read_model(osp.join(sparse_model_path, "0"))
    if sparse_model is None:
        raise FileNotFoundError("Sparse model not founded.")

    return sparse_model


def find_images(path: str, extensions: list) -> list:
    image_list = []
    for extension in extensions:
        image_list += Path(path).rglob("*.{}".format(extension))
    return image_list


def get_resized_size(width, height, down_sample_factor: int = 1, rescale_width: int = -1):
    if down_sample_factor > 0:
        resized_width, resized_height = round(width / down_sample_factor), round(height / down_sample_factor)
    elif rescale_width > 0:
        if width > height:
            resized_width = int(rescale_width)
            resized_height = round(float(height) / width * rescale_width)
        else:
            resized_height = int(rescale_width)
            resized_width = round(float(width) / height * rescale_width)
    else:
        raise NotImplementedError
    return resized_width, resized_height


def resize_image(image_path: str, dst_path: str, down_sample_factor: int = 1, rescale_width: int = -1):
    image = PILImage.open(image_path)

    width, height = image.size
    resized_width, resized_height = get_resized_size(width, height, down_sample_factor, rescale_width)

    resized_image = image.resize((resized_width, resized_height))

    os.makedirs(osp.dirname(dst_path), exist_ok=True)
    resized_image.save(dst_path, quality=100)


def main():
    args = make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(osp.join(args.output_dir, "image_info"), "w") as f:
        if args.rescale_width > 0:
            f.write(f"Rescale long edge of images to {args.rescale_width} pixels")
            args.down_sample_factor = -1
        else:
            f.write(f"Down sample images by {args.down_sample_factor} times")

    # preprocess images
    if not args.skip_image:
        src_dir = osp.join(args.input_dir, "images")
        dst_dir = osp.join(args.output_dir, "images")
        image_list: List[Path] = find_images(src_dir, args.ext)
        with ThreadPoolExecutor() as tpe:
            future_list = []
            for src_path in image_list:
                dst_path = Path(dst_dir) / src_path.relative_to(src_dir)
                dst_path = dst_path.with_suffix(".png")
                future_list.append(
                    tpe.submit(
                        resize_image,
                        str(src_path.absolute()),
                        str(dst_path.absolute()),
                        args.down_sample_factor,
                        args.rescale_width,
                    )
                )

            with tqdm(total=len(future_list), desc="Resizing images", mininterval=2.0) as pbar:
                for _ in as_completed(future_list):
                    pbar.update(1)

    # merge sparse model
    if not args.skip_sparse:
        cameras, images, points3d = load_sparse_model(osp.join(args.input_dir, "sparse"))

        # rescaling images
        camera_scalings, _cameras = {}, {}
        for camera_id, camera in cameras.items():
            resized_width, resized_height = get_resized_size(camera.width, camera.height, args.down_sample_factor, args.rescale_width)
            scaling_x, scaling_y = float(resized_width) / camera.width, float(resized_height) / camera.height
            camera_scalings[camera.id] = (scaling_x, scaling_y)

            _params = deepcopy(camera.params)
            _params[0::2] *= scaling_x
            _params[1::2] *= scaling_y
            _cameras[camera_id] = Camera(
                id=camera.id,
                model=camera.model,
                width=resized_width,
                height=resized_height,
                params=_params,
            )
        _images = {}
        for image_id, image in images.items():
            _xys = image.xys
            _xys[:, 0] *= camera_scalings[image.camera_id][0]
            _xys[:, 1] *= camera_scalings[image.camera_id][1]
            _images[image_id] = Image(
                id=image.id,
                qvec=image.qvec,
                tvec=image.tvec,
                camera_id=image.camera_id,
                name=str(Path(image.name).with_suffix(".png")),
                xys=_xys,
                point3D_ids=image.point3D_ids,
            )
        cameras, images = _cameras, _images

        # write model
        sparse_model_path = osp.join(args.output_dir, "sparse")
        os.makedirs(sparse_model_path, exist_ok=True)
        write_model(cameras, images, points3d, sparse_model_path)


if __name__ == "__main__":
    main()
