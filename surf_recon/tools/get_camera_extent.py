import argparse
import json

import numpy as np

from internal.utils.graphics_utils import get_center_and_diag


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_json", type=str, help="Path to camera json file.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.camera_json, "r") as f:
        camera_data = json.load(f)

    cam_centers = []
    for cam in camera_data:
        cam_centers.append(np.array(cam["position"]).reshape(3, 1))

    center, diagonal = get_center_and_diag(cam_centers)

    print("Camera Center:", center)
    print("Camera Extent (Diagonal):", diagonal)


if __name__ == "__main__":
    main()
