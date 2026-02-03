import argparse
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import numpy as np

import internal.utils.colmap as colmap_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="", help="Path to transformation text file.")
    args = parser.parse_args()
    if len(args.output_path) <= 0 or not osp.exists(args.output_path):
        args.output_path = osp.join(args.dataset_path, "transforms.txt")
    return args


def contain_colmap_files(sparse_path: str) -> bool:
    required_files = ["cameras", "images", "points3D"]
    valid_exts = [".txt", ".bin"]
    for ext in valid_exts:
        if all(osp.exists(osp.join(sparse_path, f + ext)) for f in required_files):
            return True
    return False


def rotation_matrix_align_vectors(
    normal: np.ndarray,
    target_direction: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute a 3x3 rotation matrix R such that:
        R @ n_hat = t_hat
    where n_hat = normal / ||normal|| and t_hat = target_direction / ||target_direction||.

    Robustly handles:
      - nearly parallel vectors (returns identity),
      - nearly opposite vectors (returns 180° rotation around a perpendicular axis).

    Args:
        normal: (3,) source direction.
        target_direction: (3,) target direction.
        eps: small number for numerical stability.

    Returns:
        R: (3,3) rotation matrix.
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    t = np.asarray(target_direction, dtype=np.float64).reshape(3)

    n_norm = np.linalg.norm(n)
    t_norm = np.linalg.norm(t)
    if n_norm < eps or t_norm < eps:
        raise ValueError("normal and target_direction must be non-zero vectors.")

    n = n / n_norm
    t = t / t_norm

    v = np.cross(n, t)
    s = np.linalg.norm(v)
    c = float(np.dot(n, t))  # cos(theta)

    # Case 1: already aligned (theta ~ 0)
    if s < eps and c > 0:
        return np.eye(3, dtype=np.float64)

    # Case 2: opposite direction (theta ~ pi)
    if s < eps and c < 0:
        # Choose an axis perpendicular to n (and thus also to t = -n)
        # Pick any vector not parallel to n, then cross to get perpendicular axis.
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(n[0]) > 0.9:
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(n, a)
        axis = axis / (np.linalg.norm(axis) + eps)

        # 180-degree rotation around 'axis':
        # R = 2 aa^T - I
        R = 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
        return R

    # General case: Rodrigues' rotation formula
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )

    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return R


def print_rotmat(rotmat: np.ndarray, precision: int = 6):
    return "\n".join([" ".join([f"{elem:.{precision}f}" for elem in line]) for line in rotmat.tolist()])


def main():
    args = parse_args()

    # Find COLMAP sparse model path
    sparse_path = osp.join(args.dataset_path, "sparse")
    if not contain_colmap_files(sparse_path):
        sparse_path = osp.join(sparse_path, "0")
    if not contain_colmap_files(sparse_path):
        raise FileNotFoundError(f"COLMAP sparse model files not found in {sparse_path}.")

    # Load COLMAP images
    images: Dict[int, colmap_utils.Image]
    if osp.exists(osp.join(sparse_path, "images.bin")):
        images = colmap_utils.read_images_binary(osp.join(sparse_path, "images.bin"))
    else:
        images = colmap_utils.read_images_text(osp.join(sparse_path, "images.txt"))

    # Fit camera centers to plane
    cam_centers = []
    for image in images.values():
        # world to camera
        R = image.qvec2rotmat()
        t = image.tvec
        cam_center = -R.T @ t
        cam_centers.append(cam_center.flatten())
    cam_centers = np.stack(cam_centers, axis=0)
    centroid = np.mean(cam_centers, axis=0)
    centered_campos = cam_centers - centroid
    _, _, Vt = np.linalg.svd(centered_campos, full_matrices=False)
    normal = Vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # Create reorientation matrix
    rotmat = rotation_matrix_align_vectors(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    print("Reorientation rotation matrix:\n{}".format(print_rotmat(rotmat, precision=6)))


if __name__ == "__main__":
    main()
