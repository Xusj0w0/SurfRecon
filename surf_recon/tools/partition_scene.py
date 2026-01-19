import argparse
import os.path as osp
from dataclasses import fields
from typing import Any, Dict

import yaml

from surf_recon.utils.partitionable_scene import (PartitionableScene,
                                                  SceneConfig)
from surf_recon.utils.path_utils import get_partition_info_dir, get_project_dir


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--project", "-p", type=str, required=True)
    parser.add_argument("--dataset_path", "-d", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--scene_bbox_enlarge_by_pts", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--scene_bbox_outlier_by_pts", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--scene_bbox_enlarge_by_campos", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--bbox_enlarge_by_campos", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--bbox_enlarge_by_camvis", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--camera_visibility_threshold", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--bbox_enlarge_by_gaussian_pos", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gaussian_score_prune_ratio", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gpu_idx", type=int, default=0)
    return parser


def load_from_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_dict = yaml.safe_load(f) or {}
    config_fields = {field.name for field in fields(SceneConfig)}
    filtered_dict = {k: v for k, v in yaml_dict.items() if k in config_fields}
    return filtered_dict


def merge_yaml_with_args(yaml_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config_fields = {f.name for f in fields(SceneConfig)}
    args_dict = vars(args)

    override = {k: v for k, v in args_dict.items() if k in config_fields}
    merged = dict(yaml_cfg)
    merged.update(override)
    return merged


def main():
    args = make_parser().parse_args()

    yaml_cfg = load_from_yaml(args.config) if args.config else {}
    merged = merge_yaml_with_args(yaml_cfg, args)

    merged["coarse_model_path"] = osp.join(get_project_dir(args.project), "coarse")
    scene_config = SceneConfig(**merged)
    scene = scene_config.instantiate(gpu_idx=args.gpu_idx)
    scene.run(output_path=get_partition_info_dir(args.project))


if __name__ == "__main__":
    main()
