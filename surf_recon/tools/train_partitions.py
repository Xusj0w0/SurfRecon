import argparse
import json
import os
import os.path as osp
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import py3nvml.py3nvml as nvml

from surf_recon.utils.path_utils import (get_cell_partition_info_dir,
                                         get_cells_output_dir,
                                         get_partition_info_dir,
                                         get_trained_cells_dir)
from utils.argparser_utils import parser_stoppable_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", "-p", type=str, required=True, help="Project name. Output path will be `outputs/{project}`")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config file.")
    parser.add_argument("--gpu_indices", type=str, default="0,1,2,3,4,5,6,7")
    args, training_args = parser_stoppable_args(parser)
    return args, training_args


def train_a_cell(project, cell, config_path, training_args, free_gpu):
    cell_idx, cell_name = cell
    gpu_id, gpu_uuid = free_gpu
    output_path = get_cells_output_dir(project)
    partition_info_dir = get_partition_info_dir(project)

    # fmt: off
    args = [
        "python", "main.py", "fit",
        "--config={}".format(config_path),
        "--output={}".format(output_path),
        "--name={}".format(cell_name),
        "--data.parser.image_list={}".format(osp.join(get_cell_partition_info_dir(partition_info_dir, cell_name), "image_list.txt")),
        "--model.initialize_from={}".format(osp.join(partition_info_dir, "gaussians.ply")),
        "--logger=tensorboard",
    ]
    # fmt: on
    args += training_args

    os.makedirs(os.path.join(output_path, cell_name), exist_ok=True)
    with open(os.path.join(output_path, cell_name, "command.txt"), "w") as f:
        f.write(" ".join(args) + "\n")
    with open(os.path.join(output_path, cell_name, "train.log"), "w") as f:
        f.write(" ".join(args) + "\n")

        ret_code = subprocess.run(
            args,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "TQDM_MININTERVAL": "5",
                "TQDM_MINITERS": "200",
            },
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    if ret_code.returncode == 0:
        trained_cells_dir = get_trained_cells_dir(project)
        os.makedirs(trained_cells_dir, exist_ok=True)
        with open(os.path.join(trained_cells_dir, f"{cell_name}.txt"), "w") as f:
            f.write("Trained")


def get_free_gpus(
    valid_gpu_ids: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
    max_vram_usage: int = 1000,
) -> List[int]:
    """
    Args:
        valid_gpu_ids (List): NVML GPU indices to consider (e.g., [0,1,2,3])
        max_vram_usage (int): in MiB
    Returns:
        available_gpus (List): List of GPU UUIDs that have less than `max_vram_usage` MB used VRAM.
    """
    MiB = 1024 * 1024
    threshold = max_vram_usage * MiB
    n = nvml.nvmlDeviceGetCount()
    valid_ids = [i for i in valid_gpu_ids if 0 <= i < n]
    available_gpus = []
    for i in valid_ids:
        h = nvml.nvmlDeviceGetHandleByIndex(i)
        mem = nvml.nvmlDeviceGetMemoryInfo(h)
        if mem.used < threshold:
            uuid = nvml.nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")
            available_gpus.append((i, uuid))
    return available_gpus


def main():
    args, training_args = parse_args()

    assert Path(args.config).exists(), f"Config file {args.config} does not exist."
    gpu_indices = [int(i) for i in args.gpu_indices.split(",")]

    part_info_dir = Path(get_partition_info_dir(args.project))
    metadata = json.load(open(str((part_info_dir / "metadata.json").absolute()), "r"))
    num_cells = len(metadata["cells"])

    nvml.nvmlInit()
    tasks = []
    with ProcessPoolExecutor(max_workers=num_cells) as ex:
        for cell_idx in range(num_cells):
            name = metadata["cells"][cell_idx]["name"]
            gpu_available = False
            fail_cnt = 0
            while not gpu_available:
                free_gpus = get_free_gpus(valid_gpu_ids=gpu_indices)
                if len(free_gpus) > 0:
                    gpu_available = True
                elif fail_cnt >= 240:
                    print("No free GPUs available in 8 hour, exiting...")
                else:
                    fail_cnt += 1
                    print("No free GPUs available, waiting for 2 minutes...")
                    time.sleep(120)

            gpu_id, gpu_uuid = free_gpus[-1]
            tasks.append(
                ex.submit(
                    train_a_cell,
                    args.project,
                    (cell_idx, name),
                    args.config,
                    training_args,
                    (gpu_id, gpu_uuid),
                )
            )
            time.sleep(120)

        for f in as_completed(tasks):
            f.result()


if __name__ == "__main__":
    main()
