import os
import os.path as osp


def get_project_dir(project: str) -> str:
    return osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "outputs", project)


def get_partition_info_dir(project: str) -> str:
    return osp.join(get_project_dir(project), "partition_info")


def get_cells_output_dir(project: str) -> str:
    return osp.join(get_project_dir(project), "cells")


def get_trained_cells_dir(project: str) -> str:
    return osp.join(get_project_dir(project), "trained")


def get_cell_partition_info_dir(part_info_dir: str, cell_name: str):
    return osp.join(part_info_dir, "cells", cell_name)


def get_project_ckpt_dir(project: str) -> str:
    return osp.join(get_project_dir(project), "checkpoints")
