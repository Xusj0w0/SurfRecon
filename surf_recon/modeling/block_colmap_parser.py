import json
import os
import os.path as osp
from dataclasses import dataclass, field

import numpy as np
import torch

from internal.dataparsers import DataParserConfig, DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers.estimated_depth_colmap_dataparser import (
    EstimatedDepthColmap, EstimatedDepthColmapDataParser)

from ..utils.path_utils import get_cell_partition_info_dir


@dataclass
class BlockMixin:
    partition_info_dir: str = ""

    cell_name: str = ""


class BlockDataParserMixin:
    def _get_bounding_box(self):
        # metadata
        _transforms, _bounding_box = None, None
        metadata_path = osp.join(self.params.partition_info_dir, "metadata.json")
        if osp.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            _transforms = metadata["scene"]["transforms"]
            try:
                _bounding_box = [m["bbox"] for m in metadata["cells"] if m["name"] == self.params.cell_name][0]
            except:
                _bounding_box = None
        return _transforms, _bounding_box

    def get_outputs(self) -> DataParserOutputs:
        cell_part_info_dir = get_cell_partition_info_dir(self.params.partition_info_dir, self.params.cell_name)
        if not osp.exists(cell_part_info_dir):
            return

        # image_list
        image_list_path = osp.join(cell_part_info_dir, "image_list.txt")
        if osp.exists(image_list_path):
            self.params.image_list = image_list_path

        outputs = super().get_outputs()
        outputs._bounding_box = self._get_bounding_box()
        return outputs


@dataclass
class BlockColmap(BlockMixin, Colmap):
    def instantiate(self, path, output_path, global_rank):
        return BlockColmapDataParser(path, output_path, global_rank, self)


class BlockColmapDataParser(BlockDataParserMixin, ColmapDataParser):
    pass


@dataclass
class BlockDepthColmap(BlockMixin, EstimatedDepthColmap):
    depth_dir: str = field(default="da3_depths")
    depth_scale_name: str = field(default="da3_depths-scales")

    def instantiate(self, path, output_path, global_rank):
        return BlockDepthColmapDataParser(path, output_path, global_rank, self)


class BlockDepthColmapDataParser(BlockDataParserMixin, EstimatedDepthColmapDataParser):
    pass
