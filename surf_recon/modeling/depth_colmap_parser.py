import json
import os
import os.path as osp
from dataclasses import dataclass, field

import numpy as np

from internal.dataparsers import DataParserConfig, DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers.estimated_depth_colmap_dataparser import (
    EstimatedDepthColmap, EstimatedDepthColmapDataParser)


@dataclass
class DepthColmap(EstimatedDepthColmap):
    depth_dir: str = field(default="da3_inverse_depths")

    depth_scale_name: str = field(default="da3_inverse_depths-scales")

    depth_in_uint16: bool = field(default=True)

    def instantiate(self, path, output_path, global_rank):
        return EstimatedDepthColmapDataParser(path, output_path, global_rank, self)
