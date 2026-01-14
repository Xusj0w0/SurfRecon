import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch

from internal.cameras import Camera, Cameras
from internal.models.mip_splatting import MipSplattingModel
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers import RendererOutputInfo, RendererOutputTypes

from .importance import ImportanceMixin
from .mesh import MeshRasterizationConfig, MeshRendererMixin
from .radegs import RaDeGSRenderer, RaDeGSRendererModule


@dataclass
class FullRaDeGSRenderer(RaDeGSRenderer):
    mesh_rast_config: MeshRasterizationConfig = field(default_factory=lambda: MeshRasterizationConfig())

    def instantiate(self, *args, **kwargs):
        return FullRaDeGSRendererModule(self)


class FullRaDeGSRendererModule(MeshRendererMixin, RaDeGSRendererModule, ImportanceMixin):
    def forward(self, *args, **kwargs):
        render_types = kwargs.pop("render_types", None)
        if render_types is None:
            render_types = ["rgb"]

        gs_render_types, mesh_render_types = [], []
        for t in render_types:
            if t.startswith("mesh"):
                mesh_render_types.append(t)
            else:
                gs_render_types.append(t)

        output_pkg = super().forward(*args, **kwargs, render_types=gs_render_types)
        if len(mesh_render_types) > 0:
            output_pkg.update(self.render_mesh(*args, **kwargs, render_types=mesh_render_types))

        return output_pkg

    def get_available_outputs(self):
        available_outputs = super().get_available_outputs()
        available_outputs.update(
            {
                "mesh_rgb": RendererOutputInfo("mesh_rgb"),
                "mesh_depth": RendererOutputInfo("mesh_depth", RendererOutputTypes.GRAY),
                "mesh_normal": RendererOutputInfo("mesh_normal", RendererOutputTypes.NORMAL_MAP),
            }
        )

    def before_training_step(self, step, module):
        config = module.metric.config
        if step > config.dn_start_iter:
            module.renderer_output_types = ["rgb", "depth", "normal"]
        if step > config.mesh_regularization_schedule.start_iter:
            module.renderer_output_types = ["rgb", "depth", "normal", "mesh_depth", "mesh_normal"]

        return super().before_training_step(step, module)
