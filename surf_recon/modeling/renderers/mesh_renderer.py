import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch

from internal.cameras import Camera, Cameras
from internal.models.mip_splatting import MipSplattingModel
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers import RendererOutputInfo, RendererOutputTypes

from ...utils.appearance_model import DecoupledAppearanceModelConfig
from .importance import ImportanceMixin
from .nvdr import NVDRRasterizationConfig, NVDRRendererMixin
from .radegs import RaDeGSRenderer, RaDeGSRendererModule


@dataclass
class MeshBindedRenderer(RaDeGSRenderer):
    nvdr_config: NVDRRasterizationConfig = field(default_factory=lambda: NVDRRasterizationConfig())

    appearance_model: DecoupledAppearanceModelConfig = field(default_factory=lambda: DecoupledAppearanceModelConfig())

    def instantiate(self, *args, **kwargs):
        return MeshBindedRendererModule(self)


class MeshBindedRendererModule(NVDRRendererMixin, RaDeGSRendererModule, ImportanceMixin):
    config: MeshBindedRenderer

    def forward(self, viewpoint_camera: Camera, pc, bg_color: torch.Tensor = None, render_types: list = None, *args, **kwargs):
        if render_types is None:
            render_types = ["rgb"]
        gs_render_types, mesh_render_types = [], []
        for t in render_types:
            if t.startswith("mesh"):
                mesh_render_types.append(t)
            else:
                gs_render_types.append(t)

        output_pkg = super().forward(viewpoint_camera, pc, bg_color=bg_color, render_types=gs_render_types, *args, **kwargs)
        if len(mesh_render_types) > 0:
            mesh = getattr(pc, "mesh", None)
            if mesh is None:
                mesh = pc.extract_mesh()
            output_pkg.update(
                self.render_mesh(
                    viewpoint_camera,
                    mesh,
                    render_types=mesh_render_types,
                    anti_aliased=self.config.nvdr_config.anti_aliased,
                    check_errors=self.config.nvdr_config.check_errors,
                    *args,
                    **kwargs,
                )
            )

        # Augmented appearance
        output_pkg["render_aug"] = None

        return output_pkg

    def training_forward(
        self,
        step: int,
        module,
        viewpoint_camera: Camera,
        pc,
        bg_color: torch.Tensor,
        render_types: list = None,
        **kwargs,
    ):
        outputs = super().training_forward(step, module, viewpoint_camera, pc, bg_color, render_types, **kwargs)

        # Augment with appearance model if exists
        render = outputs.get("render", None)
        if hasattr(self, "appearance_model") and render is not None:
            render_aug = self.appearance_model(render.unsqueeze(0), viewpoint_camera.appearance_id).squeeze(0)
            outputs["render_aug"] = render_aug
        return outputs

    def setup(self, stage: str, lightning_module=None, *args, **kwargs):
        super().setup(stage=stage, lightning_module=lightning_module, use_opengl=self.config.nvdr_config.use_opengl, *args, **kwargs)
        if stage == "fit" and self.config.appearance_model.enabled:
            self.appearance_model = self.config.appearance_model.instantiate()
            self.appearance_model.setup(stage, lightning_module)

    def training_setup(self, module):
        optimizers, schedulers = super().training_setup(module)
        if hasattr(self, "appearance_model"):
            _optimizers, _schedulers = self.appearance_model.training_setup(module)
            if optimizers is None:
                optimizers = []
            optimizers.extend(_optimizers)
            if _schedulers is not None:
                if schedulers is None:
                    schedulers = []
                schedulers.extend(_schedulers)

        return optimizers, schedulers

    def get_available_outputs(self):
        available_outputs = super().get_available_outputs()
        available_outputs.update(
            {
                "mesh_rgb": RendererOutputInfo("mesh_rgb"),
                "mesh_depth": RendererOutputInfo("mesh_depth", RendererOutputTypes.GRAY),
                "mesh_normal": RendererOutputInfo("mesh_normal", RendererOutputTypes.NORMAL_MAP),
            }
        )
        return available_outputs

    def before_training_step(self, step, module):
        config = module.metric.config
        if step > config.dn_start_iter:
            module.renderer_output_types = ["rgb", "depth", "normal"]
        if step > config.mesh_regularization_schedule.start_iter:
            module.renderer_output_types = ["rgb", "depth", "normal", "mesh_depth", "mesh_normal"]
            module.gaussian_model.extract_mesh()

        return super().before_training_step(step, module)

    def load_state_dict(self, state_dict, strict=True):
        if not hasattr(self, "appearance_model"):
            keys_to_remove = []
            for key in state_dict.keys():
                if key.startswith("appearance_model."):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del state_dict[key]
        return super().load_state_dict(state_dict, strict)
