import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
# from diff_gaussian_rasterization_ms import (GaussianRasterizationSettings,
#                                             GaussianRasterizer)
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)

from internal.cameras import Camera, Cameras
from internal.models.gaussian import GaussianModel
from internal.utils.sh_utils import eval_sh


def rasterize_importance(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    scaling_modifier: float = 1.0,
    compute_cov3D_python: bool = False,
    override_opacity: Optional[torch.Tensor] = None,
    override_scale: Optional[torch.Tensor] = None,
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=bg_color.device) + 0

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
    tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=None,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_to_camera,
        projmatrix=viewpoint_camera.full_projection,
        sh_degree=None,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    _opacities = override_opacity if override_opacity is not None else pc.get_opacity
    _scales = override_scale if override_scale is not None else pc.get_scaling

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = _opacities

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if compute_cov3D_python is True:
        raise NotImplementedError("compute_cov3D_python is not supported in RaDeGSRendererModule")
    else:
        rotations = pc.get_rotation
        scales = _scales

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    num_rendered, radii, accum_weights, num_hit_pixels, num_max_pixels = rasterizer.rasterize_importance(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "accum_weights": accum_weights,
        "num_hit_pixels": num_hit_pixels,
        "num_max_pixels": num_max_pixels,
    }


class ImportanceMixin:
    def rasterize_importance(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        scaling_modifier: float = 1.0,
        *args,
        **kwargs,
    ):
        _opacities, _scales = self._get_opacities_and_scales(pc)

        return rasterize_importance(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            scaling_modifier=scaling_modifier,
            compute_cov3D_python=self.compute_cov3D_python,
            override_opacity=_opacities,
            override_scale=_scales,
        )
