import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from diff_gaussian_rasterization_ms import (GaussianRasterizationSettings,
                                            GaussianRasterizer)

from internal.cameras import Camera, Cameras
from internal.models.gaussian import GaussianModel
from internal.utils.sh_utils import eval_sh


def render_simp(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    bg_color: Optional[torch.Tensor] = None,
    scaling_modifier: float = 1.0,
    compute_cov3D_python: bool = False,
    convert_SHs_python: bool = False,
    override_color=None,
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
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_to_camera,
        projmatrix=viewpoint_camera.full_projection,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    _opacities = override_opacity if override_opacity is not None else pc.get_opacity
    _scales = override_scale if override_scale is not None else pc.get_scale

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

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc, shs = None, None
    colors_precomp = None
    if override_color is None:
        if convert_SHs_python is True:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            dc, shs = shs[:, :1, :], shs[:, 1:, :]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, accum_weights_ptr, accum_weights_count, accum_max_count = rasterizer.render_simp(
        means3D=means3D,
        means2D=means2D,
        dc=dc,
        shs=shs,
        culling=torch.zeros(means3D.shape[0], dtype=torch.bool, device=means3D.device),
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "accum_weights": accum_weights_ptr,
        "area_proj": accum_weights_count,
        "area_max": accum_max_count,
    }


class ImportanceMixin:
    def render_simp(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        bg_color: Optional[torch.Tensor] = None,
        scaling_modifier: float = 1.0,
        override_color=None,
        *args,
        **kwargs,
    ):
        if bg_color is None:
            bg_color = torch.tensor(self.config.default_background, device=pc.get_xyz.device)

        _opacities, _scales = self._get_opacities_and_scales(pc)

        return render_simp(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            compute_cov3D_python=self.compute_cov3D_python,
            convert_SHs_python=self.convert_SHs_python,
            override_color=override_color,
            override_opacity=_opacities,
            override_scale=_scales,
        )
