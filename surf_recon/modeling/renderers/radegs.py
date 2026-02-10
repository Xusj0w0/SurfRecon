import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch
from diff_gaussian_rasterization_radegs import (GaussianRasterizationSettings,
                                                GaussianRasterizer)

from internal.cameras import Camera, Cameras
from internal.models.mip_splatting import MipSplattingModel
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers.renderer import (RendererConfig, RendererOutputInfo,
                                         RendererOutputTypes)
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.sh_utils import eval_sh


def rasterize_radegs(
    viewpoint_camera: Camera,
    pc: Union[VanillaGaussianModel, MipSplattingModel],
    bg_color: Optional[torch.Tensor],
    scaling_modifier: float = 1.0,
    override_color=None,
    render_types: list = None,
    filter_2d_kernel_size: float = 0.0,
    compute_cov3D_python: bool = False,
    convert_SHs_python: bool = False,
):
    """
    This function is adopted from RaDe-GS:
        https://github.com/BaowenZ/RaDe-GS/blob/main/gaussian_renderer/__init__.py
    """
    if render_types is None:
        render_types = ["rgb"]
    # Determine which render types are requested
    coord_required = "coord" in render_types
    depth_required = "depth" in render_types or "median_depth" in render_types
    normal_required = "normal" in render_types
    # Normal map is rendered alongside coord map and depth map
    if normal_required and not coord_required and not depth_required:
        render_types.append("depth")
        depth_required = True

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz,
            dtype=pc.get_xyz.dtype,
            requires_grad=True,
            device=bg_color.device,
        )
        + 0
    )

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
    tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=filter_2d_kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_to_camera,
        projmatrix=viewpoint_camera.full_projection,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_coord=coord_required,
        require_depth=depth_required,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if hasattr(pc, "get_3d_filtered_scales_and_opacities"):
        _opacities, _scales = pc.get_3d_filtered_scales_and_opacities()
    else:
        _opacities, _scales = pc.get_opacity, pc.get_scaling
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
    shs = None
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
    else:
        colors_precomp = override_color

    (
        rendered_image,
        radii,
        num_hit_pixels,
        rendered_expected_coord,
        rendered_median_coord,
        rendered_expected_depth,
        rendered_median_depth,
        rendered_alpha,
        rendered_normal,
    ) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "mask": rendered_alpha,
        "expected_coord": rendered_expected_coord if coord_required else None,
        "median_coord": rendered_median_coord if coord_required else None,
        "expected_depth": rendered_expected_depth if depth_required else None,
        "median_depth": rendered_median_depth if depth_required else None,
        "normal": rendered_normal if normal_required else None,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "num_hit_pixels": num_hit_pixels,
    }


@dataclass
class RaDeGSRenderer(RendererConfig):
    filter_2d_kernel_size: float = 0.0

    default_background: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    "Default background color for rendering when `bg_color` not provided. [0.0, 1.0]"

    def instantiate(self, *args, **kwargs) -> "RaDeGSRendererModule":
        assert len(self.default_background) == 3 and all(
            0.0 <= c <= 1.0 for c in self.default_background
        ), "default_background must be a list of 3 floats within [0, 1]."
        return RaDeGSRendererModule(self, *args, **kwargs)


class RaDeGSRendererModule(VanillaRenderer):
    def __init__(self, config: RaDeGSRenderer, *args, **kwargs):
        self.config = config
        super().__init__(*args, **kwargs)

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: Union[VanillaGaussianModel, MipSplattingModel],
        bg_color: Optional[torch.Tensor] = None,
        scaling_modifier: float = 1.0,
        override_color=None,
        render_types: list = None,
    ):
        """
        This function is adopted from RaDe-GS:
            https://github.com/BaowenZ/RaDe-GS/blob/main/gaussian_renderer/__init__.py
        """
        if bg_color is None:
            bg_color = torch.tensor(self.config.default_background, device=pc.get_xyz.device)

        return rasterize_radegs(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            render_types=render_types,
            filter_2d_kernel_size=self.config.filter_2d_kernel_size,
            compute_cov3D_python=self.compute_cov3D_python,
            convert_SHs_python=self.convert_SHs_python,
        )

    def integrate(
        self,
        points3D: torch.Tensor,
        viewpoint_camera: Camera,
        pc: Union[VanillaGaussianModel, MipSplattingModel],
        bg_color: Optional[torch.Tensor] = None,
        scaling_modifier: float = 1.0,
        override_color=None,
    ):
        """
        This function is adopted from GOF for marching tetrahedra:
            https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/gaussian_renderer/__init__.py
        """
        is_mip = MipSplattingModel._filter_3d_name in pc.get_property_names()
        if bg_color is None:
            bg_color = torch.tensor(self.config.default_background, device=pc.get_xyz.device)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz,
                dtype=pc.get_xyz.dtype,
                requires_grad=True,
                device=bg_color.device,
            )
            + 0
        )

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=float(self.config.filter_2d_kernel_size),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            require_coord=True,
            require_depth=True,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        if is_mip:
            _opacity, _scales = pc.get_3d_filtered_scales_and_opacities()
        else:
            _opacity, _scales = pc.get_opacity, pc.get_scaling

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = _opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python is True:
            raise NotImplementedError("compute_cov3D_python is not supported in RaDeGSRendererModule")
        else:
            rotations = pc.get_rotation
            scales = _scales

        depth_plane_precomp = None

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if self.convert_SHs_python is True:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        (
            rendered_image,
            alpha_integrated,
            color_integrated,
            point_coordinate,
            point_sdf,
            radii,
        ) = rasterizer.integrate(
            points3D=points3D,
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            view2gaussian_precomp=depth_plane_precomp,
        )

        return {
            "render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
            "point_coordinate": point_coordinate,
            "point_sdf": point_sdf,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_outputs(self):
        return {
            "rgb": RendererOutputInfo("render"),
            "depth": RendererOutputInfo("expected_depth", RendererOutputTypes.GRAY),
            "median_depth": RendererOutputInfo("median_depth", RendererOutputTypes.GRAY),
            "normal": RendererOutputInfo("normal", RendererOutputTypes.NORMAL_MAP),
        }
