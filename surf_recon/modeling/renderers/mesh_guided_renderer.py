import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)

from internal.cameras import Camera, Cameras
from internal.models.mip_splatting import MipSplattingModel
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.renderers import (RendererConfig, RendererOutputInfo,
                                RendererOutputTypes)
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.sh_utils import eval_sh

from ...utils.mesh import Meshes
from .nvdr import NVDRRasterizationConfig, NVDRRendererMixin


class MeshGuidedRendererModule(VanillaRenderer):
    def forward(
        self,
        viewpoint_camera: Camera,
        pc: VanillaGaussianModel,
        bg_color: Optional[torch.Tensor] = None,
        scaling_modifier: float = 1.0,
        override_color=None,
        render_types: list = None,
    ):
        if bg_color is None:
            bg_color = torch.ones(3, device=viewpoint_camera.device, dtype=viewpoint_camera.full_projection.dtype)

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

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python is True:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

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

        zbuf = self.rasterize_to_zbuf(viewpoint_camera)
        if zbuf is not None:
            zbuf = zbuf.to(means3D)
        tolerance = self.get_tolerance()

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer.rasterize_with_zbuf(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            zbuf=zbuf,
            tolerance=tolerance,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


@dataclass
class NVDRGuidedRenderer(RendererConfig):
    nvdr_config: NVDRRasterizationConfig = field(default_factory=lambda: NVDRRasterizationConfig())

    def instantiate(self, mesh, tolerance, *args, **kwargs):
        return NVDRGuidedRendererModule(self, mesh, tolerance, *args, **kwargs)


class NVDRGuidedRendererModule(MeshGuidedRendererModule, NVDRRendererMixin):
    def __init__(self, config: NVDRGuidedRenderer, mesh: Optional[Meshes], tolerance: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.mesh = mesh
        self.tolerance = tolerance

    def setup(self, stage: str, *args, **kwargs):
        super().setup(stage, use_opengl=self.config.nvdr_config.use_opengl, *args, **kwargs)

    def get_tolerance(self):
        return self.tolerance

    def rasterize_to_zbuf(self, viewpoint_camera: Camera):
        if self.mesh is None:
            return None
        mesh = self.cull_mesh(self.mesh, viewpoint_camera)
        rast_out, _ = self.nvdiff_rasterization(viewpoint_camera, mesh.verts, mesh.faces, self._gl_context)
        return rast_out.zbuf.squeeze()


@dataclass
class Open3DGuidedRenderer(RendererConfig):
    def instantiate(self, mesh, tolerance, *args, **kwargs):
        return Open3DGuidedRendererModule(mesh, tolerance, *args, **kwargs)


class Open3DGuidedRendererModule(MeshGuidedRendererModule):
    config: Open3DGuidedRenderer

    def __init__(self, mesh: Optional[o3d.geometry.TriangleMesh], tolerance: float, resolution: Tuple[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh = mesh
        self.tolerance = tolerance
        self.image_width, self.image_height = resolution

    def _init(self):
        renderer = o3d.visualization.rendering.OffscreenRenderer(self.image_width, self.image_height)
        renderer.scene.set_background([0, 0, 0, 1])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        renderer.scene.add_geometry("mesh", self.mesh, mat)
        self._o3d_renderer = renderer

    def setup(self, stage: str, *args, **kwargs):
        self._init()
        super().setup(stage, *args, **kwargs)

    def get_tolerance(self):
        return self.tolerance

    def rasterize_to_zbuf(self, camera: Camera):
        if self.mesh is None:
            return None
        image_width, image_height = int(camera.width), int(camera.height)
        if (image_width, image_height) != (self.image_width, self.image_height):
            self._init()

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.image_width,
            height=self.image_height,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
        )
        extrinsic = camera.world_to_camera.T.cpu().numpy().astype(np.float64)
        self._o3d_renderer.setup_camera(intrinsics=intrinsic, extrinsic_matrix=extrinsic)
        self._o3d_renderer.scene.camera.set_projection(
            intrinsics=intrinsic.intrinsic_matrix,
            near_plane=0.01,
            far_plane=100.0,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        zbuf = self._o3d_renderer.render_to_depth_image()
        return torch.from_numpy(np.asarray(zbuf)).squeeze()
