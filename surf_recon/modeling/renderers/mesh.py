from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import nvdiffrast.torch as dr
import torch

from internal.cameras import Camera, Cameras
from internal.models.gaussian import GaussianModel
from internal.renderers import RendererConfig

from ...utils.mesh import Meshes


@dataclass
class RasterizerOutputs:
    rast_out: torch.Tensor
    image_width: int = field(init=False)
    image_height: int = field(init=False)

    def __post_init__(self):
        self.image_width = self.rast_out.shape[-2]
        self.image_height = self.rast_out.shape[-3]

    @property
    def bary_coords(self) -> torch.Tensor:
        "Return Tensor [1, H, W, 3]"
        if not hasattr(self, "_bary_coords"):
            bary_coords = self.rast_out[..., :2]
            bary_coords = torch.cat([bary_coords, 1.0 - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)
            self._bary_coords = bary_coords.view(1, self.image_height, self.image_width, 3)
        return self._bary_coords

    @property
    def zbuf(self) -> torch.Tensor:
        "Return Tensor [1, H, W, 1]"
        if not hasattr(self, "_zbuf"):
            self._zbuf = self.rast_out[..., 2].view(1, self.image_height, self.image_width, 1)
        return self._zbuf

    @property
    def pix_to_face(self) -> torch.Tensor:
        "Return Tensor [1, H, W, 1]"
        if not hasattr(self, "_pix_to_face"):
            self._pix_to_face = (self.rast_out[..., 3].int() - 1).view(1, self.image_height, self.image_width, 1)
        return self._pix_to_face

    def fuse(self, rast_out: "RasterizerOutputs"):
        assert (self.image_width, self.image_height) == (rast_out.image_width, rast_out.image_height)
        width, height = self.image_width, self.image_height

        mask1 = self.pix_to_face > -1
        mask2 = rast_out.pix_to_face > -1
        no_raster_mask = (~mask1) & (~mask2)

        # Remove results with larger zbuf at each pixel
        zbuf1 = torch.where(mask1, self.zbuf, 1000.0)
        zbuf2 = torch.where(mask2, rast_out.zbuf, 1000.0)
        all_zbufs = torch.stack([zbuf1, zbuf2], dim=-1)  # [1, H, W, 1, 2]
        zbuf, argzbuf = torch.min(all_zbufs, dim=-1)  # argzbuf [1, H, W, 1]
        zbuf[no_raster_mask] = 0.0

        pix_to_face1, pix_to_face2 = self.rast_out[..., 3].view(1, height, width, 1), rast_out.rast_out[..., 3].view(1, height, width, 1)
        all_pix_to_face = torch.stack([pix_to_face1, pix_to_face2], dim=-1)  # [1, H, W, 1, 2]
        pix_to_face = torch.gather(all_pix_to_face, dim=-1, index=argzbuf[..., None])[..., 0]  # [1, H, W, 1]
        pix_to_face[no_raster_mask] = 0.0

        all_bary_coords = torch.stack([self.bary_coords[..., :2], rast_out.bary_coords[..., :2]], dim=-1)  # [1, H, W, 2, 2]
        bary_coords_index = argzbuf[..., None].expand(-1, -1, -1, 2, -1)
        bary_coords = torch.gather(all_bary_coords, dim=-1, index=bary_coords_index)[..., 0]  # [1, H, W, 2]

        all_rast_out = torch.cat([bary_coords, zbuf, pix_to_face], dim=-1)[0]
        return RasterizerOutputs(rast_out=all_rast_out)


@dataclass
class MeshRasterizationConfig:
    use_opengl: bool = False
    "Nvdiff use OpenGL or CUDA context"

    anti_aliased: bool = True

    check_errors: bool = True


class MeshRendererMixin:
    def render_mesh(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        render_types: Optional[list] = None,
        anti_aliased: Optional[bool] = None,
        use_filtered: bool = True,
        max_triangles_in_batch: int = -1,
        *args,
        **kwargs,
    ):
        if render_types is None:
            render_types = ["mesh_depth"]
        if anti_aliased is None:
            anti_aliased = self.config.mesh_rast_config.anti_aliased

        mesh: Meshes = getattr(pc, "mesh", None)
        if mesh is None:
            mesh = pc.extract_mesh()
        # filter by camera frustum
        vert_mask = self.is_in_view_frustum(mesh.verts, viewpoint_camera)
        face_mask = vert_mask[mesh.faces].any(dim=-1)
        if use_filtered:
            face_mask = face_mask & pc.face_mask
        mesh_view = Meshes(verts=mesh.verts, faces=mesh.faces[face_mask])

        # Rasterize
        if max_triangles_in_batch < 0:
            rast_out, verts_in_ndc = self.nvdiff_rasterization(viewpoint_camera, mesh_view.verts, mesh_view.faces, self._gl_context)
        else:
            idx_shift = 0
            rast_out: RasterizerOutputs = None
            for st in range(0, mesh_view.faces.shape[0], max_triangles_in_batch):
                ed = min(mesh_view.faces.shape[0], st + max_triangles_in_batch)
                sub_rast_out, verts_in_ndc = self.nvdiff_rasterization(
                    viewpoint_camera, mesh_view.verts, mesh_view.faces[st:ed], self._gl_context
                )
                if rast_out is None:
                    rast_out = sub_rast_out
                else:
                    sub_rast_out.rast_out[..., 3] = torch.where(
                        sub_rast_out.pix_to_face > -1, sub_rast_out.rast_out[..., 3] + idx_shift, sub_rast_out.rast_out[..., 3]
                    )
                    if hasattr(sub_rast_out, "_pix_to_face"):
                        delattr(sub_rast_out, "_pix_to_face")
                    rast_out = rast_out.fuse(sub_rast_out)
                idx_shift += ed - st

            filtered_face_idx, filtered_pix_to_face = rast_out.pix_to_face.unique(return_inverse=True)
            filtered_face_idx = filtered_face_idx[1:]
            filtered_faces = mesh_view.faces[filtered_face_idx]
            filtered_pix_to_face = filtered_pix_to_face - 1

            new_rast_tensor = torch.zeros((1, rast_out.image_height, rast_out.image_width, 4), device=rast_out.zbuf.device)
            new_rast_tensor[..., :2] = rast_out.bary_coords[..., :2]
            new_rast_tensor[..., 2] = rast_out.zbuf
            new_rast_tensor[..., 3] = rast_out.pix_to_face.float() + 1

            rast_out = RasterizerOutputs(rast_out=new_rast_tensor)

        # Parse render types requested
        color_required = "mesh_rgb" in render_types and mesh_view.verts_colors is not None
        depth_required = "mesh_depth" in render_types
        normal_required = "mesh_normal" in render_types

        # Compute per-vertex features to render
        features = torch.zeros((mesh_view.verts.shape[0], 0), device=mesh_view.verts.device)

        if depth_required:
            depth_idx = features.shape[-1]
            verts_h = torch.cat([mesh_view.verts, mesh_view.verts.new_ones((len(mesh_view.verts), 1))], dim=1)
            verts_depths = (verts_h @ viewpoint_camera.world_to_camera)[..., 2].squeeze()
            features = torch.cat([features, verts_depths.unsqueeze(-1)], dim=-1)
        if color_required:
            color_idx = features.shape[-1]
            features = torch.cat([features, mesh_view.verts_colors], dim=-1)

        # Compute image
        feature_img, _ = dr.interpolate(features[None], rast_out.rast_out, mesh_view.faces)  # (1, H, W, n_features)

        # Antialiasing for propagating gradients
        if anti_aliased:
            feature_img = dr.antialias(feature_img, rast_out.rast_out, verts_in_ndc, mesh_view.faces)

        output_pkg = {}
        output_pkg["mesh_rgb"] = feature_img[..., color_idx : color_idx + 3][0].permute(2, 0, 1) if color_required else None
        output_pkg["mesh_depth"] = feature_img[..., depth_idx : depth_idx + 1][0].permute(2, 0, 1) if depth_required else None

        # Compute per-face normals
        output_pkg["mesh_normal"] = None
        if normal_required:
            pix_to_face = rast_out.pix_to_face
            valid_mask = pix_to_face >= 0
            if self.config.mesh_rast_config.check_errors:
                error_mask = pix_to_face >= mesh_view.faces.shape[0]
                error_encourtered = torch.sum(error_mask)
                if error_encourtered > 0:
                    pix_to_face = torch.clamp(pix_to_face, min=0, max=mesh_view.faces.shape[0] - 1)
                    valid_mask = valid_mask & ~error_mask
            output_pkg["mesh_normal"] = (mesh_view.face_normals[pix_to_face].squeeze()[None] * valid_mask)[0].permute(2, 0, 1)

        return output_pkg

    def setup(self, stage: str, *args, **kwargs):
        super().setup(stage, *args, **kwargs)
        if self.config.mesh_rast_config.use_opengl:
            self._gl_context = dr.RasterizeGLContext()
        else:
            self._gl_context = dr.RasterizeCudaContext()

    @staticmethod
    def nvdiff_rasterization(
        camera: Camera, verts: torch.Tensor, faces: torch.Tensor, glctx=None
    ) -> Tuple[RasterizerOutputs, torch.Tensor]:
        device = verts.device
        width, height = camera.width.item(), camera.height.item()

        # Get full projection matrix
        camera_mtx = camera.full_projection

        # Convert to homogeneous coordinates
        pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device)], axis=1)

        # Transform points to NDC/clip space
        pos = torch.matmul(pos, camera_mtx)[None]

        # Rasterize with NVDiffRast
        # TODO: WARNING: pix_to_face is not in the correct range [-1, F-1] but in [0, F],
        # With 0 indicating that no triangle was hit.
        # So we need to subtract 1.
        rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[height, width])
        # bary_coords, zbuf, pix_to_face = rast_out[..., :2], rast_out[..., 2], rast_out[..., 3].int() - 1

        return RasterizerOutputs(rast_out=rast_out), pos

    @staticmethod
    def is_in_view_frustum(points: torch.Tensor, camera: Camera):
        pts_clip = torch.cat([points, points.new_ones((len(points), 1))], dim=1) @ camera.full_projection
        pts_ndc = pts_clip[..., :2] / (pts_clip[..., -1:] + 1e-8)
        valid_mask = (pts_ndc > -1.1).all(dim=-1) & (pts_ndc < 1.1).all(dim=-1) & (pts_clip[..., -1] > 0.01) & (pts_clip[..., -1] < 100.0)
        return valid_mask
