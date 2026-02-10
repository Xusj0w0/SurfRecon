from typing import List, Optional, Tuple, Union

import torch


class Meshes:
    """
    Meshes class for storing meshes parameters.
    """

    def __init__(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        verts_colors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert verts_colors is None or verts_colors.shape[0] == verts.shape[0]
        self.verts = verts
        self.faces = faces.to(torch.int32)
        self.verts_colors = verts_colors

    @property
    def face_normals(self):
        faces_verts = self.verts[self.faces]
        faces_verts_normals = torch.cross(
            faces_verts[:, 1] - faces_verts[:, 0],
            faces_verts[:, 2] - faces_verts[:, 0],
            dim=-1,
        )
        faces_verts_normals = torch.nn.functional.normalize(faces_verts_normals, dim=-1)
        return faces_verts_normals

    @property
    def vertex_normals(self):
        raise NotImplementedError("Vertex normals are not implemented yet")

    def submesh(
        self,
        vert_idx: Optional[torch.Tensor] = None,
        face_idx: Optional[torch.Tensor] = None,
        vert_mask: Optional[torch.Tensor] = None,
        face_mask: Optional[torch.Tensor] = None,
    ):
        assert (
            (vert_idx is not None) or (vert_mask is not None) or (face_idx is not None) or (face_mask is not None)
        ), "Either vert_idx, vert_mask, face_idx, or face_mask must be provided"

        if (vert_idx is not None) or (vert_mask is not None):
            if vert_mask is None:
                vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
                vert_mask[vert_idx] = True
            face_mask = vert_mask[self.faces].all(dim=1)

        elif (face_idx is not None) or (face_mask is not None):
            if face_mask is None:
                face_mask = torch.zeros(self.faces.shape[0], dtype=torch.bool, device=self.verts.device)
                face_mask[face_idx] = True
            vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
            vert_mask[self.faces[face_mask]] = True

        old_vert_idx_to_new_vert_idx = torch.zeros(self.verts.shape[0], dtype=self.faces.dtype, device=self.verts.device)
        old_vert_idx_to_new_vert_idx[vert_mask] = torch.arange(vert_mask.sum(), dtype=self.faces.dtype, device=self.verts.device)

        new_verts = self.verts[vert_mask]
        new_verts_colors = None if self.verts_colors is None else self.verts_colors[vert_mask]
        new_faces = old_vert_idx_to_new_vert_idx[self.faces][face_mask]

        return Meshes(verts=new_verts, faces=new_faces, verts_colors=new_verts_colors)
