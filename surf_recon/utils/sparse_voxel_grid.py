from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import open3d as o3d
import torch
from open3d import core as o3c


def _torch_to_o3c_zero_copy(t: torch.Tensor) -> o3c.Tensor:
    """
    Zero-copy conversion from a torch.Tensor to an open3d.core.Tensor via DLPack.
    The returned Open3D tensor shares the same underlying memory with torch.
    """
    if not t.is_cuda:
        raise ValueError("Expected a CUDA torch.Tensor.")
    return o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(t))


def _o3c_to_torch_zero_copy(t: o3c.Tensor) -> torch.Tensor:
    """
    Zero-copy conversion from an open3d.core.Tensor to a torch.Tensor via DLPack.
    The returned torch tensor shares the same underlying memory with Open3D.
    """
    return torch.utils.dlpack.from_dlpack(t.to_dlpack())


def _cuda_sync():
    """
    Synchronize CUDA work across libraries.
    We try Open3D sync first; fall back to torch if unavailable.
    """
    try:
        # Different Open3D versions expose different spellings
        if hasattr(o3c, "cuda") and hasattr(o3c.cuda, "synchronize"):
            o3c.cuda.synchronize()
        elif hasattr(o3c, "cuda") and hasattr(o3c.cuda, "Synchronize"):
            o3c.cuda.Synchronize()
        else:
            torch.cuda.synchronize()
    except Exception:
        torch.cuda.synchronize()


@dataclass
class SparseVoxelFeatureAccumulator:
    """
    Sparse voxel grid accumulator backed by open3d.core.HashMap.

    - Keys: voxel indices (ix, iy, iz) in int64
    - Values:
        * feature_sum: float32[D]
        * count: int32[1] (optional)
    """

    voxel_size: float
    feat_dim: int
    device: str = "CUDA:0"
    init_capacity: int = 100_000
    store_count: bool = True
    # Optional: pre-reserve to reduce rehash frequency (rehash is expensive and can invalidate views)
    reserve_capacity: Optional[int] = None

    def __post_init__(self):
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive.")
        if self.feat_dim <= 0:
            raise ValueError("feat_dim must be positive.")

        self._dev = o3c.Device(self.device)

        key_dtype = o3c.int64
        key_shape = (3,)

        if self.store_count:
            self._map = o3c.HashMap(
                self.init_capacity,
                key_dtype=key_dtype,
                key_element_shape=key_shape,
                value_dtypes=(o3c.float32, o3c.int32),
                value_element_shapes=((self.feat_dim,), (1,)),
                device=self._dev,
            )
        else:
            self._map = o3c.HashMap(
                self.init_capacity,
                key_dtype=key_dtype,
                key_element_shape=key_shape,
                value_dtype=o3c.float32,
                value_element_shape=(self.feat_dim,),
                device=self._dev,
            )

        if self.reserve_capacity is not None:
            # Reserve helps reduce rehash. Rehash can change internal buffers and indices.
            self._map.reserve(int(self.reserve_capacity))

    @torch.no_grad()
    def update(self, points_xyz: torch.Tensor, features: torch.Tensor):
        """
        Accumulate per-point features into sparse voxels.

        Args:
            points_xyz: (N, 3) float torch CUDA tensor
            features:   (N, D) float torch CUDA tensor
        """
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError(f"points_xyz must have shape (N,3), got {tuple(points_xyz.shape)}")
        if features.ndim != 2 or features.shape[1] != self.feat_dim:
            raise ValueError(f"features must have shape (N,{self.feat_dim}), got {tuple(features.shape)}")
        if not points_xyz.is_cuda or not features.is_cuda:
            raise ValueError("points_xyz and features must be CUDA tensors.")
        if points_xyz.device != features.device:
            raise ValueError("points_xyz and features must be on the same CUDA device.")

        # Ensure contiguous and stable dtypes.
        pts = points_xyz.contiguous().to(dtype=torch.float32)
        feat = features.contiguous().to(dtype=torch.float32)

        vs = float(self.voxel_size)
        vox = torch.floor(pts / vs).to(dtype=torch.int64)  # (N,3)

        # Aggregate duplicates within this batch (torch side).
        unique_vox, inv = torch.unique(vox, dim=0, return_inverse=True)
        m = unique_vox.shape[0]

        feat_sum = torch.zeros((m, self.feat_dim), device=feat.device, dtype=torch.float32)
        feat_sum.index_add_(0, inv, feat)

        if self.store_count:
            ones = torch.ones((pts.shape[0], 1), device=feat.device, dtype=torch.int32)
            cnt_sum = torch.zeros((m, 1), device=feat.device, dtype=torch.int32)
            cnt_sum.index_add_(0, inv, ones)

        # IMPORTANT: sync before handing tensors to Open3D via DLPack.
        # This prevents stream races where Open3D reads tensors before torch finishes writing them.
        _cuda_sync()

        o3_keys = _torch_to_o3c_zero_copy(unique_vox)  # int64 (M,3)
        o3_feat = _torch_to_o3c_zero_copy(feat_sum)  # float32 (M,D)
        if self.store_count:
            o3_cnt = _torch_to_o3c_zero_copy(cnt_sum)  # int32 (M,1)

        # Always fetch fresh buffer views; do NOT cache them across updates.
        # Rehash during insert can invalidate old views and lead to illegal memory access.
        if self.store_count:
            buf_feat = self._map.value_tensor(0)
            buf_cnt = self._map.value_tensor(1)
        else:
            buf_feat = self._map.value_tensor()

        # Find existing keys.
        buf_indices, masks = self._map.find(o3_keys)  # Open3D tensors on CUDA
        masks_t = masks  # bool (M,)
        idx_t = buf_indices[masks_t].to(o3c.int64)  # int64 (K,)

        # Update existing entries.
        if idx_t.shape[0] > 0:
            buf_feat[idx_t] = buf_feat[idx_t] + o3_feat[masks_t]
            if self.store_count:
                buf_cnt[idx_t] = buf_cnt[idx_t] + o3_cnt[masks_t]

        # Sync to ensure updates are visible before potential rehash/insert.
        _cuda_sync()

        # Insert missing entries (keys must be truly missing in the map).
        missing_mask = masks_t.logical_not()
        miss_keys = o3_keys[missing_mask]
        if miss_keys.shape[0] > 0:
            miss_feat = o3_feat[missing_mask]
            if self.store_count:
                miss_cnt = o3_cnt[missing_mask]
                self._map.insert(miss_keys, [miss_feat, miss_cnt])
            else:
                self._map.insert(miss_keys, miss_feat)

        # Sync so that Open3D finishes reading the DLPack-backed torch tensors
        # before this function returns (torch tensors may be freed/reused).
        _cuda_sync()

    @torch.no_grad()
    def finalize(self, return_count: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Export active voxels.

        Returns:
            centers: (M,3) float32 torch CUDA tensor, voxel centers in world coordinates
            feat_sum: (M,D) float32 torch CUDA tensor, accumulated feature sums
            counts: (M,1) int32 torch CUDA tensor if requested and store_count=True, else None
        """
        active = self._map.active_buf_indices().to(o3c.int64)
        if active.shape[0] == 0:
            dev = torch.device("cuda")
            centers = torch.empty((0, 3), device=dev, dtype=torch.float32)
            feat_sum = torch.empty((0, self.feat_dim), device=dev, dtype=torch.float32)
            cnt = torch.empty((0, 1), device=dev, dtype=torch.int32) if (return_count and self.store_count) else None
            return centers, feat_sum, cnt

        # Always fetch fresh views here too.
        keys_buf = self._map.key_tensor()
        if self.store_count:
            feat_buf = self._map.value_tensor(0)
            cnt_buf = self._map.value_tensor(1)
        else:
            feat_buf = self._map.value_tensor()

        keys = keys_buf[active]  # int64 (M,3)
        feat_sum_o3 = feat_buf[active]  # float32 (M,D)

        centers_o3 = keys.to(o3c.float32)
        half = o3c.Tensor([0.5, 0.5, 0.5], dtype=o3c.float32, device=self._dev)
        centers_o3 = (centers_o3 + half) * float(self.voxel_size)

        centers = _o3c_to_torch_zero_copy(centers_o3).contiguous()
        feat_sum = _o3c_to_torch_zero_copy(feat_sum_o3).contiguous()

        cnt = None
        if return_count and self.store_count:
            cnt_o3 = cnt_buf[active]
            cnt = _o3c_to_torch_zero_copy(cnt_o3).contiguous()

        return centers, feat_sum, cnt

    def clear(self):
        """Reset the accumulator by recreating an empty HashMap."""
        self.__post_init__()
