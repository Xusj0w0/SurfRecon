import gc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from internal.cameras import Camera, Cameras

from .tetmesh import marching_tetrahedra


@dataclass
class TSDFFusion:
    """
    Modified from MILo: https://github.com/Anttwo/MILo/blob/master/milo/regularization/sdf/depth_fusion.py
    """

    points: torch.Tensor
    """Tensor (N, 3): Requested points"""

    trunc_margin: Optional[float] = None
    """Truncation margin of SDF value."""

    znear: float = 0.01
    zfar: float = 100.0
    initial_sdf_value: float = -1.0
    use_binary_opacity: bool = False
    interpolate_depth: bool = True
    interpolation_mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear"
    padding_mode: Literal["zeros", "border", "reflection"] = "border"
    align_corners: bool = True

    tsdf: torch.Tensor = field(init=False)
    """Tensor (N, 1): Evaluated TSDF values of requested points"""

    colors: torch.Tensor = field(init=False)
    """Tensor (N, 3): Evaluated colors of requested points"""

    weights: torch.Tensor = field(init=False)

    def __post_init__(self):
        assert self.points.shape[1] == 3, "Points must have shape (N, 3)"
        assert self.znear > 0 and self.zfar > self.znear, "znear must be positive and zfar must be greater than znear"

        if self.use_binary_opacity:
            self.tsdf = torch.ones((self.n_points, 1), device=self.device, dtype=torch.float32)
        else:
            self.tsdf = torch.full((self.n_points, 1), fill_value=self.initial_sdf_value, device=self.device, dtype=torch.float32)
        self.colors = torch.zeros((self.n_points, 3), device=self.device, dtype=torch.float32)
        self.weights = torch.zeros((self.n_points, 1), device=self.device, dtype=torch.float32)

    def run(
        self,
        cameras: Cameras,
        render_fn: Callable,
        obs_weights: Optional[torch.Tensor] = None,
        weight_by_softmax: bool = False,
        softmax_temperature: float = 1.0,
    ):
        """
        Evaluate SDF values of input points.

        Args:
            cameras (Cameras): viewpoint cameras.
            render_fn (Callable): render function that takes in a single camera and outputs a dict:
                "rgb":      (3, H, W) Tensor,
                "depth":    (1, H, W) Tensor.
            interpolate_depth (bool): interpolate depth from rendered map or directly return values of closet pixel
        """
        if self.trunc_margin is None:
            if not self.use_binary_opacity:
                self.trunc_margin = 2e-3 * self.get_cameras_center_and_diag(cameras)["diag"]
            else:
                self.trunc_margin = 1.0

        if obs_weights is None:
            obs_weights = torch.ones((len(cameras),), dtype=torch.float32)

        for idx in tqdm(range(len(cameras)), desc="Depth Fusing", leave=False):
            viewpoint = cameras[idx].to_device(self.device)
            render_pkg = render_fn(viewpoint)
            self.integrate(
                image=render_pkg["rgb"],
                depth=render_pkg["depth"],
                camera=viewpoint,
                obs_weight=float(obs_weights[idx]),
                weight_by_softmax=weight_by_softmax,
                softmax_temperature=softmax_temperature,
            )

        return self

    def get_outputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tsdf (Tensor, (N, 1)): Evaluated TSDF values of requested points
            colors (Tensor, (N, 3)): Evaluated colors of requested points
            weights (Tensor, (N, 1)): Weights during TSDF Fusion
        """
        return self.tsdf, self.colors, self.weights

    def integrate(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        camera: Camera,
        obs_weight: float = 1.0,
        override_points: Optional[torch.Tensor] = None,
        weight_by_softmax: bool = False,
        softmax_temperature: float = 1.0,
    ):
        # Reshape image and depth to (H, W, 3) and (H, W) respectively
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        depth = depth.squeeze()
        H, W = depth.shape

        points = self.points if override_points is None else override_points
        assert points.shape[0] == self.n_points, f"Points must have shape ({self.n_points}, 3)"

        # Transform points to view space
        view_points = self.transform_points_world_to_view(points=points, camera=camera)

        # Project points to pixel space
        pix_points = self.transform_points_to_pixel_space(
            points=view_points,
            camera=camera,
            points_are_already_in_view_space=True,
            keep_float=True,
        )

        int_pix_points = pix_points.round().long()  # (N, 2)
        pix_x, pix_y, pix_z = pix_points[..., 0], pix_points[..., 1], view_points[..., 2]
        int_pix_x, int_pix_y = int_pix_points[..., 0], int_pix_points[..., 1]

        # Remove points outside view frustum and outside depth range
        valid_mask = (
            (pix_x >= 0)
            & (pix_x <= W - 1)
            & (pix_y >= 0)
            & (pix_y <= H - 1)
            # & (pix_z > (camera.znear if self._znear is None else self._znear))
            # & (pix_z < (camera.zfar if self._zfar is None else self._zfar))
            & (pix_z > self.znear)
            & (pix_z < self.zfar)
        )  # (N,)

        if valid_mask.sum() > 0:
            # Get depth and image values at pixel locations
            packed_values = torch.cat(
                [
                    -torch.ones(len(valid_mask), 1, device=self.device),  # Depth values
                    torch.zeros(len(valid_mask), 3, device=self.device),  # Image values
                ],
                dim=-1,
            )  # (N, 4)
            if self.interpolate_depth:
                packed_values[valid_mask] = self.get_interpolated_value_from_pixel_coordinates(
                    value_img=torch.cat([depth.unsqueeze(-1), image], dim=-1),  # (H, W, 4)
                    pix_coords=pix_points[valid_mask],
                    interpolation_mode=self.interpolation_mode,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                )  # (N_valid, 4)
            else:
                packed_values[valid_mask] = torch.cat([depth.unsqueeze(-1), image], dim=-1)[
                    int_pix_y[valid_mask], int_pix_x[valid_mask]
                ]  # (N_valid, 4)
            depth_values = packed_values[..., :1]  # (N, 1)
            img_values = packed_values[..., 1:]  # (N, 3)
            valid_mask = valid_mask & (depth_values[..., 0] > 0.0)  # (N,)

            # Compute distance
            sdf = ((depth_values - pix_z.unsqueeze(-1)) / self.trunc_margin).clamp_max(1.0)  # (N, 1)
            # if not self._use_binary_opacity:
            #     valid_mask = valid_mask & (sdf[..., 0] >= -1.)
            valid_mask = valid_mask & (sdf[..., 0] >= -1.0)

            # Compute observation weight
            _obs_weight = obs_weight
            if weight_by_softmax:
                _obs_weight = _obs_weight * torch.exp(sdf / softmax_temperature)  # (N_valid, 1)

            # Update Field Values
            new_weights = self.weights + _obs_weight  # (N, 1)
            if self.use_binary_opacity:
                new_tsdf = torch.minimum(self.tsdf, (sdf < 0.0).float())
            else:
                new_tsdf = (self.tsdf * self.weights + sdf * _obs_weight) / new_weights  # (N, 1)
            new_colors = (self.colors * self.weights + img_values * _obs_weight) / new_weights  # (N, 3)
            new_colors = new_colors.clamp(min=0.0, max=1.0)  # (N, 3)

            # Update field values
            new_weights = torch.where(valid_mask.unsqueeze(-1), new_weights, self.weights)  # (N, 1)
            new_tsdf = torch.where(valid_mask.unsqueeze(-1), new_tsdf, self.tsdf)  # (N, 1)
            new_colors = torch.where(valid_mask.unsqueeze(-1), new_colors, self.colors)  # (N, 3)
            self.weights = new_weights.detach()
            self.tsdf = new_tsdf.detach()
            self.colors = new_colors.detach()

        else:
            new_weights = self.weights
            new_tsdf = self.tsdf
            new_colors = self.colors

    @staticmethod
    def transform_points_world_to_view(
        points: torch.Tensor,
        camera: Camera,
        use_p3d_convention: bool = False,
    ):
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (N, 4)
        view_points = (points_h @ camera.world_to_camera)[..., :3]  # (N, 3)
        if use_p3d_convention:
            factors = torch.tensor([[-1, -1, 1]], device=points.device)  # (1, 3)
            view_points = factors * view_points  # (N, 3)
        return view_points

    @staticmethod
    def transform_points_to_pixel_space(
        points: torch.Tensor,
        camera: Camera,
        points_are_already_in_view_space: bool = False,
        use_p3d_convention: bool = False,
        znear: float = 1e-6,
        keep_float: bool = False,
    ):
        if points_are_already_in_view_space:
            full_proj_transforms = camera.projection  # (4, 4)
            if use_p3d_convention:
                points = torch.tensor([[-1, -1, 1]], device=points.device) * points
        else:
            full_proj_transforms = camera.full_projection  # (4, 4)

        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (N, 4)
        proj_points = points_h @ full_proj_transforms  # (N, 4)
        proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (N, 2)
        # proj_points is currently in a normalized space where
        # (-1, -1) is the left-top corner of the left-top pixel,
        # and (1, 1) is the right-bottom corner of the right-bottom pixel.

        # For converting to pixel space, we need to scale and shift the normalized coordinates
        # such that (-1/2, -1/2) is the left-top corner of the left-top pixel,
        # and (H-1/2, W-1/2) is the right-bottom corner of the right-bottom pixel.

        height, width = camera.height, camera.width
        image_size = torch.tensor([[width, height]], device=points.device)

        # proj_points = (1. + proj_points) * image_size / 2
        proj_points = (1.0 + proj_points) / 2 * image_size - 1.0 / 2.0

        if keep_float:
            return proj_points
        else:
            return torch.round(proj_points).long()

    @staticmethod
    def get_interpolated_value_from_pixel_coordinates(
        value_img: torch.Tensor,
        pix_coords: torch.Tensor,
        interpolation_mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
        padding_mode: Literal["zeros", "border", "reflection"] = "border",
        align_corners: bool = True,
    ):
        height, width = value_img.shape[:2]
        n_points = pix_coords.shape[0]

        # Scale and shift pixel coordinates to the range [-1, 1]
        factors = 0.5 * torch.tensor([[width - 1, height - 1]], dtype=torch.float32).to(pix_coords.device)  # (1, 2)
        scaled_pix_coords = pix_coords / factors - 1.0  # (N, 2)
        scaled_pix_coords = scaled_pix_coords.view(1, -1, 1, 2)  # (1, N, 1, 2)

        # Interpolate the value
        interpolated_value = F.grid_sample(
            input=value_img.permute(2, 0, 1)[None],  # (1, C, H, W)
            grid=scaled_pix_coords,  # (1, N, 1, 2)
            mode=interpolation_mode,
            padding_mode=padding_mode,  # 'reflection', 'zeros'
            align_corners=align_corners,
        )  # (1, C, N, 1)

        # Reshape to (N, C)
        interpolated_value = interpolated_value.reshape(-1, n_points).permute(1, 0)
        return interpolated_value

    @staticmethod
    def get_cameras_center_and_diag(cameras: Cameras):
        """
        Modified from internal.dataparsers.dataparser.DataParserOutputs.__post_init__
        """
        camera_centers = cameras.camera_center
        average_camera_center = torch.mean(camera_centers, dim=0)
        camera_distance = torch.linalg.norm(camera_centers - average_camera_center, dim=-1)
        max_distance = torch.max(camera_distance)
        return {"center": average_camera_center, "diag": max_distance.item() * 1.1}

    @property
    def device(self) -> torch.device:
        return self.points.device

    @property
    def n_points(self) -> int:
        return self.points.shape[0]


@dataclass
class PointIntegration:
    """
    Modified from MILo: https://github.com/Anttwo/MILo/blob/master/milo/regularization/sdf/integration.py
    """

    points: torch.Tensor
    """Tensor (N, 3): Requested points"""

    isosurface_value: float = 0.5
    min_occupancy_value: float = 1e-10
    transform_sdf_to_linear_space: bool = False
    return_colors: bool = False

    sdf: torch.Tensor = field(init=False)
    """Tensor (N, 1): Evaluated SDF values of requested points"""

    colors: Optional[torch.Tensor] = field(init=False)
    """Tensor (N, 3): Evaluated colors of requested points"""

    weights: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.sdf = torch.ones((self.n_points, 1), device=self.device, dtype=torch.float32)
        self.weights = torch.zeros((self.n_points, 1), device=self.device, dtype=torch.float32)
        if self.return_colors:
            self.colors = torch.zeros((self.n_points, 3), device=self.device, dtype=torch.float32)
        else:
            self.colors = None

    def run(self, cameras: Cameras, integrate_fn: Callable, chunk_size: int = 0):
        for idx in tqdm(range(len(cameras)), desc="Point Integrating", leave=False):
            if idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            viewpoint = cameras[idx].to_device(self.device)
            if chunk_size > 0:
                for start_idx in range(0, self.n_points, chunk_size):
                    end_idx = min(start_idx + chunk_size, self.n_points)
                    chunk_slice = slice(start_idx, end_idx, 1)
                    integrate_pkg = integrate_fn(self.points[chunk_slice], viewpoint)
                    self.integrate(
                        alpha=integrate_pkg["alpha"],
                        point_coords=integrate_pkg["point_coords"],
                        mask=integrate_pkg["mask"],
                        color=integrate_pkg["color"],
                        camera=viewpoint,
                        chunk_slice=chunk_slice,
                    )
            else:
                integrate_pkg = integrate_fn(self.points, viewpoint)
                self.integrate(
                    alpha=integrate_pkg["alpha"],
                    point_coords=integrate_pkg["point_coords"],
                    mask=integrate_pkg["mask"],
                    color=integrate_pkg["color"],
                    camera=viewpoint,
                )

        return self

    def get_outputs(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            tsdf (Tensor, (N, 1)): Evaluated TSDF values of requested points
            colors (Tensor, (N, 3)): Evaluated colors of requested points
            weights (Tensor, (N, 1)): Weights during TSDF Fusion
        """
        isosurface_value = torch.tensor(self.isosurface_value, device=self.sdf.device)
        if self.transform_sdf_to_linear_space:
            sdf = self.sdf.clamp(min=self.min_occupancy_value)
            sdf = torch.sqrt(-2.0 * torch.log(sdf)) - torch.sqrt(-2.0 * torch.log(isosurface_value))
        else:
            sdf = isosurface_value - self.sdf
        sdf = torch.where(self.weights > 0, sdf, -100)
        return sdf, self.colors, self.weights

    @torch.no_grad()
    def integrate(
        self,
        alpha: torch.Tensor,
        point_coords: Optional[torch.Tensor],
        mask: torch.Tensor,
        color: torch.Tensor,
        camera: Camera,
        chunk_slice: Optional[slice] = None,
    ):
        if chunk_slice is None:
            chunk_slice = slice(0, self.n_points, 1)

        if point_coords is not None:
            valid_mask = self.get_valid_point_mask(point_coords, camera, mask)
        else:
            valid_mask = torch.ones_like(alpha, dtype=torch.bool, device=alpha.device)

        valid_mask = valid_mask.unsqueeze(-1)
        if self.return_colors:
            self.colors[chunk_slice] = torch.where(
                valid_mask * (alpha.unsqueeze(-1) < self.sdf[chunk_slice]).reshape(-1, 1), color, self.colors[chunk_slice]
            )
        self.sdf[chunk_slice] = torch.where(valid_mask, torch.min(alpha.unsqueeze(-1), self.sdf[chunk_slice]), self.sdf[chunk_slice])
        self.weights[chunk_slice] = torch.where(valid_mask, self.weights[chunk_slice] + 1, self.weights[chunk_slice])

    @staticmethod
    def get_valid_point_mask(point_coords: torch.Tensor, camera: Camera, mask: torch.Tensor):
        """
        Args:
            point_coords:
            camera:
            mask (Tensor, (H, W)):
        """
        point_coords[..., 0] = (point_coords[..., 0] * 2 + 1) / (camera.width - 1.0) - 1.0
        point_coords[..., 1] = (point_coords[..., 1] * 2 + 1) / (camera.height - 1.0) - 1.0
        valid_point_prob = F.grid_sample(
            mask[None, None, ...].to(torch.float32),
            point_coords[None, None, ...],
            padding_mode="zeros",
            align_corners=False,
        ).squeeze()
        return valid_point_prob > 0.5

    @property
    def device(self) -> torch.device:
        return self.points.device

    @property
    def n_points(self) -> int:
        return self.points.shape[0]


class SDFUtils:
    """
    Modified from MILo: https://github.com/Anttwo/MILo/blob/master/milo/regularization/sdf/learnable.py
    """

    @staticmethod
    def convert_occupancy_to_sdf(occupancy: torch.Tensor) -> torch.Tensor:
        return -(occupancy - 0.5) * 2.0 / 0.99

    @staticmethod
    def convert_sdf_to_occupancy(sdf: torch.Tensor) -> torch.Tensor:
        return -sdf * 0.99 / 2.0 + 0.5

    @classmethod
    @torch.no_grad()
    def compute_initial_sdf_with_binary_search(
        cls,
        voronoi_points: torch.Tensor,
        voronoi_scales: torch.Tensor,
        delaunay_tets: torch.Tensor,
        sdf_function: Callable,
        n_binary_steps: int,
        n_linearization_steps: int,
        enforce_std: float = None,
    ) -> torch.Tensor:
        """
        Compute initial SDF values with binary search and linearization.

        Args:
            voronoi_points (torch.Tensor): The voronoi points. (N_voronoi, 3)
            voronoi_scales (torch.Tensor): The voronoi scales. (N_voronoi, 1)
            delaunay_tets (torch.Tensor): The delaunay tets. (N_tets, 4)
            sdf_function (Callable): The SDF function.
            n_binary_steps (int): The number of binary steps.
            n_linearization_steps (int): The number of linearization steps.

        Returns:
            linearized_sdf (torch.Tensor): The linearized SDF values. (N_voronoi, )
        """

        # Compute initial SDF values
        voronoi_sdf = sdf_function(voronoi_points)  # (N_voronoi, )

        # Refine the initial SDF values with binary search
        if n_binary_steps > 0:
            # Initial Marching Tetrahedra
            verts_list, scale_list, faces_list, interp_v = marching_tetrahedra(
                vertices=voronoi_points[None], tets=delaunay_tets, sdf=voronoi_sdf.reshape(1, -1), scales=voronoi_scales[None]
            )
            end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
            end_scales = scale_list[0]  # (N_verts, 2, 1)
            end_idx = interp_v[0]  # (N_verts, 2)

            refined_verts = cls.refine_intersections_with_binary_search(
                end_points=end_points,
                end_sdf=end_sdf,
                sdf_function=sdf_function,
                n_binary_steps=n_binary_steps,
            )
        # If no binary search, just return the initial SDF values
        else:
            n_linearization_steps = 0
            # norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)
            # refined_verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]
            end_points = None
            end_idx = None
            refined_verts = None

        # Apply iterative linearization to the SDF values
        linearized_sdf = cls.apply_iterative_linearization_to_sdf(
            initial_sdf=voronoi_sdf,
            isosurface_verts=refined_verts,
            end_points=end_points,
            end_idx=end_idx,
            n_steps=n_linearization_steps,
            enforce_std=enforce_std,
        )

        return linearized_sdf

    @staticmethod
    def refine_intersections_with_binary_search(
        end_points: torch.Tensor,
        end_sdf: torch.Tensor,
        sdf_function: Callable,
        n_binary_steps: int,
    ) -> torch.Tensor:
        """
        Refine the intersected isosurface points with binary search.

        Args:
            end_points (torch.Tensor): The end points. (N_verts, 2, 3)
            end_sdf (torch.Tensor): The SDF values at the end points. (N_verts, 2, 1)
            sdf_function (Callable): The SDF function. Takes a tensor of points and returns the SDF values.
            n_binary_steps (int): The number of binary steps.

        Returns:
            refined_points (torch.Tensor): The refined points. (N_verts, 3)
        """

        left_points = end_points[:, 0, :].clone()  # (N_verts, 3)
        right_points = end_points[:, 1, :].clone()  # (N_verts, 3)
        left_sdf = end_sdf[:, 0, :].clone()  # (N_verts, 1)
        right_sdf = end_sdf[:, 1, :].clone()  # (N_verts, 1)
        points = (left_points + right_points) / 2  # (N_verts, 3)

        for step in range(n_binary_steps):
            print("binary search in step {}".format(step))
            mid_points = (left_points + right_points) / 2

            mid_sdf = sdf_function(mid_points)
            mid_sdf = mid_sdf.unsqueeze(-1)
            ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

            left_sdf[ind_low] = mid_sdf[ind_low]
            right_sdf[~ind_low] = mid_sdf[~ind_low]
            left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
            right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]

            points = (left_points + right_points) / 2  # (N_verts, 3)

            torch.cuda.empty_cache()
            gc.collect()

        return points

    @classmethod
    def apply_iterative_linearization_to_sdf(
        cls,
        initial_sdf: torch.Tensor,
        isosurface_verts: torch.Tensor,
        end_points: torch.Tensor,
        end_idx: torch.Tensor,
        n_steps: int = 500,
        enforce_std: float = None,
    ) -> torch.Tensor:
        """
        Starting from initial SDF values, progressively linearize the SDF values
        to make them match a set of isosurface vertices after applying Marching Tetrahedra.

        Args:
            initial_sdf (torch.Tensor): The initial SDF values. (N_voronoi, )
            isosurface_verts (torch.Tensor): The intersected isosurface vertices along the edges. (N_edges, 3)
            end_points (torch.Tensor): The edge endpoints. (N_edges, 2, 3)
            end_idx (torch.Tensor): The indices in [0, N_voronoi) corresponding to the edge endpoints. (N_edges, 2)
            n_steps (int, optional): The number of steps. Defaults to 500.
            enforce_std (float, optional): The standard deviation to enforce. Defaults to None.

        Returns:
            linearized_sdf (torch.Tensor): The linearized SDF values in range [-1, 1]. (N_voronoi, )
        """
        linearized_sdf = initial_sdf.clone()

        for _ in range(n_steps):
            linearized_sdf = cls.linearize_sdf_values(
                sdf_values=linearized_sdf,
                end_points=end_points,
                end_sdf=linearized_sdf[end_idx].unsqueeze(-1),
                end_idx=end_idx,
                verts=isosurface_verts,
                min_shift_length=1e-8,
            )
            # linearized_sdf = linearized_sdf / linearized_sdf.abs().max()
            # linearized_sdf = linearized_sdf / linearized_sdf.std()
            if enforce_std is not None:
                linearized_sdf = (enforce_std * linearized_sdf / (linearized_sdf.std())).clamp(min=-1, max=1)
            else:
                linearized_sdf = linearized_sdf / linearized_sdf.abs().max()

        if n_steps == 0:
            if enforce_std is not None:
                linearized_sdf = (enforce_std * linearized_sdf / (linearized_sdf.std())).clamp(min=-1, max=1)
            else:
                linearized_sdf = linearized_sdf / linearized_sdf.abs().max()

        return linearized_sdf

    @staticmethod
    def linearize_sdf_values(
        sdf_values: torch.Tensor,
        end_points: torch.Tensor,
        end_sdf: torch.Tensor,
        end_idx: torch.Tensor,
        verts: torch.Tensor,
        min_shift_length: float = 1e-8,
    ) -> torch.Tensor:
        """
        Linearize the SDF values.

        Args:
            sdf_values (torch.Tensor): The SDF values. (N_voronoi, )
            end_points (torch.Tensor): The end points. (N_verts, 2, 3)
            end_sdf (torch.Tensor): The SDF values at the end points. (N_verts, 2, 1)
            end_idx (torch.Tensor): The indices of the end points. (N_verts, 2)
            verts (torch.Tensor): The vertices. (N_verts, 3)

        Returns:
            sdf_values (torch.Tensor): The linearized SDF values. (N_voronoi, 9)
        """

        shifts = (end_points[:, 0, :] - end_points[:, 1, :]).norm(dim=-1).clamp(min=min_shift_length)  # (N_verts, )
        factors = (end_sdf[:, 0].abs() + end_sdf[:, 1].abs()).squeeze()  # (N_verts, )

        sdf_0 = end_sdf[:, 0].sign().squeeze() * (verts - end_points[:, 0, :]).norm(dim=-1) / shifts * factors  # (N_verts, )
        sdf_1 = end_sdf[:, 1].sign().squeeze() * (verts - end_points[:, 1, :]).norm(dim=-1) / shifts * factors  # (N_verts, )
        sdfs = torch.cat([sdf_0[:, None], sdf_1[:, None]], dim=-1)  # (N_verts, 2)

        new_sdf_values = torch.zeros_like(sdf_values)
        sdf_counts = torch.zeros_like(sdf_values)

        new_sdf_values.index_add_(dim=0, index=end_idx.flatten().long(), source=sdfs.flatten())
        sdf_counts.index_add_(dim=0, index=end_idx.flatten().long(), source=torch.ones_like(sdfs.flatten()))

        new_sdf_values = torch.where(sdf_counts > 0, new_sdf_values / sdf_counts, sdf_values)

        return new_sdf_values
