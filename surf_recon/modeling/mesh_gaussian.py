import gc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import trimesh
from einops import rearrange
from tqdm.auto import tqdm

from internal.cameras import Camera, Cameras
from internal.models.mip_splatting import (MipSplatting, MipSplattingModel,
                                           MipSplattingModelMixin)
from internal.models.vanilla_gaussian import \
    OptimizationConfig as VanillaOptimizationConfig
from internal.models.vanilla_gaussian import (VanillaGaussian,
                                              VanillaGaussianModel)
from internal.optimizers import (Adam, OptimizerConfig, SelectiveAdam,
                                 SparseGaussianAdam)
from internal.schedulers import ExponentialDecayScheduler, Scheduler
from internal.utils.general_utils import build_rotation, inverse_sigmoid

from ..utils.general_utils import get_cameras_center_and_diag, init_cdf_mask
from ..utils.mesh import Meshes
from ..utils.sdf import PointIntegration, SDFUtils, TSDFFusion
from ..utils.tetmesh import marching_tetrahedra
from .renderers.importance import rasterize_importance
from .renderers.nvdr import NVDRRendererMixin
from .renderers.radegs import RaDeGSRendererModule
from .selective_adam import SelectiveOccupancyAdam

try:
    from tetranerf.utils.extension import cpp
except ImportError:
    cpp = None
    print("[WARNING] Could not import 'tetranerf.utils.extension.cpp'. Mesh regularization requires this.")


@dataclass
class OptimizationConfig(VanillaOptimizationConfig):
    opacities_lr: float = 0.025

    occupancy_lr: float = 0.025

    optimizer: OptimizerConfig = field(default_factory=lambda: {"class_path": "Adam"})


@dataclass
class MeshConfigMixin:
    max_num_delaunay_gaussians: int = 600_000
    "Max number of Delaunay gaussian samples"
    num_delaunay_points_per_gaussian: int = 9
    "Construct axis-aligned cuboid using 3\sigma bounds, with its 8 corners and center point as Delaunay points."
    delaunay_sampling_method: Literal["random", "surface"] = "surface"
    "Sampling method of Delaunay gaussians"

    sdf_eval_method: Literal["integration", "depth_fusion"] = "depth_fusion"
    sdf_isosurface: float = 0.5
    reset_sdf_n_binary_steps: int = 0
    reset_sdf_n_linearization_steps: int = 20
    sdf_reset_linearization_enforce_std: float = 0.5
    sdf_ema_alpha: float = 0.4

    filter_large_edges: bool = True
    "Filter faces with edge lengths larger than scales"
    collapse_large_edges: bool = False
    "Collapse large edges to vertex with small sdf"


class MeshMixin:
    def extract_mesh(self):
        """
        Extract mesh from gaussian model. Require `delaunay_gaussian_ids`, `base_occupancy`,
        `occupancy_shift`, and `delaunay_tets` to be set. `delaunay_gaussian_ids` is used for
        computing tetrahedra vertices; while others are used for marching tetrahedra.
        """
        # Compute tetra vertices
        voronoi_points, voronoi_scales = self.compute_tetra_vertices()
        # Run marching tetrahedra to extract mesh
        occupancy = self.get_delaunay_occupancy.reshape(-1)  # (N * 9,)
        sdf = SDFUtils.convert_occupancy_to_sdf(occupancy)
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(
            vertices=voronoi_points[None],
            tets=self.get_delaunay_tets,
            sdf=sdf[None],
            scales=voronoi_scales[None],
        )

        end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
        end_scales = scale_list[0]  # (N_verts, 2, 1)

        # Compute mesh vertices differentially
        norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)
        verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]
        faces = faces_list[0]

        # Filtering
        if self.config.filter_large_edges or self.config.collapse_large_edges:
            dmtet_distance = torch.norm(end_points[:, 0, :] - end_points[:, 1, :], dim=-1)
            dmtet_scale = end_scales[:, 0, 0] + end_scales[:, 1, 0]
            dmtet_vertex_mask = dmtet_distance <= dmtet_scale
        if self.config.filter_large_edges:
            faces_mask = dmtet_vertex_mask[faces].all(axis=1)
        if self.config.collapse_large_edges:
            min_end_points = end_points[
                np.arange(end_points.shape[0]),
                end_sdf.argmin(dim=1).flatten().cpu().numpy(),
            ]  # TODO: Do the computation only for filtered vertices
            verts = torch.where(dmtet_vertex_mask[:, None], verts, min_end_points)

        self._mesh = Meshes(verts=verts, faces=faces[faces_mask])
        return self._mesh

    def on_train_batch_end(self, step, module):
        super().on_train_batch_end(step, module)

        schedule = getattr(module.metric.config, "mesh_regularization_schedule", None)
        if step < schedule.start_iter:
            return

        renderer = module.renderer
        train_cameras: Cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras

        # Determine tasks in current iteration before training this batch
        _step = step - schedule.start_iter
        if schedule.reset_delaunay_interval > 0:
            reset_delaunay = _step % schedule.reset_delaunay_interval == 0
        else:
            reset_delaunay = False
        if _step == 0:  # start iter
            reset_delaunay = True

        reset_tetra = _step % schedule.reset_tetrahedralization_interval == 0
        reset_occupancy = _step % schedule.reset_occupancy_interval == 0
        if step > schedule.reset_occupancy_stop_iter:
            reset_occupancy = False
        reset_occupancy_label = _step % schedule.reset_occupancy_label_interval == 0
        if reset_delaunay:
            reset_tetra, reset_occupancy, reset_occupancy_label = True, True, True

        if not module.metric.config.use_occupancy_label_loss:
            reset_occupancy_label = False

        # Reset Delaunay gaussian samples
        if reset_delaunay:
            self.sample_delaunay_gaussians(train_cameras=train_cameras)
        # Compute tetra vertices
        if reset_tetra or reset_occupancy or reset_occupancy_label:
            voronoi_points, voronoi_scales = self.compute_tetra_vertices()
        # Rerun tetrahedralization
        if reset_tetra:
            self.delaunay_tetrahedralization(voronoi_points)
        # Reset occupancies
        if reset_occupancy:
            self.reset_occupancies(
                voronoi_points=voronoi_points,
                voronoi_scales=voronoi_scales,
                renderer=renderer,
                train_cameras=train_cameras,
            )
        # if occupancy is reset, optimizer should be reset too
        if getattr(self, "_update_optimizer_occupancy", False):
            self._set_delaunay_occupancy_optimizer(module.gaussian_optimizers)

        # Reset occupancy labels
        if reset_occupancy_label:
            with torch.no_grad():
                self.extract_mesh()
            self.reset_occupancy_labels(
                voronoi_points=voronoi_points,
                renderer=renderer,
                train_cameras=train_cameras,
            )

    def sample_delaunay_gaussians(self, train_cameras: Cameras):
        delaunay_gaussian_ids = MeshGaussianUtils.sample_delaunay_gaussians(
            self.config.max_num_delaunay_gaussians,
            self,
            cameras=train_cameras,
            sampling_method=self.config.delaunay_sampling_method,
        )
        self.set_delaunay_gaussian_ids(delaunay_gaussian_ids)
        # set base_occupancy and occupancy_shift to 0.0 (0.5 activated).
        self.set_delaunay_occupancy(
            torch.full(
                (
                    self.n_delaunay_gaussians,
                    self.config.num_delaunay_points_per_gaussian,
                ),
                0.5,
            ),
            torch.full(
                (
                    self.n_delaunay_gaussians,
                    self.config.num_delaunay_points_per_gaussian,
                ),
                0.5,
            ),
        )
        # set delaunay_tets to empty
        self.set_delaunay_tets(torch.zeros((0, 4)))

    def compute_tetra_vertices(self, detach_grad: bool = False, **kwargs):
        return MeshGaussianUtils.compute_tetra_vertices(self, detach_grad=detach_grad, **kwargs)

    def delaunay_tetrahedralization(self, tetra_vertices: torch.Tensor):
        delaunay_tets = MeshGaussianUtils.compute_delaunay_tetrahedralization(tetra_vertices=tetra_vertices)
        self.set_delaunay_tets(delaunay_tets)

    @torch.no_grad()
    def reset_occupancies(
        self,
        voronoi_points: torch.Tensor,
        voronoi_scales: torch.Tensor,
        renderer: RaDeGSRendererModule,
        train_cameras: Cameras,
    ):
        initial_sdf = MeshGaussianUtils.compute_voronoi_sdf(
            voronoi_points=voronoi_points,
            voronoi_scales=voronoi_scales,
            gaussian_model=self,
            renderer=renderer,
            train_cameras=train_cameras,
            n_binary_steps=self.config.reset_sdf_n_binary_steps,
            n_linearization_steps=self.config.reset_sdf_n_linearization_steps,
            enforce_std=self.config.sdf_reset_linearization_enforce_std,
            sdf_eval_method=self.config.sdf_eval_method,
            isosurface_value=self.config.sdf_isosurface,
        )
        base_occupancy = SDFUtils.convert_sdf_to_occupancy(initial_sdf)  # (N*9,)
        base_occupancy = base_occupancy.reshape(-1, self.config.num_delaunay_points_per_gaussian)
        new_occupancy = torch.where(
            self.get_base_occupancy != 0.0,
            self.config.sdf_ema_alpha * base_occupancy + (1.0 - self.config.sdf_ema_alpha) * self.get_delaunay_occupancy,
            base_occupancy,
        )
        new_occupancy = new_occupancy.clamp(min=0.005, max=0.995)
        self.set_delaunay_occupancy(base_occupancy=base_occupancy, occupancy=new_occupancy)

    @torch.no_grad()
    def reset_occupancy_labels(
        self,
        voronoi_points: torch.Tensor,
        renderer: NVDRRendererMixin,
        train_cameras: Cameras,
    ):
        def _render(viewpoint: Camera):
            render_pkg = renderer.render_mesh(
                viewpoint_camera=viewpoint,
                mesh=self.mesh,
                render_types=["mesh_depth"],
                anti_aliased=False,
            )
            depth = render_pkg["mesh_depth"]
            return {"rgb": depth.new_zeros((3, *depth.shape[-2:])), "depth": depth}

        occupancy_labels, *_ = (
            TSDFFusion(points=voronoi_points, use_binary_opacity=True).run(cameras=train_cameras, render_fn=_render).get_outputs()
        )
        occupancy_labels = rearrange(
            occupancy_labels.squeeze(),
            "(n k) -> n k",
            k=self.config.num_delaunay_points_per_gaussian,
        )
        self.set_delaunay_occupancy_label(occupancy_labels)

    def setup_extra_properties(self, num_delaunay_gaussians: int = 0, num_tetrahedra: int = 0):
        gaussian_ids = torch.zeros((num_delaunay_gaussians,), dtype=torch.long)
        delaunay_tets = torch.zeros((num_tetrahedra, 4), dtype=torch.long)
        base_occupancy = torch.zeros(
            (num_delaunay_gaussians, self.config.num_delaunay_points_per_gaussian),
            dtype=torch.float32,
        )
        occupancy_shift = torch.zeros_like(base_occupancy, dtype=torch.float32)
        occupancy_label = torch.full_like(base_occupancy, -1, dtype=torch.float32)
        self.tetrahedra = nn.ParameterDict(
            {
                "gaussian_ids": nn.Parameter(gaussian_ids, requires_grad=False),
                "delaunay_tets": nn.Parameter(delaunay_tets, requires_grad=False),
                "base_occupancy": nn.Parameter(base_occupancy, requires_grad=False),
                "occupancy_shift": nn.Parameter(occupancy_shift, requires_grad=True),
                "occupancy_label": nn.Parameter(occupancy_label, requires_grad=False),
            }
        )

    def training_setup(self, module):
        optimizers, schedulers = super().training_setup(module)

        occupancy_params = {
            "params": [self.tetrahedra["occupancy_shift"]],
            "lr": self.config.optimization.occupancy_lr,
            "name": "occupancy",
        }
        optimizer = SelectiveOccupancyAdam().instantiate([occupancy_params], lr=0.0, eps=1e-15)
        optimizers.append(optimizer)
        self._add_optimizer_after_backward_hook_if_available(optimizer, module)

        return optimizers, schedulers

    def before_setup_set_properties_from_number(self, n, property_dict, checkpoint=None, *args, **kwargs):
        super().before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        num_delaunay_gaussians, num_tetrahedra = 0, 0
        if checkpoint is not None:
            num_delaunay_gaussians = len(checkpoint["state_dict"].get("gaussian_model.tetrahedra.gaussian_ids", []))
            num_tetrahedra = len(checkpoint["state_dict"].get("gaussian_model.tetrahedra.delaunay_tets", []))
        self.setup_extra_properties(num_delaunay_gaussians=num_delaunay_gaussians, num_tetrahedra=num_tetrahedra)

    def before_setup_set_properties_from_pcd(self, xyz, rgb, property_dict, *args, **kwargs):
        super().before_setup_set_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        self.setup_extra_properties()

    @property
    def n_delaunay_gaussians(self) -> int:
        return self.tetrahedra["gaussian_ids"].shape[0]

    @property
    def get_delaunay_gaussian_ids(self) -> torch.Tensor:
        return self.tetrahedra["gaussian_ids"]

    def set_delaunay_gaussian_ids(self, gaussian_ids: torch.Tensor):
        self.tetrahedra["gaussian_ids"] = nn.Parameter(gaussian_ids.to(dtype=torch.long, device=self.device), requires_grad=False)

    @property
    def get_delaunay_tets(self) -> torch.Tensor:
        return self.tetrahedra["delaunay_tets"]

    def set_delaunay_tets(self, delaunay_tets: torch.Tensor):
        self.tetrahedra["delaunay_tets"] = nn.Parameter(delaunay_tets.to(dtype=torch.long, device=self.device), requires_grad=False)

    @property
    def get_delaunay_occupancy(self) -> torch.Tensor:
        """Tensor (N, 9)"""
        return self.occupancy_activation(self.tetrahedra["base_occupancy"] + self.tetrahedra["occupancy_shift"])

    @property
    def get_base_occupancy(self) -> torch.Tensor:
        return self.tetrahedra["base_occupancy"]

    @property
    def get_delaunay_occupancy_logit(self) -> torch.Tensor:
        return self.tetrahedra["base_occupancy"] + self.tetrahedra["occupancy_shift"]

    @torch.no_grad()
    def set_delaunay_occupancy(self, base_occupancy: torch.Tensor, occupancy: Optional[torch.Tensor] = None):
        _base_occupancy = self.occupancy_inverse_activation(base_occupancy.to(dtype=torch.float32, device=self.device))
        if occupancy is not None:
            _occupancy_shift = self.occupancy_inverse_activation(occupancy.to(dtype=torch.float32, device=self.device)) - _base_occupancy
        else:
            _occupancy_shift = torch.zeros_like(base_occupancy, dtype=torch.float32, device=self.device)

        if self.tetrahedra["base_occupancy"].shape == _base_occupancy.shape:
            self.tetrahedra["base_occupancy"][...] = _base_occupancy
            self.tetrahedra["occupancy_shift"][...] = _occupancy_shift
        else:
            self._update_optimizer_occupancy = True
            self.tetrahedra["base_occupancy"] = nn.Parameter(_base_occupancy, requires_grad=False)
            self.tetrahedra["occupancy_shift"] = nn.Parameter(_occupancy_shift, requires_grad=True)

    @torch.no_grad()
    def _set_delaunay_occupancy_optimizer(self, optimizers: List[torch.optim.Optimizer]):
        self._update_optimizer_occupancy = False
        for opt in optimizers:
            for group in opt.param_groups:
                if group["name"] != "occupancy":
                    continue

                assert len(group["params"]) == 1

                # get current states
                stored_state = opt.state.get(group["params"][0], None)
                if stored_state is not None:
                    # reset states
                    stored_state["exp_avg"] = torch.zeros_like(self.tetrahedra["occupancy_shift"])
                    stored_state["exp_avg_sq"] = torch.zeros_like(self.tetrahedra["occupancy_shift"])

                    # delete old state key by old params from optimizer
                    del opt.state[group["params"][0]]
                    # set new parameters to optimizer
                    group["params"][0] = self.tetrahedra["occupancy_shift"]
                    opt.state[group["params"][0]] = stored_state

                else:
                    group["params"][0] = self.tetrahedra["occupancy_shift"]

    @property
    def get_delaunay_occupancy_label(self) -> Optional[torch.Tensor]:
        label = self.tetrahedra["occupancy_label"]
        if label[0][0] < 0:
            return None
        return label

    def set_delaunay_occupancy_label(self, delaunay_occupancy_label: torch.Tensor):
        self.tetrahedra["occupancy_label"] = nn.Parameter(
            delaunay_occupancy_label.to(dtype=torch.float32, device=self.device),
            requires_grad=False,
        )

    @property
    def mesh(self) -> Optional[Meshes]:
        return getattr(self, "_mesh", None)

    @property
    def face_mask(self) -> Optional[torch.Tensor]:
        return getattr(self, "_faces_mask", None)

    def occupancy_activation(self, occupancies: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(occupancies)

    def occupancy_inverse_activation(self, occupancies: torch.Tensor) -> torch.Tensor:
        return inverse_sigmoid(occupancies)

    @property
    def device(self) -> torch.device:
        return self.get_xyz.device


@dataclass
class MeshGaussian(MeshConfigMixin, MipSplatting):
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return MeshGaussianModel(self)


class MeshGaussianModel(MeshMixin, MipSplattingModel):
    def setup_from_gaussians(self, gaussian_model: VanillaGaussianModel, *args, **kwargs):
        property_dict = {}
        for key, val in gaussian_model.gaussians.items():
            property_dict[key] = nn.Parameter(val.data.clone().detach(), requires_grad=val.requires_grad)
        if MipSplattingModel._filter_3d_name in self.property_names:
            property_dict[MipSplattingModel._filter_3d_name] = nn.Parameter(
                torch.zeros((gaussian_model.n_gaussians, 1)), requires_grad=False
            )

        self.setup_extra_properties()
        self.set_properties(property_dict)

        self.active_sh_degree = gaussian_model.active_sh_degree
        return self


@dataclass
class MeshVanillaGaussian(MeshConfigMixin, VanillaGaussian):
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs):
        return MeshVanillaGaussianModel(self)


class MeshVanillaGaussianModel(MeshMixin, VanillaGaussianModel):
    pass


class MeshGaussianUtils:
    @classmethod
    @torch.no_grad()
    def post_extract_mesh(
        cls,
        gaussian_model: VanillaGaussianModel,
        renderer,
        cameras: Cameras,
        max_num_delaunay_gaussians: int = 600_000,
        opacity_threshold: float = 0.2,
        trunc_margin: Optional[float] = None,
        sdf_n_binary_steps: int = 8,
        without_color: bool = False,
        device: Optional[torch.device] = None,
        tetra_on_cpu: bool = False,
        skip_filtering: bool = False,
    ) -> Meshes:
        if device is None:
            device = gaussian_model.get_xyz.device
        # Tetrahedralization
        delaunay_gaussian_ids = cls.sample_delaunay_gaussians(
            n_samples=max_num_delaunay_gaussians,
            gaussian_model=gaussian_model,
            cameras=cameras,
        )
        voronoi_points, voronoi_scales = cls.compute_tetra_vertices(
            gaussian_model=gaussian_model,
            delaunay_gaussian_ids=delaunay_gaussian_ids,
            opacity_threshold=opacity_threshold,
        )
        print("Running tetrahedralization with {} points...".format(len(voronoi_points)))

        delaunay_tets = cls.compute_delaunay_tetrahedralization(voronoi_points)
        gc.collect()
        torch.cuda.empty_cache()

        def _render(viewpoint: Camera):
            outputs = renderer.forward(viewpoint, gaussian_model, render_types=["rgb", "depth"])
            return {"rgb": outputs["render"], "depth": outputs["median_depth"]}

        def _voronoi_sdf(points: torch.Tensor):
            voronoi_sdf, *_ = (
                TSDFFusion(points=points, trunc_margin=trunc_margin, use_binary_opacity=False)
                .run(cameras=cameras, render_fn=_render)
                .get_outputs()
            )
            return voronoi_sdf.squeeze()

        voronoi_sdf = _voronoi_sdf(voronoi_points)
        torch.cuda.empty_cache()
        gc.collect()

        # Marching Tetrahedra
        if tetra_on_cpu:
            verts_list, scale_list, faces_list, _ = marching_tetrahedra(
                voronoi_points.cpu()[None],
                delaunay_tets.cpu().long(),
                voronoi_sdf.cpu()[None],
                voronoi_scales.cpu()[None],
            )
        else:
            verts_list, scale_list, faces_list, _ = marching_tetrahedra(
                voronoi_points[None],
                delaunay_tets.to(device).long(),
                voronoi_sdf[None],
                voronoi_scales[None],
            )
        end_points, end_sdf = verts_list[0]
        end_points, end_sdf = end_points.to(device), end_sdf.to(device)
        end_scales = scale_list[0].to(device)
        faces = faces_list[0].to(device)
        print("Extracted mesh with {} vertices and {} faces".format(len(end_points), len(faces)))

        # Refine result of marching tetrahedra with binary search along intersected edges
        left_points, right_points = end_points[:, 0, :], end_points[:, 1, :]
        left_sdf, right_sdf = end_sdf[:, 0, :], end_sdf[:, 1, :]
        left_scale, right_scale = end_scales[:, 0, 0], end_scales[:, 1, 0]
        distance = torch.norm(left_points - right_points, dim=-1)
        scale = left_scale + right_scale
        points = (left_points + right_points) / 2
        for _ in range(sdf_n_binary_steps):
            mid_sdf = _voronoi_sdf(points)
            mid_sdf = mid_sdf.unsqueeze(-1)
            ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

            left_sdf[ind_low] = mid_sdf[ind_low]
            right_sdf[~ind_low] = mid_sdf[~ind_low]
            left_points[ind_low.flatten()] = points[ind_low.flatten()]
            right_points[~ind_low.flatten()] = points[~ind_low.flatten()]
            points = (left_points + right_points) / 2

        camera_extent = get_cameras_center_and_diag(cameras)["diag"]
        if not without_color:
            # Extract vertex colors
            tsdf, vertex_colors, _ = (
                TSDFFusion(points=points, trunc_margin=trunc_margin).run(cameras=cameras, render_fn=_render).get_outputs()
            )
            invisible_pts_mask = tsdf.squeeze() < -1.1
            _trunc_margin = camera_extent * 1.0
            extra_vertex_colors, *_ = (
                TSDFFusion(points=points[invisible_pts_mask], trunc_margin=_trunc_margin)
                .run(cameras=cameras, render_fn=_render)
                .get_outputs()
            )
            vertex_colors[invisible_pts_mask] = extra_vertex_colors
            vertex_colors = (vertex_colors * 255).clip(min=0, max=255).astype(torch.uint8)

        # Build mesh
        mesh = Meshes(
            verts=points,
            faces=faces,
            verts_colors=None if without_color else vertex_colors,
        )

        if not skip_filtering:
            # Filter by scales
            vert_mask = torch.logical_and(distance <= scale, distance <= camera_extent * 0.01)
            mesh = mesh.submesh(vert_mask=vert_mask)
            # Filter by visibility
            is_visible = torch.zeros((len(mesh.verts),), dtype=torch.bool, device=device)
            for idx in tqdm(range(len(cameras)), desc="Filtering mesh by visibility", leave=False):
                camera = cameras[idx].to_device(device)
                mesh_view = renderer.cull_mesh(mesh, camera)
                rast_out, _ = renderer.nvdiff_rasterization(camera, mesh_view.verts, mesh_view.faces)
                pix_to_faces = rast_out.pix_to_face.squeeze()
                valid = pix_to_faces >= 0
                face_indices = pix_to_faces[valid]
                vert_indices = mesh_view.faces[face_indices]  # (N_valid, 3)
                is_visible[vert_indices.flatten()] = True
            mesh = mesh.submesh(vert_mask=is_visible)

        return mesh

    @classmethod
    def sample_delaunay_gaussians(
        cls,
        n_samples,
        gaussian_model: MeshMixin,
        cameras: Cameras,
        sampling_method: Literal["random", "surface"] = "surface",
    ):
        """
        Sample a subset of gaussians as Delaunay tetrahedron vertices.
        Args:
            n_samples: number of Delaunay gaussians to sample
            gaussian_model: the gaussian model
            renderer: the renderer used to compute importance scores
            train_cameras: list of training cameras
            sampling_method: sampling method, "random" or "surface"
        Returns:
            delaunay_gaussian_ids: (n_samples,) long tensor
        """
        n_gaussians = gaussian_model.get_xyz.shape[0]
        if n_gaussians <= n_samples:
            return torch.arange(n_gaussians, device=gaussian_model.get_xyz.device)
        else:
            if sampling_method == "random":
                return torch.randperm(n_gaussians, device=gaussian_model.get_xyz.device)[:n_samples]
            else:
                return cls.sample_surface_gaussians(
                    n_samples=n_samples,
                    gaussian_model=gaussian_model,
                    train_cameras=cameras,
                )

    @staticmethod
    def compute_tetra_vertices(
        gaussian_model: MeshMixin,
        delaunay_gaussian_ids: Optional[torch.Tensor] = None,
        detach_grad: bool = False,
        scale_points_with_downsample_ratio: bool = True,
        scale_points_factor: float = 1.0,
        opacity_threshold: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute tetradron vertices from Delaunay gaussian samples.

        Args:
            gaussian_model: the gaussian model
            detach_grad (bool): whether detach gradient
            scale_points_with_downsample_ratio (bool): whether to adjust scale parameters according to
                downsample ratio if gaussians are downsampled (lower spatial density)
            scale_points_factor (float, optional): manually specified scaling factor
            opacity_threshold (float, optional): whether filter gaussians with low opacity

        Returns:
            vertices (Tensor, (N * 9, 3)): tetrahedron vertices
            vertice_scales (Tensor, (N * 9, 1)): scales of tetrahedron vertices
        """
        if delaunay_gaussian_ids is None:
            delaunay_gaussian_ids = gaussian_model.get_delaunay_gaussian_ids

        def _compute_tetra_vertices():
            xyz = gaussian_model.get_xyz
            if MipSplattingModel._filter_3d_name in gaussian_model.get_property_names():
                opacities, scales = gaussian_model.get_3d_filtered_scales_and_opacities()
            else:
                opacities, scales = (
                    gaussian_model.get_opacity,
                    gaussian_model.get_scaling,
                )
            scales = scales * 3.0
            quats = gaussian_model.get_rotation

            # convert indices to bool mask
            mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
            mask[delaunay_gaussian_ids] = True

            # opacity filtering
            if opacity_threshold > 0.0:
                mask = mask & (opacities.squeeze() >= opacity_threshold)

            downsample_ratio = mask.sum() / xyz.shape[0]
            xyz, scales, quats = xyz[mask], scales[mask], quats[mask]
            if scale_points_with_downsample_ratio:
                scales = scales / (downsample_ratio ** (1 / 3))
            elif scale_points_factor != 1.0:
                scales = scales * scale_points_factor
            rots = build_rotation(quats)  # (N, 3, 3), splat2world

            M = trimesh.creation.box()
            M.vertices *= 2
            vertices = M.vertices.T  # (3, 8)
            vertices = torch.from_numpy(vertices).to(xyz).unsqueeze(0).repeat(xyz.shape[0], 1, 1)  # (N, 3, 8)
            # splat --> world
            vertices = vertices * scales.unsqueeze(-1)
            vertices = torch.bmm(rots, vertices) + xyz.unsqueeze(-1)  # (N, 3, 8)
            vertices = torch.cat([vertices.permute(0, 2, 1), xyz.unsqueeze(1)], dim=1)  # (N, 9, 3)
            vertices = vertices.reshape(-1, 3).contiguous()  # (N * 9, 3)

            max_scale, _ = scales.max(dim=-1, keepdim=True)  # (N, 1)
            vertices_scale = max_scale.repeat(1, 9).reshape(-1, 1)  # (N * 9, 1)
            return vertices, vertices_scale

        if detach_grad:
            with torch.no_grad():
                return _compute_tetra_vertices()
        else:
            return _compute_tetra_vertices()

    @staticmethod
    def compute_delaunay_tetrahedralization(tetra_vertices: torch.Tensor):
        """
        Compute delaunay tetrahedralization from vertices.

        Args:
            voronoi_vertices (Tensor, (N, 3)): Voronoi vertices.

        Returns:
            delaunay_tets (Tensor, (M, 4)): vertex indices of each tetrahedron.
        """
        with torch.no_grad():
            delaunay_tets = cpp.triangulate(tetra_vertices.detach()).to(tetra_vertices.device).long()
        torch.cuda.empty_cache()
        return delaunay_tets

    @classmethod
    def compute_voronoi_sdf(
        cls,
        voronoi_points: torch.Tensor,
        voronoi_scales: torch.Tensor,
        gaussian_model: MeshMixin,
        renderer: RaDeGSRendererModule,
        train_cameras: Cameras,
        n_binary_steps: int = 0,
        n_linearization_steps: int = 20,
        enforce_std: float = 0.5,
        sdf_eval_method: Literal["integration", "depth_fusion"] = "depth_fusion",
        trunc_margin: Optional[float] = None,
        isosurface_value: float = 0.5,
    ):
        if sdf_eval_method == "integration":

            def _integrate(points: torch.Tensor, viewpoint: Camera):
                integrate_pkg = renderer.integrate(
                    points3D=points,
                    viewpoint_camera=viewpoint,
                    pc=gaussian_model,
                )
                return {
                    "alpha": integrate_pkg["alpha_integrated"],
                    "point_coords": integrate_pkg["point_coordinate"],
                    "color": integrate_pkg["color_integrated"],
                    "mask": integrate_pkg["render"][7],
                }

            def voronoi_sdf(points: torch.Tensor):
                sdf, _, _ = (
                    PointIntegration(points=points, isosurface_value=isosurface_value)
                    .run(cameras=train_cameras, integrate_fn=_integrate)
                    .get_outputs()
                )
                return sdf.squeeze()

        elif sdf_eval_method == "depth_fusion":

            def _render(viewpoint: Camera):
                render_pkg = renderer.forward(
                    viewpoint_camera=viewpoint,
                    pc=gaussian_model,
                    render_types=["rgb", "depth"],
                )
                return {
                    "rgb": render_pkg["render"],
                    "depth": render_pkg["median_depth"],
                }

            def voronoi_sdf(points: torch.Tensor):
                sdf, _, _ = (
                    TSDFFusion(
                        points=points,
                        trunc_margin=trunc_margin,
                        use_binary_opacity=False,
                    )
                    .run(cameras=train_cameras, render_fn=_render)
                    .get_outputs()
                )
                return sdf.squeeze()

        else:
            raise RuntimeError(f"Unsupported sdf_eval_method: {sdf_eval_method}")

        initial_sdf = SDFUtils.compute_initial_sdf_with_binary_search(
            voronoi_points=voronoi_points,
            voronoi_scales=voronoi_scales,
            delaunay_tets=gaussian_model.get_delaunay_tets,
            sdf_function=voronoi_sdf,
            n_binary_steps=n_binary_steps,
            n_linearization_steps=n_linearization_steps,
            enforce_std=enforce_std,
        )

        torch.cuda.empty_cache()
        gc.collect()
        return initial_sdf

    @classmethod
    def sample_surface_gaussians(
        cls,
        n_samples: int,
        gaussian_model: MeshMixin,
        train_cameras: Cameras,
    ):
        n_gaussians, device = (
            gaussian_model.get_xyz.shape[0],
            gaussian_model.get_xyz.device,
        )
        imp_score = torch.zeros(n_gaussians, device=device)
        accum_area_max = torch.zeros(n_gaussians, device=device)
        count_rad = torch.zeros(n_gaussians, device=device, dtype=torch.int32)
        count_vis = torch.zeros(n_gaussians, device=device, dtype=torch.int32)

        for idx in tqdm(range(len(train_cameras)), desc="Sampling surface Gaussians", leave=False):
            viewpoint = train_cameras[idx].to_device(device)
            outputs = rasterize_importance(viewpoint, gaussian_model)
            visibility_filter = outputs["visibility_filter"]
            accum_weights = outputs["accum_weights"]
            num_hit_pixels = outputs["num_hit_pixels"]
            num_max_pixels = outputs["num_max_pixels"]

            accum_area_max += num_max_pixels
            mask = num_max_pixels != 0
            _score = imp_score + accum_weights / num_hit_pixels  # gaussian's average blending weight per pixel
            imp_score[mask] = _score[mask]

            non_prune_mask = init_cdf_mask(accum_weights, threshold=0.99)
            count_rad[visibility_filter] += 1
            count_vis[non_prune_mask] += 1

        imp_score[accum_area_max == 0] = 0
        prob = imp_score / imp_score.sum()  # normalize
        n_nonzero_prob = (prob != 0).sum().item()

        n_samples = min(n_samples, n_nonzero_prob)
        indices = torch.from_numpy(np.random.choice(len(prob), size=n_samples, p=prob.detach().cpu().numpy(), replace=False)).to(device)
        non_prune_mask = torch.zeros(n_gaussians, device=device, dtype=torch.bool)
        non_prune_mask[indices] = True

        prune_mask = torch.logical_or(count_vis <= 1, ~non_prune_mask)
        sampled_idx = torch.arange(n_gaussians, device=device)[~prune_mask]
        return sampled_idx
