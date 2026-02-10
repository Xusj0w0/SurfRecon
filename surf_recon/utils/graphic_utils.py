import torch
import torch.nn.functional as F

from internal.cameras import Camera


def fix_normal_map(view: Camera, normal: torch.Tensor, normal_in_view_space=True):
    W, H = view.width.item(), view.height.item()
    intrins_inv = torch.linalg.inv(view.get_K()[:3, :3])
    grid_x, grid_y = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).to(normal)
    rays_d = (intrins_inv @ points).reshape(3, H, W)

    if normal_in_view_space:
        normal_view = normal
    else:
        normal_view = normal.clone()
        if normal.shape[0] == 3:
            normal_view = normal_view.permute(1, 2, 0)
        normal_view = normal_view @ view.world_to_camera[:3, :3]
        if normal.shape[0] == 3:
            normal_view = normal_view.permute(2, 0, 1)

    if normal_view.shape[0] != 3:
        rays_d = rays_d.permute(1, 2, 0)
        dim_to_sum = -1
    else:
        dim_to_sum = 0

    return torch.sign((-rays_d * normal_view).sum(dim=dim_to_sum, keepdim=True)) * normal_view


def depth_to_normal(depth: torch.Tensor, camera: Camera):
    """
    Convert depth map to normal map in camera coordinate.

    Args:
        depth (Tensor (1, H, W)): depth map
        camera: Camera

    Returns:
        normal (Tensor, (3, H, W)): normal map
    """
    pointmap = depth_to_pointmap(depth=depth, camera=camera)
    normal = pointmap_to_normal(pointmap=pointmap)
    return normal


def depth_to_pointmap(depth: torch.Tensor, camera: Camera):
    """
    Convert depth map to point map in camera coordinate.

    Args:
        depth (Tensor (1, H, W)): depth map
        camera: Camera

    Returns:
        pointmap (Tensor, (3, H, W)): point map
    """
    H, W = depth.shape[-2:]
    assert camera.height.item() == H and camera.width.item() == W

    intrins_inv = torch.tensor(
        [
            [1 / camera.fx, 0.0, -camera.width / (2 * camera.fx)],
            [0.0, 1 / camera.fy, -camera.height / (2 * camera.fy)],
            [0.0, 0.0, 1.0],
        ]
    ).to(depth)
    grid_x, grid_y = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).to(intrins_inv).reshape(3, -1)
    rays_d = intrins_inv @ points
    pointmap = depth.reshape(1, -1) * rays_d
    return pointmap.reshape(3, H, W)


def pointmap_to_normal(pointmap: torch.Tensor):
    """
    Convert point map to normal map in camera coordinate.

    Args:
        pointmap (Tensor (3, H, W)): point map
        camera: Camera

    Returns:
        normal (Tensor, (3, H, W)): normal
    """
    normal = pointmap.new_zeros(pointmap.shape)
    dx = pointmap[..., 2:, 1:-1] - pointmap[..., :-2, 1:-1]
    dy = pointmap[..., 1:-1, 2:] - pointmap[..., 1:-1, :-2]
    normal[..., 1:-1, 1:-1] = F.normalize(torch.cross(dx, dy, dim=0), dim=0)
    return normal
