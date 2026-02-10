import torch


class TopKTracker:
    def __init__(
        self,
        n_samples: int,
        k: int,
        device: torch.device = torch.device("cpu"),
        is_reverse: bool = False,
    ):
        self.n_samples = n_samples
        self.k = k
        self.device = device
        self.is_reverse = is_reverse
        self._top_k_values = torch.full((n_samples, k), -torch.inf, device=device)
        # self._kth_indices = torch.zeros((n_samples, 1), dtype=torch.long, device=device)

    def update(self, values: torch.Tensor):
        """
        scores: (n_samples,)
        """
        assert values.shape[0] == self.n_samples

        if self.is_reverse:
            values = -values
        kth_values, indices = torch.min(self._top_k_values, dim=-1)
        update_mask = values.to(kth_values) > kth_values  # (n_samples,)

        if update_mask.any():
            rows = update_mask.nonzero(as_tuple=True)[0]
            cols = indices[update_mask]
            self._top_k_values[rows, cols] = values[rows]

    @property
    def top_k_values(self) -> torch.Tensor:
        if self.is_reverse:
            return -self._top_k_values
        return self._top_k_values

    @property
    def means(self) -> torch.Tensor:
        mask = torch.isneginf(self._top_k_values)
        valid_values = self._top_k_values.clone()
        valid_values[mask] = 0.0
        avg = valid_values.sum(dim=-1) / (~mask).sum(dim=-1).clamp(min=1)
        if self.is_reverse:
            return -avg
        return avg


def get_cameras_center_and_diag(cameras):
    """
    Modified from internal.dataparsers.dataparser.DataParserOutputs.__post_init__
    """
    camera_centers = cameras.camera_center
    average_camera_center = torch.mean(camera_centers, dim=0)
    camera_distance = torch.linalg.norm(camera_centers - average_camera_center, dim=-1)
    max_distance = torch.max(camera_distance)
    return {"center": average_camera_center, "diag": max_distance.item() * 1.1}


def init_cdf_mask(importance: torch.Tensor, threshold: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Keep a minimal set of elements whose cumulative importance mass reaches `threshold`
    of the total mass (CDF from the top, i.e., largest-first).

    Args:
        importance: any shape, real-valued tensor
        threshold: in [0, 1]. 1.0 -> keep all. 0.0 -> keep none (unless total=0)
        eps: small value to avoid divide-by-zero

    Returns:
        mask: bool tensor of shape importance.numel() (flattened)
    """
    imp = importance.flatten()

    # Edge cases
    if threshold >= 1.0:
        return torch.ones_like(imp, dtype=torch.bool)
    if threshold <= 0.0:
        return torch.zeros_like(imp, dtype=torch.bool)

    # Sanitize: treat non-finite as 0 (or you can choose to drop them)
    imp = torch.where(torch.isfinite(imp), imp, torch.zeros_like(imp))

    # If importance can be negative, "mass" is ambiguous.
    # Common choice: clamp to >=0 so "total mass" is meaningful.
    imp_mass = imp.clamp_min(0)

    total = imp_mass.sum()
    if total <= eps:
        # All zeros -> nothing is important; choose keep none (or keep all, depending on your policy)
        return torch.zeros_like(imp, dtype=torch.bool)

    target = total * threshold

    # Sort descending by mass
    vals, idx = torch.sort(imp_mass, descending=True)
    csum = torch.cumsum(vals, dim=0)

    # Find smallest k with csum[k-1] >= target
    k = torch.searchsorted(csum, target, right=False).item() + 1  # +1 because index->count

    # Build mask: select those top-k indices
    mask = torch.zeros_like(imp, dtype=torch.bool)
    mask[idx[:k]] = True
    return mask
