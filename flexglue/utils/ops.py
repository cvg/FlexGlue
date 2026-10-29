import math
from pathlib import Path

import torch
import torch.nn.functional as F


def load_state_dict(weights: str, prefix: str) -> dict:
    """Load a state dict from a local path or URL and keep the entries under prefix."""
    if Path(weights).exists():
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    else:
        state_dict = torch.hub.load_state_dict_from_url(weights, map_location="cpu")
    state_dict = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    if not state_dict:
        raise KeyError(f"No weights with prefix {prefix!r} in {weights}.")
    return state_dict


def normalize_keypoints(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Normalize points (b, ..., 2) of a (w, h) image with size (b, 2), to [-1, 1]."""
    size = size.view(size.shape[0], *[1] * (kpts.dim() - 2), 2)
    return kpts / size * 2 - 1


def denormalize_keypoints(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Denormalize points (b, ..., 2) from [-1, 1] to pixels of a (w, h) image."""
    size = size.view(size.shape[0], *[1] * (kpts.dim() - 2), 2)
    return (kpts + 1) / 2 * size


def cell_centers(h: int, w: int, scale: tuple, device=None) -> torch.Tensor:
    """Normalized (x, y) centers (h*w, 2) of an h x w grid, divided by (sx, sy)."""
    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    xy = torch.stack([(x + 0.5) / scale[0], (y + 0.5) / scale[1]], -1)
    return xy.flatten(0, 1) * 2 - 1


def to_hwc(x: torch.Tensor) -> torch.Tensor:
    """(b, c, h, w) -> (b, h, w, c)."""
    return x.permute(0, 2, 3, 1)


def to_chw(x: torch.Tensor) -> torch.Tensor:
    """(b, h, w, c) -> (b, c, h, w)."""
    return x.permute(0, 3, 1, 2)


def sample_features(features: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample (B, C, H, W) features at points (B, N, 2) in [-1, 1]."""
    out = F.grid_sample(
        features, pts[:, None], "bilinear", padding_mode="border", align_corners=False
    )
    return out[:, :, 0].transpose(1, 2)


def sorted_entmax15_threshold(ys: torch.Tensor) -> torch.Tensor:
    """Exact 1.5-entmax threshold from scores sorted in descending order."""
    rho = torch.arange(1, ys.shape[-1] + 1, device=ys.device, dtype=ys.dtype)
    mean = ys.cumsum(-1) / rho
    mean_sq = ys.square().cumsum(-1) / rho
    tau = mean - ((1 - rho * (mean_sq - mean.square())) / rho).clamp_min(0).sqrt()
    support = (tau <= ys).sum(-1, keepdim=True)
    return tau.gather(-1, support - 1)


def entmax15_threshold(
    y: torch.Tensor, k: int = 512, exact: bool = False
) -> torch.Tensor:
    """Tau of entmax15 = (y - tau)_+^2 from the top k, exact if the support is <= k."""
    if y.shape[-1] <= k:
        return sorted_entmax15_threshold(y.sort(-1, descending=True).values)
    yk = y.topk(k + exact, dim=-1).values
    tau = sorted_entmax15_threshold(yk[..., :k])
    if exact:  # full sort where the support extends past the top k (host sync)
        redo = (yk[..., k:] > tau).squeeze(-1)
        if redo.any():
            tau[redo] = sorted_entmax15_threshold(
                y[redo].sort(-1, descending=True).values
            )
    return tau


def entmax15_window_readout(logits, grid, hw, radius: int = 1) -> tuple:
    """1.5-entmax expected position (B, N, 2) and mass (B, N) around the argmax."""
    vmax, idx = logits.max(-1, keepdim=True)
    y = (logits - vmax) / 2
    tau = entmax15_threshold(y)
    h, w = hw
    r = torch.arange(-radius, radius + 1, device=y.device)
    ny = (idx // w)[..., None] + r[:, None]
    nx = (idx % w)[..., None] + r[None, :]
    inside = ((ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)).flatten(-2)
    nidx = (ny.clamp(0, h - 1) * w + nx.clamp(0, w - 1)).flatten(-2)
    p = (y.gather(-1, nidx) - tau).clamp_min(0).square()  # entmax sums to 1
    position = (p[..., None] * grid[nidx]).sum(-2) / p.sum(-1, keepdim=True)
    return position, (p * inside).sum(-1)  # clamped border cells count once


def sq_dist_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared dist (B, M, N) of x (B, M, D) and y (B, N, D) in one GEMM, no sqrt."""
    ones_x, ones_y = x.new_ones(*x.shape[:-1], 1), y.new_ones(*y.shape[:-1], 1)
    xa = torch.cat([-2 * x, x.square().sum(-1, keepdim=True), ones_x], -1)
    ya = torch.cat([y, ones_y, y.square().sum(-1, keepdim=True)], -1)
    return xa @ ya.mT


def mutual_nn(dist: torch.Tensor):
    """Mutual nearest neighbors of a (B, M, N) distance matrix, -1 if not mutual."""
    m0, m1 = dist.argmin(2), dist.argmin(1)
    mutual0 = torch.arange(m0.shape[1], device=m0.device) == m1.gather(1, m0)
    mutual1 = torch.arange(m1.shape[1], device=m1.device) == m0.gather(1, m1)
    return torch.where(mutual0, m0, -1), torch.where(mutual1, m1, -1)


def inside_border(
    pts: torch.Tensor, size: torch.Tensor, border: torch.Tensor
) -> torch.Tensor:
    """Points (B, ..., 2) in pixels at least border (B,) pixels inside (w, h) images."""
    size = size.view(size.shape[0], *[1] * (pts.dim() - 2), 2)
    border = border.view(size.shape[0], *[1] * (pts.dim() - 1))
    return ((pts > border) & (pts < size - border)).all(-1)  # image edges at 0 and size


def grid_sources(grid: torch.Tensor) -> torch.Tensor:
    """Normalized source position (B, H, W, 2) of every cell of a grid (B, H, W, 2)."""
    h, w = grid.shape[1:3]
    centers = cell_centers(h, w, scale=(w, h), device=grid.device)
    return centers.view(1, h, w, 2).expand_as(grid)


def sample_cells(weights: torch.Tensor, n: int) -> tuple:
    """Sample at most n unique cells of (B, N) weights: indices (B, n), (B, n, 2)."""
    n = min(n, weights.shape[1])
    idx = torch.multinomial(weights.clamp_min(1e-12), n, replacement=False)
    return idx, idx[..., None].expand(-1, -1, 2)


def lattice_kde(
    points: torch.Tensor, weights: torch.Tensor, std: float
) -> torch.Tensor:
    """Weighted gaussian kde (B, N) of points (B, N, D) in [-1, 1] at themselves."""
    b, n, d = points.shape
    pitch = std / 2  # splat to a lattice, blur it, read it back: cost independent of N
    size = math.ceil(2 / pitch) + 2
    u = ((points.float() + 1) / pitch).clamp(0, size - 1 - 1e-4)
    base, frac = u.floor().long(), u.frac()
    corners = torch.cartesian_prod(*[torch.arange(2, device=points.device)] * d)
    strides = size ** torch.arange(d - 1, -1, -1, device=points.device)
    idx = ((base[:, :, None] + corners) * strides).sum(-1).flatten(1)  # (B, N * 2^d)
    tent = torch.where(corners.bool(), frac[:, :, None], 1 - frac[:, :, None]).prod(-1)
    lattice = u.new_zeros(b, size**d)
    lattice.scatter_add_(1, idx, (tent * weights[..., None].float()).flatten(1))
    s = math.sqrt(std**2 - pitch**2 / 3)  # splat and read tents add pitch^2 / 6 each
    r = math.ceil(3 * s / pitch)
    k = torch.arange(-r, r + 1, device=points.device) * pitch
    kernel = (std / s * torch.exp(-(k**2) / (2 * s**2))).view(1, 1, -1)
    lattice = lattice.view(b, *[size] * d)
    for dim in range(1, d + 1):
        x = lattice.movedim(dim, -1)
        x = F.conv1d(x.reshape(-1, 1, size), kernel, padding=r).view(x.shape)
        lattice = x.movedim(-1, dim)
    lattice = lattice.reshape(b, -1)
    return (lattice.gather(1, idx).view_as(tent) * tent).sum(-1)


def kde_weights(
    dense_p2d0_n1: torch.Tensor,
    dense_p2d1_n0: torch.Tensor,
    scores: torch.Tensor,
    num_samples: int,
    std: float = 0.1,
    min_density: float = 5e-4,
    expansion: int = 4,
) -> torch.Tensor:
    """RoMa v2's balanced sampling weights (B, N) of the valid (score > 0) cells."""
    valid = (scores > 0).float()
    pool = valid.sum(1).clamp(max=expansion * num_samples)
    scale = pool / valid.sum(1).clamp_min(1)  # density in samples of RoMa's pool
    min_support = (min_density * pool)[:, None]  # RoMa: 10 of a 4 x 5000 pool
    points = []
    for i, grid in enumerate((dense_p2d0_n1, dense_p2d1_n0)):
        pair = [grid_sources(grid).flatten(1, 2), grid.flatten(1, 2)]
        points.append(torch.cat([pair[i], pair[1 - i]], -1))  # (x0, x1) normalized
    density = (
        lattice_kde(torch.cat(points, 1), valid, std) * scale[:, None]
    )  # unweighted
    balance = torch.where(density < min_support, 1e-7, 1 / (density + 1))
    return balance * valid  # valid cells only, no extra confidence weighting


def identity_matches(
    kpts0: torch.Tensor, kpts1: torch.Tensor, scores: torch.Tensor, threshold: float
) -> dict:
    """Row k of kpts0 matches row k of kpts1 if its score is above threshold."""
    valid = scores > threshold
    matches = torch.where(valid, torch.arange(valid.shape[1], device=valid.device), -1)
    scores = torch.where(valid, scores, 0)
    return {
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "matches0": matches,
        "matches1": matches,
        "matching_scores0": scores,
        "matching_scores1": scores,
    }


def pack_matches(pred: dict) -> dict:
    """Copy of pred with LightGlue-style lists matches (S, 2) and scores (S,) (sync)."""
    matches0, scores0 = pred["matches0"], pred["matching_scores0"]
    valid = matches0 > -1
    matches = [
        torch.stack([torch.where(v)[0], m[v]], -1) for v, m in zip(valid, matches0)
    ]
    scores = [s[v] for s, v in zip(scores0, valid)]
    return {**pred, "matches": matches, "scores": scores}
