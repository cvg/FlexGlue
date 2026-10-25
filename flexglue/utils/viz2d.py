"""Plotting adapted from LightGlue's viz2d (Apache-2.0), with points in corner coordinates."""

from typing import List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

Points = Union[np.ndarray, torch.Tensor]


def to_numpy(x: Points) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def cm_RdGn(x: np.ndarray) -> np.ndarray:
    """Colormap red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def cm_grad2d(xy: np.ndarray) -> np.ndarray:
    """2D colormap yellow (0, 0) -> green (1, 0) -> red (0, 1) -> blue (1, 1)."""
    tl, tr = np.array([1.0, 0, 0]), np.array([0, 0.0, 1])
    ll, lr = np.array([1.0, 1.0, 0]), np.array([0, 1.0, 0])
    xy = np.clip(xy, 0, 1)
    x, y = xy[..., :1], xy[..., -1:]
    rgb = (1 - x) * (1 - y) * ll + x * (1 - y) * lr + x * y * tr + (1 - x) * y * tl
    return rgb.clip(0, 1)


def plot_images(
    imgs: Sequence[Points],
    titles: Optional[Sequence[str]] = None,
    cmaps: Union[str, Sequence[str]] = "gray",
    dpi: int = 100,
    pad: float = 0.5,
    adaptive: bool = True,
) -> None:
    """Images side by side: numpy (H, W, 3), torch (3, H, W) or mono (H, W)."""
    imgs = [
        (
            img.permute(1, 2, 0).cpu().numpy()
            if isinstance(img, torch.Tensor) and img.dim() == 3
            else to_numpy(img)
        )
        for img in imgs
    ]
    n = len(imgs)
    if isinstance(cmaps, str):
        cmaps = [cmaps] * n
    ratios = [i.shape[1] / i.shape[0] for i in imgs] if adaptive else [4 / 3] * n
    fig, ax = plt.subplots(
        1,
        n,
        figsize=[sum(ratios) * 4.5, 4.5],
        dpi=dpi,
        gridspec_kw={"width_ratios": ratios},
    )
    if n == 1:
        ax = [ax]
    for i in range(n):
        h, w = imgs[i].shape[:2]
        # corner coordinates: the image spans [0, w] x [0, h], pixel centers at +0.5
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]), extent=(0, w, h, 0))
        ax[i].set_axis_off()
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(
    kpts: Sequence[Points],
    colors: Union[str, List] = "lime",
    ps: float = 4,
    axes: Optional[Sequence[plt.Axes]] = None,
    a: Union[float, List[float]] = 1.0,
) -> None:
    """Keypoints (N, 2) in corner coordinates, one array per plotted image."""
    colors = colors if isinstance(colors, list) else [colors] * len(kpts)
    a = a if isinstance(a, list) else [a] * len(kpts)
    axes = plt.gcf().axes if axes is None else axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        k = to_numpy(k)
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


def plot_matches(
    kpts0: Points,
    kpts1: Points,
    color=None,
    lw: float = 1.5,
    ps: float = 4,
    a: float = 1.0,
    axes: Optional[Tuple[plt.Axes, plt.Axes]] = None,
) -> None:
    """Matches (N, 2) in corner coordinates between the two plotted images, gradient colors if no color."""
    fig = plt.gcf()
    ax0, ax1 = fig.axes[:2] if axes is None else axes
    kpts0, kpts1 = to_numpy(kpts0), to_numpy(kpts1)
    if len(kpts0) != len(kpts1):
        raise ValueError(f"Got {len(kpts0)} and {len(kpts1)} points.")
    if color is None:
        span = np.ptp(kpts0, axis=0, keepdims=True).clip(min=1e-6)
        color = cm_grad2d((kpts0 - kpts0.min(axis=0, keepdims=True)) / span)
    elif len(color) > 0 and not isinstance(color[0], (tuple, list, np.ndarray)):
        color = [color] * len(kpts0)
    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)
    ax0.autoscale(enable=False)  # freeze the axes so the lines keep their transform
    ax1.autoscale(enable=False)
    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(
    idx: int,
    text: str,
    pos: Tuple[float, float] = (0.01, 0.99),
    fs: float = 15,
    color: str = "w",
    lcolor: Optional[str] = "k",
    lwidth: float = 2,
    ha: str = "left",
    va: str = "top",
) -> None:
    """Text on the idx-th plotted image, in axes coordinates, outlined in lcolor."""
    ax = plt.gcf().axes[idx]
    t = ax.text(
        *pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )


def save_plot(path: str, **kw) -> None:
    """Save the current figure without white margins."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)


def warp_image(
    image: torch.Tensor,
    p2d_n: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
    size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Image (C, H, W) sampled at a grid (h, w, 2) in [-1, 1], faded by confidence."""
    if p2d_n.dim() == 4:  # batched grid of one pair
        if p2d_n.shape[0] != 1:
            raise ValueError(f"Expected a single grid, got {tuple(p2d_n.shape)}.")
        p2d_n = p2d_n[0]
        confidence = None if confidence is None else confidence[0]
    grid = p2d_n.permute(2, 0, 1)[None]  # 1 x 2 x h x w
    if size is not None:  # pixels of the target image, interpolated between grid cells
        grid = F.interpolate(grid, size, mode="bilinear", align_corners=False)
    warped = F.grid_sample(
        image[None].to(grid), grid.permute(0, 2, 3, 1), align_corners=False
    )
    if confidence is not None:  # fade to white where the warp is not confident
        alpha = confidence[None, None].to(grid).clamp(0, 1)
        alpha = F.interpolate(alpha, warped.shape[-2:], mode="bilinear")
        warped = alpha * warped + (1 - alpha)
    return warped[0]
