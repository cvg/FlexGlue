"""Dense refiners of RoMa v2, run on a FlexGlue coarse warp.

Adapted from the official implementation and weights:
    RoMa v2: Harder Better Faster Denser Feature Matching
    Johan Edstedt, David Nordström, Yushan Zhang, Georg Bökman, Jonathan Astermark,
    Viktor Larsson, Anders Heyden, Fredrik Kahl, Mårten Wadenbäck, Michael Felsberg
    ECCV 2026, https://arxiv.org/abs/2511.15706
    Code: https://github.com/Parskatt/RoMaV2 (MIT License)
    Copyright (c) 2025 Johan Edstedt
"""

from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utils import ops
from .utils import preprocess as pp


def bhwc_interpolate(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    x = F.interpolate(ops.to_chw(x), size, mode="bilinear", align_corners=False)
    return ops.to_hwc(x)


def local_correlation(
    f0: torch.Tensor, f1: torch.Tensor, warp: torch.Tensor, radius: int
) -> torch.Tensor:
    """Corr. f0 (b, c, h, w) with f1 in a (2r+1)^2 window around warp (b, h, w, 2)."""
    b, c, h, w = f0.shape
    r = torch.linspace(-2 * radius, 2 * radius, 2 * radius + 1, device=f0.device)
    dy, dx = torch.meshgrid(r / h, r / w, indexing="ij")
    window = torch.stack([dx, dy], -1).reshape(-1, 2)
    coords = (warp[..., None, :] + window).reshape(b, h, -1, 2)
    f1w = F.grid_sample(f1, coords, align_corners=False).reshape(b, c, h, w, -1)
    return ops.to_chw((f0[..., None] / c**0.5 * f1w).sum(1))


class VGG19BN(nn.Module):
    """First three blocks of VGG19-BN, returns BHWC features at strides 1, 2 and 4."""

    def __init__(self) -> None:
        super().__init__()
        layers, c_in = [], 3
        for c_out, n in ((64, 2), (128, 2), (256, 4)):
            for _ in range(n):
                layers += [
                    nn.Conv2d(c_in, c_out, 3, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                ]
                c_in = c_out
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.ModuleList(layers)
        mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor(
            [0.229, 0.224, 0.225]
        )
        self.register_buffer("mean", mean[:, None, None], persistent=False)
        self.register_buffer("std", std[:, None, None], persistent=False)

    def forward(
        self, image: torch.Tensor, scales: Optional[Tuple[int, ...]] = None
    ) -> Dict[int, torch.Tensor]:
        # None preserves the full operation, including hooks and BN updates.
        requested = (1, 2, 4) if scales is None else tuple(scales)
        if not requested or any(s not in (1, 2, 4) for s in requested):
            raise ValueError("scales must be a nonempty subset of (1, 2, 4)")
        last = (
            4
            if scales is None or any(m.training for m in self.modules())
            else max(requested)
        )
        x, feats, stride = (image - self.mean) / self.std, {}, 1
        with torch.autocast(image.device.type, torch.bfloat16):
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    if stride in requested:
                        feats[stride] = ops.to_hwc(x)
                    if stride == last:
                        break
                    stride *= 2
                x = layer(x)
        return feats


class Block(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv_depthwise = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv_pointwise = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, torch.bfloat16):
            return self.conv_pointwise(self.relu(self.norm(self.conv_depthwise(x))))


class ConvRefiner(nn.Module):
    """One RoMa refinement stage: residual warp and logit update at a fixed stride."""

    def __init__(
        self,
        feat_dim: int,
        proj_dim: int,
        disp_dim: int,
        radius: Optional[int],
        hidden_blocks: int = 8,
        correlation=local_correlation,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.correlation = correlation
        dim = 2 * proj_dim + disp_dim + (0 if radius is None else (2 * radius + 1) ** 2)
        self.proj = nn.Linear(feat_dim, proj_dim)
        self.block1 = Block(dim)
        self.hidden_blocks = nn.Sequential(*[Block(dim) for _ in range(hidden_blocks)])
        self.disp_emb = nn.Conv2d(2, disp_dim, 1)
        self.warp_head = nn.Conv2d(dim, 2, 1)
        self.confidence_head = nn.Conv2d(dim, 4, 1)

    def forward(
        self,
        f_i: torch.Tensor,
        f_j: torch.Tensor,
        warp: torch.Tensor,
        mlogit_i: torch.Tensor,
        scale_factor: Tuple[float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = f_i.shape[1:3]
        if f_j.shape[1:3] != (h, w):
            raise ValueError(
                "RoMa refinement requires matching feature sizes. "
                "Use a fixed (height, width) refine_resize for image extraction, "
                "or rebuild supplied fine_features on matching grids."
            )
        f_i, f_j = self.proj(f_i.float()), self.proj(f_j.float())
        f_ji = F.grid_sample(ops.to_chw(f_j), warp, align_corners=False)
        grid = ops.cell_centers(h, w, scale=(w, h), device=warp.device)
        grid = grid.reshape(1, h, w, 2)
        displacement = warp - grid
        displacement = torch.stack(
            [
                displacement[..., 0] * scale_factor[0],
                displacement[..., 1] * scale_factor[1],
            ],
            1,
        )
        x = [ops.to_chw(f_i), f_ji, self.disp_emb(displacement)]
        if self.radius is not None:
            x.append(self.correlation(x[0], ops.to_chw(f_j), warp, self.radius))
        z = self.hidden_blocks(self.block1(torch.cat(x, 1))).float()
        delta = ops.to_hwc(self.warp_head(z))
        d = ops.to_hwc(self.confidence_head(z))
        warp = warp + torch.stack(
            [delta[..., 0] / (4 * w), delta[..., 1] / (4 * h)], -1
        )
        # Cholesky parameters of the 2x2 precision matrix, in pixels
        l00, l10, l11 = (
            F.softplus(d[..., 1]) + 1e-6,
            d[..., 2],
            F.softplus(d[..., 3]) + 1e-6,
        )
        d = torch.stack([d[..., 0], l00 * l00, l00 * l10, l10 * l10 + l11 * l11], -1)
        mlogit_i = F.pad(mlogit_i, (0, 4 - mlogit_i.shape[-1])) + d
        return warp, mlogit_i


class RoMav2Refiner(nn.Module):
    url = "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt"
    anchor = 512  # RoMa's displacement embedding is scaled relative to 512 x 512
    # stride: (feature dim, projection dim, displacement embed dim, correlation radius)
    configs = {4: (256, 192, 79, 3), 2: (128, 48, 23, 1), 1: (64, 12, 8, None)}

    def __init__(
        self,
        scales: Tuple[int, ...] = (4, 2, 1),
        weights: str = url,
        correlation_backend: str = "torch",
    ) -> None:
        super().__init__()
        if correlation_backend not in ("torch", "triton"):
            raise ValueError("correlation_backend must be 'torch' or 'triton'.")
        correlation = local_correlation
        if correlation_backend == "triton":
            from .utils.correlation_triton import local_correlation as fused

            correlation = partial(fused, native=local_correlation)
        self.scales = scales
        self.features = VGG19BN()
        self.refiners = nn.ModuleDict(
            {
                str(s): ConvRefiner(*self.configs[s], correlation=correlation)
                for s in scales
            }
        )
        own = {}  # only the refiner part of RoMa v2's official checkpoint
        for k, v in ops.load_state_dict(weights, "").items():
            if k.startswith("refiner_features."):
                own["features." + k.removeprefix("refiner_features.")] = v
            elif k.startswith("refiners.") and k.split(".")[1] in self.refiners:
                own[k] = v
        self.load_state_dict(own)

    def extract(
        self, img: pp.Images, resize: pp.Resize = (512, 512)
    ) -> Dict[int, torch.Tensor]:
        """Fine features of images resized with resize_images, keyed by stride."""
        return self.features(pp.resize_images(img, resize))

    _stock_extract = extract

    def extract_selected(
        self, img: pp.Images, resize: pp.Resize = (512, 512)
    ) -> Dict[int, torch.Tensor]:
        """Extract stages needed by this refiner; stride 1 supplies image size.

        This explicit operation may skip later layer hooks in evaluation. In
        training it executes every stage to retain BatchNorm updates. A custom
        extract override is run in full before selecting the required keys.
        Selected caches may not contain keys needed by other scale settings.
        """
        scales = (1, *self.scales)
        extract = self.extract
        if getattr(extract, "__func__", None) is not RoMav2Refiner._stock_extract:
            features = extract(img, resize=resize)
            return {s: features[s] for s in scales}
        return self.features(pp.resize_images(img, resize), scales=scales)

    def forward(
        self,
        feats_i: Dict[int, torch.Tensor],
        feats_j: Dict[int, torch.Tensor],
        gridi_nj: torch.Tensor,
        mlogit_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine coarse grid (b, hc, wc, 2) with logits (b, hc, wc) to refine size."""
        h, w = feats_i[1].shape[1:3]
        scale_factor = (w / self.anchor, h / self.anchor)
        mlogit_i = mlogit_i[..., None]
        for s in self.scales:
            size = (h // s, w // s)
            gridi_nj = bhwc_interpolate(gridi_nj, size)
            mlogit_i = bhwc_interpolate(mlogit_i, size)
            gridi_nj, mlogit_i = self.refiners[str(s)](
                feats_i[s], feats_j[s], gridi_nj, mlogit_i, scale_factor
            )
        return gridi_nj, mlogit_i

    def refine(
        self,
        cgridi_nj: torch.Tensor,
        clogit_i: torch.Tensor,
        feats_i: Dict[int, torch.Tensor],
        feats_j: Dict[int, torch.Tensor],
        size_j: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Grid of image i at refine size in image j (size_j: b x (w, h)), with cov."""
        gridi_nj, mlogit_i = self(feats_i, feats_j, cgridi_nj, clogit_i)
        h_rj, w_rj = feats_j[1].shape[1:3]
        # RoMa's precision is in refine pixels of image j: invert, then scale to image j
        p00, p10, p11 = mlogit_i[..., 1], mlogit_i[..., 2], mlogit_i[..., 3]
        cov = torch.stack([p11, -p10, -p10, p00], -1).unflatten(-1, (2, 2))
        cov = cov / (p00 * p11 - p10 * p10)[..., None, None]
        scale = torch.stack([size_j[:, 0] / w_rj, size_j[:, 1] / h_rj], -1)[
            :, None, None
        ]
        return {
            "dense_p2di_nj": gridi_nj,
            "dense_p2di_ij": ops.denormalize_keypoints(gridi_nj, size_j),
            "dense_confidencei": torch.sigmoid(mlogit_i[..., 0]),
            "dense_covi_ij": cov * scale[..., :, None] * scale[..., None, :],
        }
