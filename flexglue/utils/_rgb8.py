# SPDX-License-Identifier: MIT
"""Shared upload and ownership for optional CUDA RGB8 preparation."""
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch


def fp32_codebook() -> np.ndarray:
    """FP32 values from float64(code)/255 for every uint8 code."""
    values = np.arange(256, dtype=np.float64) / np.float64(255.0)
    return values.astype(np.float32)


def select_upload_pixels(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """Borrow exact OpenCV BGR backing, or fall back to owning RGB bytes."""
    if (
        type(rgb) is not np.ndarray
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
    ):
        raise ValueError("expected HWC RGB8 NumPy image")
    width = rgb.shape[1]
    if rgb.strides == (width * 3, 3, -1):
        bgr = rgb[:, :, ::-1]
        if bgr.flags.c_contiguous and np.shares_memory(bgr, rgb):
            return bgr, 1
    if rgb.flags.c_contiguous:
        return rgb, 0
    return np.array(rgb, dtype=np.uint8, copy=True, order="C"), 0


def select_resize_samples(source, target_hw):
    """Keep the exact bilinear lattice for integer downscales, NumPy or Torch."""
    height, width, channels = source.shape
    th, tw = target_hw
    if height >= th and width >= tw and height % th == 0 and width % tw == 0:
        sy, sx = height // th, width // tw
        # Odd factors retain the zero-weight neighbour for exceptional LUTs.
        ky, kx = min(sy, 2 + sy % 2), min(sx, 2 + sx % 2)
        if ky < sy or kx < sx:
            y, x = (sy - ky) // 2, (sx - kx) // 2
            return source.reshape(th, sy, tw, sx, channels)[
                :, y : y + ky, :, x : x + kx, :
            ].reshape(th * ky, tw * kx, channels)
    return source


@dataclass
class PreparedDescriptorInput:
    tensor: torch.Tensor
    original_size: torch.Tensor
    owners: tuple
    producer_stream: torch.cuda.Stream

    def require_current_stream(self) -> None:
        current = torch.cuda.current_stream(self.tensor.device)
        if current.cuda_stream != self.producer_stream.cuda_stream:
            raise RuntimeError(
                "descriptor input must be consumed on its producer stream"
            )


def prepare_rgb8(
    rgb: np.ndarray,
    *,
    target_hw: tuple[int, int] = (512, 512),
    device: Union[torch.device, str] = "cuda",
    codebook_device: Optional[torch.Tensor] = None,
    host_sampling: bool = False,
    resize: Callable,
) -> PreparedDescriptorInput:
    if (
        type(rgb) is not np.ndarray
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
    ):
        raise ValueError("expected HWC RGB8 NumPy image")
    if not isinstance(host_sampling, bool):
        raise ValueError("host_sampling must be a bool")
    src_h, src_w = (int(v) for v in rgb.shape[:2])
    if not (
        1 <= src_h <= 65535 and 1 <= src_w <= 65535 and src_h * src_w <= 64_000_000
    ):
        raise ValueError("source grid outside supported bounds")
    if (
        not isinstance(target_hw, tuple)
        or len(target_hw) != 2
        or any(type(v) is not int or not 1 <= v <= 4096 for v in target_hw)
        or target_hw[0] * target_hw[1] * 3 * 4 > 512 * 1024**2
    ):
        raise ValueError("target grid outside supported bounds")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("explicit CUDA device required")
    current_index = torch.cuda.current_device()
    if device.index not in (None, current_index):
        raise ValueError("RGB8 preparation requires the caller's current CUDA device")
    device = torch.device("cuda", current_index)
    stream = torch.cuda.current_stream(device)
    host_pixels, source_is_bgr = select_upload_pixels(rgb)
    if host_sampling:
        host_pixels = np.ascontiguousarray(
            select_resize_samples(host_pixels, target_hw)
        )
    # Blocking Torch copy owns device bytes before the borrowed host view can retire.
    source = torch.from_numpy(host_pixels).to(device=device, non_blocking=False)
    if (
        source.dtype != torch.uint8
        or tuple(source.shape) != host_pixels.shape
        or not source.is_contiguous()
    ):
        raise RuntimeError("RGB8 upload contract differs")
    device = source.device
    if codebook_device is None:
        table = torch.from_numpy(fp32_codebook()).to(device=device, non_blocking=False)
    else:
        table = codebook_device
        if (
            not isinstance(table, torch.Tensor)
            or table.device != device
            or table.dtype != torch.float32
            or tuple(table.shape) != (256,)
            or not table.is_contiguous()
        ):
            raise ValueError("caller-owned codebook binding differs")
    output = resize(source, table, target_hw, bool(source_is_bgr))
    if (
        output.device != device
        or output.dtype != torch.float32
        or tuple(output.shape) != (1, 3, *target_hw)
        or not output.is_contiguous()
    ):
        raise RuntimeError("output must be contiguous CUDA FP32 NCHW")
    source.record_stream(stream)
    table.record_stream(stream)
    output.record_stream(stream)
    size = torch.tensor([[src_w, src_h]], device=device, dtype=torch.float32)
    return PreparedDescriptorInput(
        output, size, (rgb, host_pixels, source, table), stream
    )


@torch.no_grad()
def describe_prepared(
    descriptor, prepared: PreparedDescriptorInput, *, keypoints=None
) -> dict:
    prepared.require_current_stream()
    x = prepared.tensor
    features = descriptor._forward({"image": x}, resize=tuple(x.shape[-2:]))["features"]
    pred = {"features": features, "image_size": prepared.original_size.to(x)}
    if keypoints is not None:
        if descriptor.conf.describe:
            pred["descriptors"] = descriptor.sample_descriptors(
                features, keypoints, pred["image_size"]
            )
        pred["keypoints"] = keypoints
    return pred
