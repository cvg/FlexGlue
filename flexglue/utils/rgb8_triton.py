# SPDX-License-Identifier: MIT
"""Optional Triton RGB8 preparation for CUDA descriptors."""
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import triton
import triton.language as tl

from ._rgb8 import PreparedDescriptorInput as PreparedDescriptorInput
from ._rgb8 import describe_prepared as describe_prepared
from ._rgb8 import fp32_codebook as fp32_codebook
from ._rgb8 import prepare_rgb8 as _prepare_rgb8
from ._rgb8 import select_upload_pixels as select_upload_pixels

__all__ = [
    "PreparedDescriptorInput",
    "describe_prepared",
    "fp32_codebook",
    "prepare_rgb8",
    "select_upload_pixels",
]


@lru_cache(maxsize=1)
def _kernel():
    @triton.jit
    def resize(
        src,
        table,
        out,
        SRC_W: tl.constexpr,
        SRC_H: tl.constexpr,
        DST_W: tl.constexpr,
        DST_H: tl.constexpr,
        SCALE_X: tl.constexpr,
        SCALE_Y: tl.constexpr,
        SOURCE_IS_BGR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < (DST_W * DST_H * 3)
        channel = i % 3
        source_channel = tl.where(SOURCE_IS_BGR, 2 - channel, channel)
        pixel = i // 3
        ox = pixel % DST_W
        oy = pixel // DST_W
        output_i = channel * (DST_W * DST_H) + pixel
        if SRC_W == DST_W and SRC_H == DST_H:
            code = tl.load(src + pixel * 3 + source_channel, mask, other=0).to(tl.int32)
            value = tl.load(table + code, mask, other=0.0)
        else:
            # Host-rounded FP32 scales, align_corners=False half-pixel source order.
            scale_x = tl.full((BLOCK,), SCALE_X, tl.float32)
            scale_y = tl.full((BLOCK,), SCALE_Y, tl.float32)
            sx = tl.maximum(scale_x * (ox.to(tl.float32) + 0.5) - 0.5, 0.0)
            sy = tl.maximum(scale_y * (oy.to(tl.float32) + 0.5) - 0.5, 0.0)
            x0 = tl.minimum(tl.floor(sx).to(tl.int32), SRC_W - 1)
            y0 = tl.minimum(tl.floor(sy).to(tl.int32), SRC_H - 1)
            x1 = tl.minimum(x0 + 1, SRC_W - 1)
            y1 = tl.minimum(y0 + 1, SRC_H - 1)
            wx = sx - x0.to(tl.float32)
            wy = sy - y0.to(tl.float32)
            a = (y0 * SRC_W + x0) * 3 + source_channel
            b = (y0 * SRC_W + x1) * 3 + source_channel
            c = (y1 * SRC_W + x0) * 3 + source_channel
            d = (y1 * SRC_W + x1) * 3 + source_channel
            v00 = tl.load(
                table + tl.load(src + a, mask, other=0).to(tl.int32), mask, other=0.0
            )
            v01 = tl.load(
                table + tl.load(src + b, mask, other=0).to(tl.int32), mask, other=0.0
            )
            v10 = tl.load(
                table + tl.load(src + c, mask, other=0).to(tl.int32), mask, other=0.0
            )
            v11 = tl.load(
                table + tl.load(src + d, mask, other=0).to(tl.int32), mask, other=0.0
            )
            w0, w1 = 1.0 - wx, wx
            h0, h1 = 1.0 - wy, wy
            value = h0 * (w0 * v00 + w1 * v01) + h1 * (w0 * v10 + w1 * v11)
        tl.store(out + output_i, value, mask)

    return resize


def _resize_rgb8(source, table, target_hw, source_is_bgr):
    src_h, src_w = source.shape[:2]
    dst_h, dst_w = target_hw
    output = torch.empty(
        (1, 3, dst_h, dst_w), dtype=torch.float32, device=source.device
    )
    count = dst_h * dst_w * 3
    # Request FMA contraction explicitly for the qualified interpolation order.
    _kernel()[((count + 255) // 256,)](
        source,
        table,
        output,
        src_w,
        src_h,
        dst_w,
        dst_h,
        float(np.float32(src_w / dst_w)),
        float(np.float32(src_h / dst_h)),
        bool(source_is_bgr),
        256,
        num_warps=4,
        enable_fp_fusion=True,
    )
    return output


def prepare_rgb8(
    rgb: np.ndarray,
    *,
    target_hw: tuple[int, int] = (512, 512),
    device: Union[torch.device, str] = "cuda",
    codebook_device: Optional[torch.Tensor] = None,
    host_sampling: bool = False,
) -> PreparedDescriptorInput:
    return _prepare_rgb8(
        rgb,
        target_hw=target_hw,
        device=device,
        codebook_device=codebook_device,
        host_sampling=host_sampling,
        resize=_resize_rgb8,
    )
