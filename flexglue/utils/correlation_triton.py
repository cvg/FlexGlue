# SPDX-License-Identifier: MIT
"""Optional eager CUDA local correlation; imported only on explicit selection."""

import math

import torch
import triton
import triton.language as tl

INT32_MAX = 2**31 - 1
_transforms_active = getattr(torch._C, "_are_functorch_transforms_active", None)


def eligible(a, b, warp, radius):
    if (
        torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or not all(type(x) is torch.Tensor for x in (a, b, warp))
        or _transforms_active is None
        or _transforms_active()
        or torch.jit.is_tracing()
        or torch.jit.is_scripting()
        or torch.onnx.is_in_onnx_export()
        or torch.get_default_dtype() != torch.float32
    ):
        return False
    if any(
        torch.autograd.forward_ad.unpack_dual(x).tangent is not None
        for x in (a, b, warp)
    ):
        return False
    if (
        type(radius) is not int
        or radius != 1
        or a.ndim != 4
        or b.shape != a.shape
        or warp.ndim != 4
        or not a.is_cuda
        or a.device != b.device
        or a.device != warp.device
        or any(x.dtype != torch.float32 for x in (a, b, warp))
    ):
        return False
    n, c, h, w = a.shape
    if c != 48 or min(n, h, w) <= 0 or warp.shape != (n, h, w, 2):
        return False
    if n * h * w * 9 > INT32_MAX:
        return False
    # The measured reduction order covers these owning feature layouts.
    if not all(
        x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last)
        for x in (a, b)
    ):
        return False
    for x in (a, b, warp):
        if any(s < 0 for s in x.stride()):
            return False
        if (
            sum((size - 1) * stride for size, stride in zip(x.shape, x.stride()))
            > INT32_MAX
        ):
            return False
    return True


@triton.jit
def fused(
    A,
    B,
    WARP,
    WIN,
    OUT,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    a0: tl.constexpr,
    a1: tl.constexpr,
    a2: tl.constexpr,
    a3: tl.constexpr,
    b0: tl.constexpr,
    b1: tl.constexpr,
    b2: tl.constexpr,
    b3: tl.constexpr,
    w0: tl.constexpr,
    w1: tl.constexpr,
    w2: tl.constexpr,
    w3: tl.constexpr,
    N: tl.constexpr,
    BC: tl.constexpr,
    BP: tl.constexpr,
    SCALE: tl.constexpr,
    TORCH_ORDER: tl.constexpr,
):
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    c = tl.arange(0, BC)
    valid = p < N
    k = p % K
    x = (p // K) % W
    y = (p // (K * W)) % H
    batch = p // (K * W * H)
    gx = tl.load(WARP + batch * w0 + y * w1 + x * w2, valid, 0) + tl.load(
        WIN + 2 * k, valid, 0
    )
    gy = tl.load(WARP + batch * w0 + y * w1 + x * w2 + w3, valid, 0) + tl.load(
        WIN + 2 * k + 1, valid, 0
    )
    ix = ((gx + 1.0) * W - 1.0) * 0.5
    iy = ((gy + 1.0) * H - 1.0) * 0.5
    # CUDA grid_sample handles non-finite/out-of-range coordinates as zero; keep
    # integer conversion safe, and mask each neighbor independently.
    safe = (
        (ix == ix)
        & (iy == iy)
        & (ix > -2.0)
        & (ix < W + 1.0)
        & (iy > -2.0)
        & (iy < H + 1.0)
    )
    ix = tl.where(safe, ix, 0.0)
    iy = tl.where(safe, iy, 0.0)
    x0 = tl.floor(ix).to(tl.int32)
    y0 = tl.floor(iy).to(tl.int32)
    dx = ix - x0
    dy = iy - y0
    sample = tl.full((BP, BC), 0.0, tl.float32)
    for oy in range(2):
        for ox in range(2):
            xx = x0 + ox
            yy = y0 + oy
            m = valid & safe & (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            v = tl.load(
                B
                + batch[:, None] * b0
                + c[None, :] * b1
                + yy[:, None] * b2
                + xx[:, None] * b3,
                m[:, None] & (c[None, :] < C),
                0,
            )
            wx = tl.where(ox == 0, 1.0 - dx, dx)
            wy = tl.where(oy == 0, 1.0 - dy, dy)
            sample += v * (wx * wy)[:, None]
    av = tl.load(
        A + batch[:, None] * a0 + c[None, :] * a1 + y[:, None] * a2 + x[:, None] * a3,
        valid[:, None] & (c[None, :] < C),
        0,
    )
    product = (av / SCALE) * sample
    if TORCH_ORDER:
        q = tl.arange(0, 4)
        accum = tl.full((BP, 4), 0.0, tl.float32)
        for j in range(tl.cdiv(C, 4)):
            idx = tl.broadcast_to((4 * j + q)[None, :], (BP, 4))
            accum += tl.gather(product, idx, axis=1)
        z0 = tl.sum(tl.where(q[None, :] == 0, accum, 0.0), axis=1)
        z1 = tl.sum(tl.where(q[None, :] == 1, accum, 0.0), axis=1)
        z2 = tl.sum(tl.where(q[None, :] == 2, accum, 0.0), axis=1)
        z3 = tl.sum(tl.where(q[None, :] == 3, accum, 0.0), axis=1)
        value = ((z0 + z1) + z2) + z3
    else:
        value = tl.sum(product, axis=1)
    tl.store(OUT + ((batch * H + y) * W + x) * K + k, value, valid)


def _correlation(a, b, warp, radius):
    assert a.dtype == b.dtype == warp.dtype == torch.float32 and a.is_cuda
    n, c, h, w = a.shape
    k = (2 * radius + 1) ** 2
    r = torch.linspace(-2 * radius, 2 * radius, 2 * radius + 1, device=a.device)
    dy, dx = torch.meshgrid(r / h, r / w, indexing="ij")
    window = torch.stack([dx, dy], -1).reshape(-1, 2)
    out = torch.empty((n, h, w, k), device=a.device, dtype=a.dtype).permute(0, 3, 1, 2)
    count = n * h * w * k
    fused[(triton.cdiv(count, 16),)](
        a,
        b,
        warp,
        window,
        out,
        c,
        h,
        w,
        k,
        *a.stride(),
        *b.stride(),
        *warp.stride(),
        count,
        triton.next_power_of_2(c),
        16,
        math.sqrt(c),
        True,
        enable_fp_fusion=True,
    )
    return out


def local_correlation(a, b, warp, radius, *, native):
    """Fuse supported eager inference; native owns all remaining cases."""
    if not eligible(a, b, warp, radius):
        return native(a, b, warp, radius)
    with torch.cuda.device(a.device):
        return _correlation(a, b, warp, radius)
