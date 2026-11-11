# SPDX-License-Identifier: MIT
"""Optional eager threshold fusion; native scans preserve accumulation order."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_transforms_active = getattr(torch._C, "_are_functorch_transforms_active", None)


@triton.jit
def _threshold(
    ybuf, sums, square_sums, selected, index, K: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, BLOCK)
    y = tl.load(ybuf + row * K + t)
    rho = (t + 1).to(tl.float32)
    mean = tl.div_rn(tl.load(sums + row * K + t), rho)
    mean_sq = tl.div_rn(tl.load(square_sums + row * K + t), rho)
    delta = tl.div_rn(1.0 - rho * (mean_sq - mean * mean), rho)
    tau = mean - libdevice.sqrt_rn(
        tl.maximum(delta, 0.0, propagate_nan=tl.PropagateNan.ALL)
    )
    support = tl.sum((tau <= y).to(tl.int32), 0)
    # Gather below owns the invalid -1 behavior. Do not read tau[-1] here.
    chosen = tl.sum(tl.where(t == support - 1, tau, 0.0), 0)
    tl.store(selected + row, chosen)
    tl.store(index + row, tl.where(support > 0, 0, -1).to(tl.int64))


def threshold(ys, *, native):
    """Fuse pointwise threshold work, retaining native scans and other cases."""
    if (
        torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or type(ys) is not torch.Tensor
        or _transforms_active is None
        or _transforms_active()
        or torch.jit.is_tracing()
        or torch.jit.is_scripting()
        or torch.onnx.is_in_onnx_export()
    ):
        return native(ys)
    if (
        torch.autograd.forward_ad.unpack_dual(ys).tangent is not None
        or not ys.is_cuda
        or ys.dtype != torch.float32
        or ys.ndim < 1
        or ys.shape[-1] != 512
        or not ys.is_contiguous()
        or ys.numel() == 0
    ):
        return native(ys)
    k = ys.shape[-1]
    rows = ys.numel() // k
    sums = ys.cumsum(-1)
    square_sums = ys.square().cumsum(-1)
    selected = torch.empty((*ys.shape[:-1], 1), device=ys.device, dtype=ys.dtype)
    index = torch.empty_like(selected, dtype=torch.int64)
    with torch.cuda.device(ys.device):
        _threshold[(rows,)](
            ys, sums, square_sums, selected, index, k, k, enable_fp_fusion=False
        )
    # Native gather retains the error for inputs with no valid support.
    return selected.gather(-1, index)
