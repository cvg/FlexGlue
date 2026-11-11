"""Generated RGB8 preparation benchmark; excludes decode and descriptor inference.

Run after installing FlexGlue: python benchmarks/benchmark_rgb8.py --help
"""
import argparse
import functools
import json
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from flexglue.utils import rgb8_torch
from flexglue.utils._rgb8 import fp32_codebook, prepare_rgb8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("torch", "torch-compile", "torch-full-compile", "triton"),
        default=["torch", "torch-compile"],
    )
    args = parser.parse_args()
    if min(args.height, args.width, args.repeats) < 1:
        parser.error("dimensions and repeats must be positive")
    rgb = np.random.default_rng(0).integers(
        0, 256, (args.height, args.width, 3), dtype=np.uint8
    )
    table = torch.from_numpy(fp32_codebook()).cuda()
    functions = {}
    for backend in args.backends:
        if backend == "triton":
            from flexglue.utils.rgb8_triton import prepare_rgb8 as triton_prepare

            functions[backend] = functools.partial(
                triton_prepare, rgb, codebook_device=table
            )
        elif backend == "torch-full-compile":
            # Reference: compile the full-resolution conversion/layout only.
            convert = torch.compile(rgb8_torch._convert_rgb8, dynamic=True)

            def full_resize(source, lut, target_hw, bgr, converter=convert):
                image = converter(source, lut, bgr)
                if source.shape[:2] != target_hw:
                    image = F.interpolate(
                        image, size=target_hw, mode="bilinear", align_corners=False
                    )
                return image

            functions[backend] = functools.partial(
                prepare_rgb8, rgb, codebook_device=table, resize=full_resize
            )
        else:
            functions[backend] = functools.partial(
                rgb8_torch.prepare_rgb8,
                rgb,
                codebook_device=table,
                compile_conversion=backend == "torch-compile",
            )
    expected = F.interpolate(
        torch.from_numpy((rgb / 255).astype(np.float32))
        .permute(2, 0, 1)[None]
        .contiguous()
        .cuda(),
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )
    expected_bits = expected.view(torch.int32)
    cold = {}
    with torch.inference_mode():
        for name, function in functions.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = function()
            torch.cuda.synchronize()
            cold[name] = time.perf_counter() - start
            if not torch.equal(result.tensor.view(torch.int32), expected_bits):
                raise RuntimeError("pixel mismatch: " + name)
            del result
            for _ in range(3):
                result = function()
                del result
        torch.cuda.synchronize()
        samples: dict[str, list[float]] = {name: [] for name in functions}
        peaks: dict[str, list[int]] = {name: [] for name in functions}
        names = list(functions)
        for repeat in range(args.repeats):
            order = names if repeat % 2 == 0 else names[::-1]
            for name in order:
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                result = functions[name]()
                torch.cuda.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000)
                peaks[name].append(torch.cuda.max_memory_allocated() - before)
                del result
    print(
        json.dumps(
            {
                "scope": (
                    "blocking uint8 upload, conversion, resize and metadata; "
                    "no decode/model; allocator cache warm"
                ),
                "shape": [args.height, args.width, 3],
                "target": [512, 512],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "gpu": torch.cuda.get_device_name(),
                "backends": {
                    name: {
                        "first_call_seconds": cold[name],
                        "wall_ms": samples[name],
                        "median_ms": statistics.median(samples[name]),
                        "peak_extra_allocated_bytes": max(peaks[name]),
                        "bitwise_pixels": True,
                    }
                    for name in names
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
