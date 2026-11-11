"""Optional readout fusion; CUDA tests require Triton and a CUDA device."""
import functools
import importlib.util
import unittest
from unittest.mock import patch

import torch

from flexglue import FlexGlue
from flexglue.utils import ops

HAS_TRITON = importlib.util.find_spec("triton") is not None


class ReadoutSelectionTest(unittest.TestCase):
    def test_default_fixed_and_invalid(self):
        with patch.object(
            ops, "load_state_dict", side_effect=AssertionError("weight read")
        ):
            with self.assertRaisesRegex(ValueError, "readout_backend"):
                FlexGlue(readout_backend="bad")
        model = FlexGlue(weights=None, describe=False, n_layers=0)
        self.assertIs(model.heads[0].threshold, ops.sorted_entmax15_threshold)
        with self.assertRaisesRegex(KeyError, "Fixed"):
            model.resolve_conf({"readout_backend": "triton"})

    def test_default_does_not_import_triton(self):
        import builtins

        original = builtins.__import__

        def checked(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise AssertionError("default imported Triton")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=checked):
            FlexGlue(weights=None, describe=False, n_layers=0)

    def test_threshold_injection_and_low_precision(self):
        grid = ops.cell_centers(24, 24, scale=(24, 24), device="cpu")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            logits = torch.randn(1, 2, 576).to(dtype)
            calls = []

            def threshold(ys):
                calls.append(ys.dtype)
                return ops.sorted_entmax15_threshold(ys)

            expected = ops.entmax15_window_readout(logits, grid, (24, 24))
            actual = ops.entmax15_window_readout(
                logits, grid, (24, 24), threshold=threshold
            )
            self.assertEqual(calls, [torch.float32])
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    @unittest.skipUnless(HAS_TRITON, "Triton not installed")
    def test_cpu_fallback(self):
        from flexglue.utils.entmax_triton import threshold

        ys = torch.randn(2, 512).sort(descending=True).values
        for state in (torch.enable_grad(), torch.no_grad()):
            with state:
                torch.testing.assert_close(
                    threshold(ys, native=ops.sorted_entmax15_threshold),
                    ops.sorted_entmax15_threshold(ys),
                    rtol=0,
                    atol=0,
                )


@unittest.skipUnless(
    HAS_TRITON and torch.cuda.is_available(), "CUDA and Triton required"
)
class ReadoutCUDATest(unittest.TestCase):
    def test_supported_and_fallback_shapes(self):
        from flexglue.utils import entmax_triton

        kernel = entmax_triton._threshold
        launches = []

        class Spy:
            def __getitem__(self, grid):
                launches.append(grid)
                return kernel[grid]

        torch.manual_seed(1701)
        with torch.no_grad(), patch.object(entmax_triton, "_threshold", Spy()):
            for shape in ((1, 512), (2, 3, 512), (2, 511), (0, 512)):
                before = len(launches)
                ys = torch.randn(shape, device="cuda").sort(descending=True).values
                torch.testing.assert_close(
                    entmax_triton.threshold(ys, native=ops.sorted_entmax15_threshold),
                    ops.sorted_entmax15_threshold(ys),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                self.assertEqual(
                    len(launches) - before, int(shape[-1] == 512 and ys.numel() > 0)
                )
            ys = torch.zeros(2, 512, device="cuda")
            torch.testing.assert_close(
                entmax_triton.threshold(ys, native=ops.sorted_entmax15_threshold),
                ops.sorted_entmax15_threshold(ys),
                rtol=0,
                atol=0,
            )

    def test_ineligible_and_compiled_delegate(self):
        from flexglue.utils import entmax_triton

        ys = torch.randn(2, 512, device="cuda").sort(descending=True).values
        sentinel = object()
        with patch.object(
            entmax_triton,
            "_threshold",
            side_effect=AssertionError("unsupported launch"),
        ):
            with torch.enable_grad():
                self.assertIs(
                    entmax_triton.threshold(ys, native=lambda x: sentinel), sentinel
                )
            with torch.no_grad():
                for x in (ys.double(), ys[:, :511], ys[:, ::2]):
                    self.assertIs(
                        entmax_triton.threshold(x, native=lambda x: sentinel), sentinel
                    )
                fn = torch.compile(
                    functools.partial(
                        entmax_triton.threshold, native=ops.sorted_entmax15_threshold
                    ),
                    backend="eager",
                    fullgraph=True,
                )
                torch.testing.assert_close(
                    fn(ys), ops.sorted_entmax15_threshold(ys), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
