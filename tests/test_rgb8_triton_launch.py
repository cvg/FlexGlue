"""The optional Triton path uses portable, explicitly required launch options."""
import importlib.util
import unittest
from unittest.mock import MagicMock, patch, sentinel


@unittest.skipUnless(importlib.util.find_spec("triton"), "Triton is not installed")
class TritonLaunchOptionsTest(unittest.TestCase):
    def test_launch_uses_only_required_options(self):
        from flexglue.utils import rgb8_triton

        source = MagicMock()
        source.shape = (513, 769, 3)
        source.device = "cpu"
        kernel = MagicMock()
        with patch.object(rgb8_triton, "_kernel", return_value=kernel), patch.object(
            rgb8_triton.torch, "empty", return_value=sentinel.output
        ):
            result = rgb8_triton._resize_rgb8(source, sentinel.table, (512, 512), False)
        self.assertIs(result, sentinel.output)
        self.assertEqual(kernel.__getitem__.call_count, 1)
        self.assertEqual(
            kernel.__getitem__.return_value.call_args.kwargs,
            {"num_warps": 4, "enable_fp_fusion": True},
        )


if __name__ == "__main__":
    unittest.main()
