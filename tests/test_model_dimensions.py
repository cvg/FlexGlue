import unittest
from unittest.mock import patch

import numpy as np
import torch

from flexglue import FlexGlue


class ModelDimensionsTest(unittest.TestCase):
    def test_invalid_dimensions_fail_before_allocation_or_loading(self):
        for dim, heads in (
            (0, 4),
            (-8, 4),
            (32, 0),
            (32, -1),
            (32.0, 4),
            (32, 4.0),
            (True, 4),
            (np.bool_(True), 4),
            (32, np.bool_(True)),
            (32, True),
            (32, 3),
            (10, 2),
        ):
            with self.subTest(dim=dim, heads=heads), patch(
                "flexglue.flexglue.nn.Linear"
            ) as allocate, patch.object(FlexGlue, "load_weights") as load:
                with self.assertRaises(ValueError):
                    FlexGlue(
                        weights="not-read",
                        describe=False,
                        n_layers=1,
                        descriptor_dim=dim,
                        num_heads=heads,
                    )
                allocate.assert_not_called()
                load.assert_not_called()

    def test_integer_protocol_and_raw_dimensions_remain_supported(self):
        for depth, dim, heads in (
            (1, 32, 4),
            (1, np.int64(32), np.int64(4)),
            (0, 10, 2),
            (0, 32, 3),
        ):
            with self.subTest(
                depth=depth, dim=dim, heads=heads
            ), torch.inference_mode():
                torch.manual_seed(23)
                model = FlexGlue(
                    weights=None,
                    describe=False,
                    input_dim=4,
                    n_layers=depth,
                    descriptor_dim=dim,
                    num_heads=heads,
                    search_dim=4,
                    mp=None,
                ).eval()
                data = {
                    "features": torch.rand(1, 4, 2, 3),
                    "image_size": torch.tensor([[24.0, 16.0]]),
                }
                pred = model.match_dense(
                    data, data, symmetric=False, max_distance_px=None
                )
                self.assertEqual(pred["dense_p2d0_i1"].shape, (1, 2, 3, 2))
                self.assertTrue(torch.isfinite(pred["dense_p2d0_i1"]).all())
                self.assertTrue(torch.isfinite(pred["dense_confidence0"]).all())


if __name__ == "__main__":
    unittest.main()
