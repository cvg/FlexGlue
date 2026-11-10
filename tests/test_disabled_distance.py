import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flexglue import FlexGlue
from flexglue.utils import ops


class DisabledDistanceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = FlexGlue(
            weights=None,
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=0,
            mp=None,
        ).eval()
        self.data = [
            {
                "features": torch.rand(2, 4, 2, 3),
                "image_size": torch.tensor([[24.0, 16.0], [24.0, 16.0]]),
            }
            for _ in range(2)
        ]

    def test_disabled_dense_thresholds_agree(self):
        for symmetric in (False, True):
            with self.subTest(symmetric=symmetric), torch.inference_mode():
                expected = self.model.match_dense(
                    *self.data, symmetric=symmetric, max_distance_px=None
                )
                for distance in (0, -1, -10.0):
                    actual = self.model.match_dense(
                        *self.data, symmetric=symmetric, max_distance_px=distance
                    )
                    self.assertEqual(expected.keys(), actual.keys())
                    for name in expected:
                        if isinstance(expected[name], torch.Tensor):
                            self.assertTrue(
                                torch.equal(expected[name], actual[name]), name
                            )
                        else:
                            self.assertEqual(expected[name], actual[name])

    def test_positive_threshold_still_requires_symmetric_readout(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self.model.match_dense(*self.data, symmetric=False, max_distance_px=3.0)
        with torch.inference_mode():
            actual = self.model.match_dense(
                *self.data, symmetric=True, max_distance_px=3.0
            )
        self.assertIn("dense_p2d0_i1", actual)
        self.assertIn("dense_p2d1_i0", actual)

    def test_disabled_refinement_keeps_requested_directions(self):
        for distance in (None, 0, -1, -10.0, 3.0):
            with self.subTest(distance=distance):
                conf = SimpleNamespace(refine=True, max_distance_px=distance)
                with patch.object(
                    self.model, "grid_correspondences", return_value={}
                ) as grid:
                    self.model.prepare_dense(object(), *self.data, [(0, 1)], conf)
                expected_directions = [0, 1] if distance == 3.0 else [0]
                self.assertEqual(
                    [call.args[1] for call in grid.call_args_list], expected_directions
                )
                for call, i in zip(grid.call_args_list, expected_directions):
                    self.assertIs(call.args[2], self.data[i])
                    self.assertIs(call.args[3], self.data[1 - i])

    def test_cycle_gate_retains_or_filters_known_cells(self):
        size = torch.tensor([[24.0, 16.0]])
        data = [{"image_size": size, "resized_size": size} for _ in range(2)]
        grid = ops.grid_sources(torch.zeros(1, 1, 3, 2))
        context = SimpleNamespace(covisibility_score=torch.ones(1))
        for distance in (None, 0, -1, -10.0, 3.0):
            with self.subTest(distance=distance):
                conf = self.model.resolve_conf(
                    {
                        "max_distance_px": distance,
                        "remove_border_px": 0,
                    }
                )
                pred = {}
                for i, j in ((0, 1), (1, 0)):
                    pred[f"dense_p2d{i}_n{j}"] = grid.clone()
                    pred[f"dense_p2d{i}_i{j}"] = ops.denormalize_keypoints(grid, size)
                    pred[f"dense_confidence{i}"] = torch.full((1, 1, 3), 0.8)
                with patch.object(
                    self.model,
                    "cycle_error",
                    return_value=(
                        torch.tensor([[0.0, 2.0, 4.0]]),
                        torch.ones(1, 3),
                    ),
                ) as cycle:
                    actual = self.model.gate_dense(pred, context, *data, conf)
                expected = torch.tensor([[[True, True, distance != 3.0]]])
                self.assertEqual(cycle.call_count, 2 if distance == 3.0 else 0)
                for i in (0, 1):
                    self.assertTrue(torch.equal(actual[f"dense_valid{i}"], expected))
                    expected_confidence = torch.tensor(
                        [[[0.8, 0.8, 0.0 if distance == 3.0 else 0.8]]]
                    )
                    self.assertTrue(
                        torch.equal(actual[f"dense_confidence{i}"], expected_confidence)
                    )


if __name__ == "__main__":
    unittest.main()
