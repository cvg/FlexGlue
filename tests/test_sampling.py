import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flexglue import FlexGlue
from flexglue.utils import ops


class SamplingValidityTest(unittest.TestCase):
    def setUp(self):
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
        self.data = {
            "features": torch.zeros(1, 4, 2, 2),
            "image_size": torch.tensor([[16.0, 16.0]]),
            "resized_size": torch.tensor([[16.0, 16.0]]),
        }
        self.context = SimpleNamespace(covisibility_score=torch.ones(1))
        self.conf = self.model.resolve_conf(
            {"kde_balance": True, "max_num_samples": 8, "max_distance_px": 0}
        )

    def prediction(self):
        grid = ops.cell_centers(2, 2, (2, 2), device="cpu").view(1, 2, 2, 2)
        return {
            **{f"dense_p2d{i}_n{1-i}": grid.clone() for i in (0, 1)},
            **{
                f"dense_p2d{i}_i{1-i}": ops.denormalize_keypoints(
                    grid, self.data["image_size"]
                )
                for i in (0, 1)
            },
            **{f"dense_confidence{i}": torch.ones(1, 2, 2) for i in (0, 1)},
        }

    def sample(self, pred):
        pred = self.model.gate_dense(
            pred, self.context, self.data, self.data, self.conf
        )
        return self.model.sample_matches(pred, self.data, self.data, self.conf)

    def test_rejected_nan_does_not_prevent_other_matches(self):
        for direction in (0, 1):
            for component in (0, 1):
                with self.subTest(direction=direction, component=component):
                    pred = self.prediction()
                    for frame in ("n", "i"):
                        pred[f"dense_p2d{direction}_{frame}{1-direction}"][
                            0, 0, 0, component
                        ] = float("nan")
                    result = self.sample(pred)
                    self.assertEqual(result["matches0"].shape, (1, 8))
                    self.assertEqual((result["matches0"] >= 0).sum().item(), 7)
                    for i in (0, 1):
                        selected = result[f"matches{i}"] >= 0
                        self.assertTrue(
                            torch.isfinite(result[f"keypoints{i}"][selected]).all()
                        )
                    self.assertTrue(torch.isfinite(result["matching_scores0"]).all())

    def test_all_rejected_nan_grids_produce_unmatched_rows(self):
        pred = self.prediction()
        for key, value in pred.items():
            if key.startswith("dense_p2d"):
                value.fill_(float("nan"))
        result = self.sample(pred)
        self.assertTrue((result["matches0"] == -1).all())
        self.assertTrue((result["matching_scores0"] == 0).all())

    def test_finite_rejected_coordinates_are_not_relocated(self):
        pred = self.prediction()
        scores = torch.tensor([[0.0, 1, 0, 1, 1, 0, 1, 0]])
        seen = []

        def record(points, weights, std):
            seen.append(points.clone())
            return torch.ones_like(weights)

        with patch.object(ops, "lattice_kde", side_effect=record):
            ops.kde_weights(pred["dense_p2d0_n1"], pred["dense_p2d1_n0"], scores, 8)
            ops.kde_weights(
                pred["dense_p2d0_n1"], pred["dense_p2d1_n0"], torch.ones_like(scores), 8
            )
        self.assertTrue(torch.equal(seen[0], seen[1]))
