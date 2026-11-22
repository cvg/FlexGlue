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
        self.assertEqual(result["matches0"].shape, (1, 8))
        self.assertEqual(result["matching_scores0"].shape, (1, 8))
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

    def test_empty_cell_selection_preserves_shapes_storage_and_rng(self):
        for pool_size, limit in ((5, 0), (0, 3), (0, 0)):
            with self.subTest(pool_size=pool_size, limit=limit):
                backing = torch.arange(2 * pool_size * 2, dtype=torch.float64)
                weights = backing.reshape(2, pool_size * 2)[:, ::2]
                before = weights.clone()
                strides = weights.stride()
                rng = torch.get_rng_state().clone()
                indices, xy_indices = ops.sample_cells(weights, limit)
                self.assertEqual(indices.shape, (2, 0))
                self.assertEqual(xy_indices.shape, (2, 0, 2))
                for result in (indices, xy_indices):
                    self.assertEqual(result.dtype, torch.int64)
                    self.assertEqual(result.device, weights.device)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertTrue(torch.equal(weights, before))
                self.assertEqual(weights.stride(), strides)

    def test_public_zero_sample_limit_keeps_empty_outputs(self):
        data0 = {
            "features": torch.zeros(2, 4, 2, 3),
            "image_size": torch.tensor([[24.0, 16.0], [48.0, 32.0]]),
        }
        data1 = {
            "features": torch.zeros(2, 4, 3, 2),
            "image_size": torch.tensor([[16.0, 24.0], [32.0, 48.0]]),
        }
        for balanced in (False, True):
            with self.subTest(kde_balance=balanced), torch.no_grad():
                rng = torch.get_rng_state().clone()
                result = self.model.match(
                    data0,
                    data1,
                    max_num_samples=0,
                    kde_balance=balanced,
                    max_distance_px=0,
                )
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                for direction in (0, 1):
                    self.assertEqual(result[f"keypoints{direction}"].shape, (2, 0, 2))
                    self.assertEqual(result[f"matches{direction}"].shape, (2, 0))
                    self.assertEqual(result[f"matches{direction}"].dtype, torch.int64)
                    self.assertEqual(
                        result[f"matching_scores{direction}"].shape, (2, 0)
                    )
                    self.assertEqual(
                        result[f"matching_scores{direction}"].dtype, torch.float32
                    )
                self.assertIs(result["matches0"], result["matches1"])
                self.assertIs(result["matching_scores0"], result["matching_scores1"])
                self.assertEqual(result["dense_confidence0"].shape, (2, 2, 3))
                self.assertEqual(result["dense_confidence1"].shape, (2, 3, 2))

    def test_positive_cell_selection_keeps_seeded_multinomial(self):
        for dtype in (torch.float32, torch.float64):
            weights = torch.tensor([[0, 0.1, 0.4, 0.5], [1, 0, 0, 0]], dtype=dtype)
            before = weights.clone()
            for limit in (1, 4, 8):
                with self.subTest(dtype=dtype, limit=limit), torch.random.fork_rng():
                    torch.manual_seed(20261122)
                    expected = torch.multinomial(
                        weights.clamp_min(1e-12), min(limit, 4), replacement=False
                    )
                    expected_rng = torch.get_rng_state().clone()
                    torch.manual_seed(20261122)
                    actual, xy_indices = ops.sample_cells(weights, limit)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(
                        torch.equal(xy_indices, expected[..., None].expand(-1, -1, 2))
                    )
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertTrue(torch.equal(weights, before))
        with self.assertRaises(RuntimeError):
            ops.sample_cells(weights, -1)

    def test_empty_sampling_keeps_gather_gradients_connected(self):
        pred = self.prediction()
        leaves = []
        for key in (
            "dense_p2d0_i1",
            "dense_p2d1_i0",
            "dense_confidence0",
            "dense_confidence1",
        ):
            pred[key] = pred[key].detach().requires_grad_()
            leaves.append(pred[key])
        conf = self.model.resolve_conf({"max_num_samples": 0, "kde_balance": False})
        result = self.model.sample_matches(pred, self.data, self.data, conf)
        loss = sum(
            result[key].sum()
            for key in ("keypoints0", "keypoints1", "matching_scores0")
        )
        self.assertTrue(loss.requires_grad)
        loss.backward()
        for leaf in leaves:
            self.assertIsNotNone(leaf.grad)
            self.assertTrue(torch.equal(leaf.grad, torch.zeros_like(leaf)))

    def test_half_empty_rows_return_unique_indices(self):
        cases = (
            torch.zeros(2, 4, dtype=torch.float16),
            torch.tensor([[0, 0.25, 0.5, 0.75], [0, 0, 0, 0]], dtype=torch.float16),
        )
        for weights in cases:
            with self.subTest(weights=weights), torch.random.fork_rng():
                before = weights.clone()
                indices, xy_indices = ops.sample_cells(weights, 3)
                self.assertEqual(indices.shape, (2, 3))
                self.assertEqual(xy_indices.shape, (2, 3, 2))
                for row in indices:
                    self.assertEqual(row.unique().numel(), 3)
                    self.assertTrue(((row >= 0) & (row < 4)).all())
                self.assertTrue(torch.equal(weights, before))

    def test_valid_half_rows_keep_seeded_values_rng_and_subnormals(self):
        cases = (
            torch.tensor(
                [[0.125, 0.25, 0.5, 1], [0.75, 0.25, 0.5, 1]], dtype=torch.float16
            ),
            torch.tensor([[1, 2, 4, 8], [16, 8, 4, 2]], dtype=torch.float16) * 2**-24,
            torch.tensor([[0, 0.25, 0.5, 0.75], [1, 0, 0, 0]], dtype=torch.float16),
            torch.tensor([[0, 1, 2, 4], [1, 0, 0, 0]], dtype=torch.float16) * 2**-24,
        )
        self.assertEqual(cases[1].min().item(), 2**-24)
        for weights in cases:
            with self.subTest(weights=weights), torch.random.fork_rng():
                before = weights.clone()
                torch.manual_seed(31)
                expected = torch.multinomial(
                    weights.clamp_min(1e-12), 3, replacement=False
                )
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(31)
                actual, _ = ops.sample_cells(weights, 3)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                self.assertTrue(torch.equal(weights, before))

    def test_public_half_rejection_without_kde(self):
        with torch.no_grad():
            self.model.heads[0].to_search.weight.zero_()
            self.model.heads[0].to_search.bias.fill_(1)
        data = {
            "features": torch.zeros(1, 4, 4, 4),
            "image_size": torch.tensor([[32.0, 32.0]]),
        }
        for dtype, balanced in (
            (torch.float32, False),
            (torch.bfloat16, False),
            (torch.float16, False),
            (torch.float16, True),
        ):
            with self.subTest(dtype=dtype, kde_balance=balanced), torch.no_grad():
                with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
                    result = self.model.match(
                        data,
                        data,
                        max_num_samples=4,
                        kde_balance=balanced,
                        max_distance_px=0,
                        filter_threshold=0.5,
                    )
                self.assertEqual(result["matches0"].shape, (1, 4))
                self.assertTrue((result["matches0"] == -1).all())
                self.assertTrue((result["matching_scores0"] == 0).all())
                self.assertEqual(result["matching_scores0"].dtype, dtype)
