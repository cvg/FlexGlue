"""Public sparse regressions for rejected geometry and zero confidence.

Use PYTHONPATH=<exact source/worktree> python this_file.py. No checkpoint needed.
Geometry probes replace learned predictions with deterministic outputs; all
matching consumers run. The zero-confidence regression uses the actual model.
"""

import unittest
from unittest.mock import patch

import torch

from flexglue import FlexGlue


class RejectedSparseGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def setUp(self):
        torch.manual_seed(20261109)
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
            "features": torch.randn(1, 4, 2, 2),
            "image_size": torch.tensor([[16.0, 16.0]]),
            "keypoints": torch.tensor([[[3.0, 3.0], [8.0, 8.0], [12.0, 12.0]]]),
        }

    def run_case(self, gemm, exact, direction, component, value, all_rows=False):
        count = 0

        def predicted(layer, points, *features):
            nonlocal count
            out = points.clone()
            if count == direction:
                out[0, slice(None) if all_rows else -1, component] = value
            count += 1
            return out, torch.ones(out.shape[:2])

        with torch.inference_mode(), patch.object(self.model, "run_head", predicted):
            result = self.model.match(
                self.data,
                self.data,
                sparse=True,
                gemm_nn=gemm,
                exact_nn=exact,
                max_distance_px=1.0,
            )
        self.assertEqual(count, 2)
        self.assertFalse(result[f"valid{direction}"][0, -1])
        for i in (0, 1):
            torch.testing.assert_close(
                result[f"matches{i}"],
                torch.tensor([[-1, -1, -1]] if all_rows else [[0, 1, -1]]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                result[f"matching_scores{i}"],
                torch.tensor([[0.0, 0.0, 0.0]] if all_rows else [[1.0, 1.0, 0.0]]),
                rtol=0,
                atol=0,
            )

    def test_rejected_nonfinite_predictions_preserve_finite_matches(self):
        for gemm, exact in ((True, False), (True, True), (False, False)):
            for direction in (0, 1):
                for component in (0, 1):
                    for value in (float("nan"), float("inf"), -float("inf")):
                        with self.subTest(
                            gemm=gemm,
                            exact=exact,
                            direction=direction,
                            component=component,
                            value=value,
                        ):
                            self.run_case(gemm, exact, direction, component, value)

    def test_all_nonfinite_predictions_leave_every_row_unmatched(self):
        for gemm, exact in ((True, False), (True, True), (False, False)):
            for direction in (0, 1):
                with self.subTest(gemm=gemm, exact=exact, direction=direction):
                    self.run_case(gemm, exact, direction, 0, float("nan"), True)

    def test_finite_rejected_predictions_retain_policy(self):
        # Out-of-border finite predictions remain subject to the existing NN
        # policy. This patch contains nonfinite arithmetic only.
        for gemm, exact in ((True, False), (True, True), (False, False)):
            for direction in (0, 1):
                with self.subTest(gemm=gemm, exact=exact, direction=direction):
                    self.run_case(gemm, exact, direction, 0, 2.0)

    def test_empty_keypoint_sets_are_unmatched_with_zero_gradients(self):
        model = FlexGlue(**{**vars(self.model.conf), "n_layers": 1}).eval()
        for batch in (1, 2):
            data = {
                key: value.expand(batch, *value.shape[1:]).clone()
                for key, value in self.data.items()
            }
            for counts in ((0, 3), (3, 0), (0, 0)):
                inputs = [
                    {**data, "keypoints": data["keypoints"][:, :count]}
                    for count in counts
                ]
                originals = [
                    {key: value.clone() for key, value in item.items()}
                    for item in inputs
                ]
                for gemm, exact in ((True, False), (True, True), (False, False)):
                    with self.subTest(
                        batch=batch, counts=counts, gemm=gemm, exact=exact
                    ):
                        model.zero_grad(set_to_none=True)
                        result = model.match(
                            *inputs, sparse=True, gemm_nn=gemm, exact_nn=exact
                        )
                        for i, count in enumerate(counts):
                            confidence = result[f"confidence{i}"]
                            matches = result[f"matches{i}"]
                            scores = result[f"matching_scores{i}"]
                            self.assertEqual(matches.shape, (batch, count))
                            self.assertEqual(scores.shape, (batch, count))
                            self.assertEqual(matches.dtype, torch.long)
                            self.assertEqual(scores.dtype, confidence.dtype)
                            self.assertEqual(matches.device, confidence.device)
                            self.assertEqual(scores.device, confidence.device)
                            self.assertTrue((matches == -1).all())
                            self.assertEqual(scores.count_nonzero(), 0)
                            if count:
                                self.assertTrue((confidence > 0).any())
                            for key, original in originals[i].items():
                                torch.testing.assert_close(inputs[i][key], original)
                        self.assertIsNot(result["matches0"], result["matches1"])
                        self.assertIsNot(
                            result["matching_scores0"], result["matching_scores1"]
                        )
                        sum(
                            result[f"matching_scores{i}"].sum() for i in (0, 1)
                        ).backward()
                        self.assertIsNotNone(model.heads[1].matchability.bias.grad)
                        gradients = [
                            p.grad for p in model.parameters() if p.grad is not None
                        ]
                        for gradient in gradients:
                            self.assertTrue(gradient.isfinite().all())
                            self.assertEqual(gradient.count_nonzero(), 0)
                        for i in (0, 1):
                            confidence = result[f"confidence{i}"].clone()
                            with torch.no_grad():
                                result[f"matching_scores{i}"].fill_(1)
                            torch.testing.assert_close(
                                result[f"confidence{i}"], confidence
                            )

    def test_rejected_zero_confidence_has_finite_model_gradients(self):
        model = FlexGlue(**{**vars(self.model.conf), "n_layers": 1}).eval()
        with torch.no_grad():
            model.heads[1].matchability.weight.zero_()
            model.heads[1].matchability.bias.fill_(-100)
        result = model.match(self.data, self.data, sparse=True, max_distance_px=0)
        for direction in (0, 1):
            for field in ("confidence", "matches", "matching_scores"):
                self.assertEqual(result[f"{field}{direction}"].shape, (1, 3))
            self.assertEqual(result[f"confidence{direction}"].count_nonzero(), 0)
            self.assertTrue((result[f"matches{direction}"] == -1).all())
            self.assertEqual(result[f"matching_scores{direction}"].count_nonzero(), 0)
        sum(result[f"matching_scores{i}"].sum() for i in (0, 1)).backward()
        self.assertIsNotNone(model.heads[1].matchability.bias.grad)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        for gradient in gradients:
            self.assertTrue(gradient.isfinite().all())
            self.assertEqual(gradient.count_nonzero(), 0)

    def test_sparse_confidence_gradients_ignore_rejected_zero_products(self):
        for gemm, exact in ((True, False), (True, True), (False, False)):
            with self.subTest(gemm=gemm, exact=exact):
                confidence0 = torch.tensor([[0.0, 0.16, 0.36]], requires_grad=True)
                confidence1 = torch.tensor([[0.25, 0.0, 0.49]], requires_grad=True)
                pred = {
                    "p2d0_i1": self.data["keypoints"],
                    "p2d1_i0": self.data["keypoints"],
                    "confidence0": confidence0,
                    "confidence1": confidence1,
                    "valid0": torch.ones(1, 3, dtype=torch.bool),
                    "valid1": torch.ones(1, 3, dtype=torch.bool),
                }
                conf = self.model.resolve_conf(
                    {"gemm_nn": gemm, "exact_nn": exact, "max_distance_px": 1.0}
                )
                result = self.model.sparse_matches(pred, self.data, self.data, conf)
                for direction in (0, 1):
                    torch.testing.assert_close(
                        result[f"matches{direction}"], torch.tensor([[-1, -1, 2]])
                    )
                    expected = (confidence0.detach() * confidence1.detach()).sqrt()
                    torch.testing.assert_close(
                        result[f"matching_scores{direction}"], expected, rtol=0, atol=0
                    )
                loss = sum(result[f"matching_scores{i}"].sum() for i in (0, 1))
                gradients = torch.autograd.grad(loss, (confidence0, confidence1))
                for gradient, expected in zip(
                    gradients, ([[0.0, 0.0, 7 / 6]], [[0.0, 0.0, 6 / 7]])
                ):
                    torch.testing.assert_close(
                        gradient, gradient.new_tensor(expected), rtol=1e-6, atol=0
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
