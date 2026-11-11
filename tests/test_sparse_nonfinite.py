"""Public matcher regression: rejected geometry must not steal finite matches.

Use PYTHONPATH=<exact source/worktree> python this_file.py. No checkpoint needed.
Only the learned prediction boundary is replaced with deterministic outputs;
describe_inputs, decode_pair, normalization, gating, NN selection, and packing run.
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
