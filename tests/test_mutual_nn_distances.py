import unittest
from unittest.mock import patch

import torch

from flexglue import FlexGlue
from flexglue.utils import ops


class MutualNearestDistancesTest(unittest.TestCase):
    def test_values_indices_ties_and_layouts(self):
        # Asymmetric, with a tie and a non-mutual row whose true min is not
        # at index zero. The ordinary reductions are the independent oracle.
        base = torch.tensor(
            [[[4.0, 1.0, 1.0], [5.0, 3.0, 2.0], [0.0, 8.0, 6.0], [2.0, 7.0, 9.0]]]
        )
        for dist in (base, base.transpose(1, 2), torch.stack((base, base), -1)[..., 0]):
            with self.subTest(shape=dist.shape, stride=dist.stride()):
                default = ops.mutual_nn(dist)
                m0, m1, d0, d1 = ops.mutual_nn(dist, return_distances=True)
                self.assertEqual(len(default), 2)
                torch.testing.assert_close(m0, default[0], rtol=0, atol=0)
                torch.testing.assert_close(m1, default[1], rtol=0, atol=0)
                torch.testing.assert_close(d0, dist.min(2).values, rtol=0, atol=0)
                torch.testing.assert_close(d1, dist.min(1).values, rtol=0, atol=0)

    def test_nan_eager_and_sanitized_values(self):
        dist = torch.tensor(
            [[[float("nan"), 2.0, float("inf")], [1.0, -float("inf"), 3.0]]]
        )
        for x in (
            dist,
            torch.nan_to_num(
                dist, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
            ),
        ):
            _, _, d0, d1 = ops.mutual_nn(x, return_distances=True)
            torch.testing.assert_close(d0, x.min(2).values, equal_nan=True)
            torch.testing.assert_close(d1, x.min(1).values, equal_nan=True)

    def test_gradient_matches_min_including_ties(self):
        x = torch.tensor([[[1.0, 1.0, 3.0], [2.0, 4.0, 2.0]]], requires_grad=True)
        _, _, d0, d1 = ops.mutual_nn(x, return_distances=True)
        (actual,) = torch.autograd.grad(d0.sum() + d1.sum(), x)
        (expected,) = torch.autograd.grad(
            x.min(2).values.sum() + x.min(1).values.sum(), x
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_public_non_gemm_matches_independent_reductions(self):
        torch.manual_seed(7)
        model = FlexGlue(
            weights=None,
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=0,
            mp=None,
        ).eval()
        data = {
            "features": torch.randn(1, 4, 2, 2),
            "image_size": torch.tensor([[16.0, 16.0]]),
            "keypoints": torch.tensor([[[3.0, 3.0], [8.0, 8.0], [12.0, 12.0]]]),
        }

        def predicted(layer, points, *features):
            return points.clone(), torch.ones(points.shape[:2])

        original = ops.mutual_nn

        def reference(dist, *, return_distances=False):
            pair = original(dist)
            if return_distances:
                return (*pair, dist.min(2).values, dist.min(1).values)
            return pair

        with torch.inference_mode(), patch.object(model, "run_head", predicted):
            actual = model.match(data, data, sparse=True, gemm_nn=False)
            with patch.object(ops, "mutual_nn", reference):
                expected = model.match(data, data, sparse=True, gemm_nn=False)
        for key in ("matches0", "matches1", "matching_scores0", "matching_scores1"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertTrue((actual["matches0"] >= 0).any())


if __name__ == "__main__":
    unittest.main()
