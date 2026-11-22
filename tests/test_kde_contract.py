import math
import unittest
from unittest.mock import patch

import torch

from flexglue.utils import ops


class KDEContractTest(unittest.TestCase):
    def test_invalid_bandwidth_is_rejected_before_tensor_indexing(self):
        points = torch.zeros(1, 1, 4)
        weights = torch.ones(1, 1)
        for std in (
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            1e-200,
            1e-6,
            1e20,
            1e150,
            1e200,
        ):
            with self.subTest(std=std), patch.object(torch, "cartesian_prod") as index:
                with self.assertRaises(ValueError):
                    ops.lattice_kde(points, weights, std)
                index.assert_not_called()

    def test_single_point_peak_for_each_coordinate_count(self):
        for dimension in (1, 2, 4):
            with self.subTest(dimension=dimension):
                actual = ops.lattice_kde(
                    torch.zeros(1, 1, dimension), torch.tensor([[2.0]]), 1.0
                )
                expected = 2.0 * math.sqrt(12.0 / 11.0) ** dimension
                torch.testing.assert_close(
                    actual, torch.tensor([[expected]]), rtol=1e-6, atol=1e-6
                )

    def test_default_float64_does_not_change_float32_kernel(self):
        points = torch.tensor([[[0.0, 0.0], [0.1, 0.2], [-0.3, 0.4]]])
        weights = torch.tensor([[1.0, 2.0, 0.5]])
        expected = ops.lattice_kde(points, weights, 0.3)
        prior = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            actual = ops.lattice_kde(points, weights, 0.3)
        finally:
            torch.set_default_dtype(prior)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_finite_large_bandwidth_remains_supported(self):
        actual = ops.lattice_kde(torch.zeros(1, 1, 2), torch.ones(1, 1), 1e10)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(
            actual, torch.tensor([[12.0 / 11.0]]), rtol=1e-6, atol=1e-6
        )

    def test_float32_rounding_neighbors_at_kernel_limit(self):
        center = torch.tensor(
            math.sqrt(torch.finfo(torch.float32).max) / 3, dtype=torch.float32
        )
        candidates = [center]
        for direction in (-float("inf"), float("inf")):
            value = center
            for _ in range(4):
                value = torch.nextafter(
                    value, torch.tensor(direction, dtype=torch.float32)
                )
                candidates.append(value)
        accepted, rejected = [], []
        for candidate in candidates:
            std = float(candidate)
            with self.subTest(std=std), patch.object(
                ops.F, "conv1d", wraps=ops.F.conv1d
            ) as convolution:
                try:
                    actual = ops.lattice_kde(
                        torch.zeros(1, 1, 2), torch.ones(1, 1), std
                    )
                except ValueError:
                    rejected.append(std)
                    convolution.assert_not_called()
                else:
                    accepted.append(std)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue((actual > 0).all())
                    self.assertEqual(convolution.call_count, 2)
                    for call in convolution.call_args_list:
                        self.assertTrue(torch.isfinite(call.args[1]).all())
        self.assertTrue(accepted)
        self.assertTrue(rejected)
        self.assertLess(max(accepted), min(rejected))


if __name__ == "__main__":
    unittest.main()
