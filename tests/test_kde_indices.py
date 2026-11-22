import itertools
import math
import unittest
from unittest.mock import patch

import torch

from flexglue.utils.ops import lattice_kde


class KdeIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.threads)

    @staticmethod
    def reference_indices(points, std):
        import math

        pitch = std / 2
        size = math.ceil(2 / pitch) + 2
        u = ((points.float() + 1) / pitch).clamp(0, size - 1 - 1e-4)
        d = points.shape[-1]
        rows = []
        for batch in u.floor().long().tolist():
            rows.append(
                [
                    sum(
                        (min(cell[j], size - 2) + corner[j]) * size ** (d - 1 - j)
                        for j in range(d)
                    )
                    for cell in batch
                    for corner in itertools.product((0, 1), repeat=d)
                ]
            )
        return torch.tensor(rows, dtype=torch.long, device=points.device)

    def test_corner_addresses_match_python_integer_oracle(self):
        original = torch.Tensor.scatter_add_
        for d in (1, 2, 3, 4):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(dimensions=d, dtype=dtype):
                    points = torch.linspace(-1.1, 1.1, 24 * d, dtype=dtype)
                    points = points.reshape(2, 6, d * 2)[..., ::2]
                    expected = self.reference_indices(points, 0.2)
                    observed = []

                    def check(tensor, dim, index, source):
                        observed.append(index)
                        self.assertTrue(torch.equal(index, expected))
                        return original(tensor, dim, index, source)

                    with patch.object(torch.Tensor, "scatter_add_", check):
                        value = lattice_kde(points, torch.ones(2, 6), 0.2)
                    self.assertEqual(len(observed), 1)
                    self.assertEqual(value.shape, (2, 6))

    def test_extreme_bandwidth_upper_cell(self):
        # Stop at scatter: validate the real addresses without blurring 64 MiB.
        std = 4 / (2**24 + 1)
        for dtype in (torch.float32, torch.float64):
            points = torch.tensor([[-1.0, 0.0, 1.0, 0.9999999]], dtype=dtype)
            points = torch.stack((points, points), -1)[..., :1]  # strided
            expected = self.reference_indices(points, std)
            observed = []

            class Observed(Exception):
                pass

            def check(tensor, dim, index, source):
                observed.append(index.clone())
                self.assertTrue(torch.equal(index, expected))
                self.assertLess(index.max().item(), tensor.shape[dim])
                # The upper endpoint belongs entirely to the last lattice cell.
                self.assertEqual(source[0, 4:6].tolist(), [0.0, 1.0])
                raise Observed

            with patch.object(torch.Tensor, "scatter_add_", check):
                with self.assertRaises(Observed):
                    lattice_kde(points, torch.ones(1, 4), std)
            self.assertEqual(len(observed), 1)

    def test_rounded_int64_boundary_without_allocating_the_lattice(self):
        std = 4 / (2**63 - 2**20)
        size = math.ceil(4 / std) + 2
        self.assertLessEqual(size, torch.iinfo(torch.int64).max)
        owner = self

        class Observed(Exception):
            pass

        class Lattice:
            def scatter_add_(self, dim, index, source):
                owner.assertEqual(index.tolist(), [[size - 2, size - 1]])
                owner.assertEqual(source.tolist(), [[0.0, 1.0]])
                raise Observed

        def allocate(tensor, *shape):
            owner.assertEqual(shape, (1, size))
            return Lattice()

        with patch.object(torch.Tensor, "new_zeros", allocate):
            with self.assertRaises(Observed):
                lattice_kde(torch.ones(1, 1, 1), torch.ones(1, 1), std)

    def test_values_and_gradients_match_reference_addressing(self):
        original = torch.Tensor.scatter_add_
        for d in (1, 2, 4):
            with self.subTest(dimensions=d):
                generator = torch.Generator().manual_seed(900 + d)
                points = torch.rand(1, 7, d, generator=generator) * 1.8 - 0.9
                weights = torch.rand(1, 7, generator=generator)
                expected = self.reference_indices(points, 0.2)
                inputs = [
                    [x.clone().requires_grad_() for x in (points, weights)]
                    for _ in range(2)
                ]

                def reference(tensor, dim, index, source):
                    return original(tensor, dim, expected, source)

                actual = lattice_kde(*inputs[0], 0.2)
                with patch.object(torch.Tensor, "scatter_add_", reference):
                    wanted = lattice_kde(*inputs[1], 0.2)
                self.assertTrue(torch.equal(actual, wanted))
                for output in (actual, wanted):
                    output.square().sum().backward()
                for a, b in zip(*inputs):
                    self.assertTrue(torch.equal(a.grad, b.grad))

    def test_corner_weights_keep_native_values_and_gradients(self):
        scatter_add = torch.Tensor.scatter_add_
        for d in (1, 2, 4):
            with self.subTest(dimensions=d):
                rng = torch.Generator().manual_seed(194 + d)
                points = torch.rand(1, 64, d, generator=rng) * 1.8 - 0.9
                weights = torch.rand(1, 64, generator=rng)
                points[0, :2] = torch.tensor([[-1.0] * d, [1.0] * d])
                weights[0, 0] = 0
                inputs = tuple(x.requires_grad_() for x in (points, weights))
                observed = []

                def capture(tensor, dim, index, source):
                    observed.append(source)
                    return scatter_add(tensor, dim, index, source)

                with patch.object(torch.Tensor, "scatter_add_", capture):
                    lattice_kde(*inputs, 0.3)
                (actual,) = observed

                pitch = 0.3 / 2
                size = math.ceil(2 / pitch) + 2
                frac = ((points + 1) / pitch).clamp(0, size - 1 - 1e-4).frac()
                corners = torch.tensor(list(itertools.product((False, True), repeat=d)))
                tent = torch.where(corners, frac[:, :, None], 1 - frac[:, :, None])
                expected = (tent.prod(-1) * weights[..., None]).flatten(1)
                self.assertTrue(torch.equal(actual, expected))

                seed = torch.rand(actual.shape, generator=rng) * 2 - 1
                actual_grad = torch.autograd.grad(actual, inputs, seed)
                expected_grad = torch.autograd.grad(expected, inputs, seed)
                for got, wanted in zip(actual_grad, expected_grad):
                    self.assertTrue(torch.equal(got, wanted))

    def test_blur_matches_independent_lines_at_broad_bandwidth(self):
        rng = torch.Generator().manual_seed(200)
        points = torch.rand(2, 64, 4, generator=rng) * 1.8 - 0.9
        weights = torch.rand(2, 64, generator=rng) + 0.1
        cotangent = torch.rand(2, 64, generator=rng) * 2 - 1
        conv1d = torch.nn.functional.conv1d

        def independent_lines(data, kernel, padding, groups=1):
            size = data.shape[-1]
            return conv1d(
                data.reshape(-1, 1, size), kernel[:1], padding=padding
            ).reshape_as(data)

        results = []
        for use_reference in (False, True):
            inputs = tuple(x.clone().requires_grad_() for x in (points, weights))
            convolution = independent_lines if use_reference else conv1d
            with patch.object(torch.nn.functional, "conv1d", convolution):
                value = lattice_kde(*inputs, 2.0)
                results.append((value, *torch.autograd.grad(value, inputs, cotangent)))
        for actual, expected in zip(*results):
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
