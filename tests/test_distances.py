import unittest

import torch

from flexglue.utils import ops


def reference(x, y):
    return sum(
        (x[..., i, None] - y[..., i][:, None]).square() for i in range(x.shape[-1])
    )


class SquaredDistanceTest(unittest.TestCase):
    def test_zero_coordinate_width_preserves_empty_sum(self):
        self.assertEqual(ops.sq_dist(torch.empty(1, 2, 0), torch.empty(1, 3, 0)), 0)

    def test_values_and_gradients(self):
        torch.manual_seed(3)
        for dtype in (torch.float32, torch.float64):
            for b, m, n, d in (
                (1, 5, 7, 1),
                (2, 13, 3, 4),
                (2, 0, 7, 4),
                (2, 3, 0, 4),
                (1, 7, 9, 6),
            ):
                with self.subTest(dtype=dtype, shape=(b, m, n, d)):
                    x = torch.randn(b, d, m, dtype=dtype).transpose(1, 2)
                    y = torch.randn(b, d, n, dtype=dtype).transpose(1, 2)
                    x.requires_grad_()
                    y.requires_grad_()
                    xx, yy = x.clone(), y.clone()
                    a, e = ops.sq_dist(x, y), reference(x, y)
                    torch.testing.assert_close(a, e, rtol=0, atol=0)
                    self.assertTrue(torch.equal(x, xx))
                    self.assertTrue(torch.equal(y, yy))
                    expected = torch.autograd.grad(e.sum(), (x, y))
                    actual = torch.autograd.grad(a.sum(), (x, y))
                    for av, ev in zip(actual, expected):
                        torch.testing.assert_close(av, ev, rtol=0, atol=0)

    def test_ties_and_special_values(self):
        x = torch.tensor(
            [[[0.0, -0.0, 1.0, 2.0], [0.0, 0.0, 1.0, 2.0], [1e10, -1e10, 3.0, 4.0]]]
        )
        y = x.clone()
        a, e = ops.sq_dist(x, y), reference(x, y)
        self.assertTrue(torch.equal(a, e))
        for av, ev in zip(ops.mutual_nn(a), ops.mutual_nn(e)):
            self.assertTrue(torch.equal(av, ev))
        for value in (float("inf"), float("nan")):
            x[0, 0, 0] = value
            torch.testing.assert_close(
                ops.sq_dist(x, y),
                reference(x, y),
                rtol=0,
                atol=0,
                equal_nan=True,
            )


if __name__ == "__main__":
    unittest.main()
