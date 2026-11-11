import unittest

import torch

from flexglue.utils import ops


def legacy_readout(logits, grid, hw, radius=1, k=512):
    # Frozen eager formula: center the full volume before gathering neighbors.
    if logits.shape[-1] > k:
        values, indices = logits.topk(k, dim=-1)
        maximum, index = values[..., :1], indices[..., :1]
        centered = (logits - maximum) / 2
        threshold = ops.sorted_entmax15_threshold((values - maximum) / 2)
    else:
        maximum, index = logits.max(-1, keepdim=True)
        centered = (logits - maximum) / 2
        threshold = ops.entmax15_threshold(centered, k)
    h, w = hw
    offsets = torch.arange(-radius, radius + 1, device=logits.device)
    y = (index // w)[..., None] + offsets[:, None]
    x = (index % w)[..., None] + offsets[None, :]
    inside = ((y >= 0) & (y < h) & (x >= 0) & (x < w)).flatten(-2)
    indices = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).flatten(-2)
    weights = (centered.gather(-1, indices) - threshold).clamp_min(0).square()
    position = (weights[..., None] * grid[indices]).sum(-2) / weights.sum(
        -1, keepdim=True
    )
    return position, (weights * inside).sum(-1)


class EntmaxReadoutTest(unittest.TestCase):
    def test_forward_matches_full_volume_reference(self):
        for dtype in (torch.float32, torch.float64):
            for batch, queries, h, w, radius, k in (
                (1, 0, 3, 5, 1, 8),
                (1, 1, 1, 1, 0, 8),
                (2, 7, 3, 5, 1, 8),
                (1, 9, 8, 9, 2, 16),
                (1, 5, 3, 7, 4, 32),
                (1, 3, 17, 31, 1, 512),
            ):
                with self.subTest(
                    dtype=dtype, shape=(batch, queries, h, w), radius=radius, k=k
                ):
                    torch.manual_seed(19)
                    logits = torch.rand(batch, h * w, queries, dtype=dtype).transpose(
                        1, 2
                    )
                    grid = ops.cell_centers(h, w, (w, h), device=logits.device).to(
                        dtype
                    )
                    actual = ops.entmax15_window_readout(
                        logits, grid, (h, w), radius, k
                    )
                    expected = legacy_readout(logits, grid, (h, w), radius, k)
                    for got, reference in zip(actual, expected):
                        self.assertTrue(torch.equal(got, reference))

    def test_ties_and_clamped_edges_preserve_selection(self):
        for k in (4, 32):
            logits = torch.zeros(1, 3, 15)
            grid = ops.cell_centers(3, 5, (5, 3), device=logits.device)
            for radius in (0, 1, 3):
                expected = legacy_readout(logits, grid, (3, 5), radius, k)
                actual = ops.entmax15_window_readout(logits, grid, (3, 5), radius, k)
                for got, reference in zip(actual, expected):
                    self.assertTrue(torch.equal(got, reference))

    def test_gradients_match_full_volume_reference(self):
        torch.manual_seed(29)
        grid = ops.cell_centers(8, 9, (9, 8), device="cpu").double()
        logits = torch.rand(2, 7, 72, dtype=torch.float64, requires_grad=True)
        actual = ops.entmax15_window_readout(logits, grid, (8, 9), k=32)
        expected = legacy_readout(logits, grid, (8, 9), k=32)
        got = torch.autograd.grad(sum(x.sum() for x in actual), logits)[0]
        reference = torch.autograd.grad(sum(x.sum() for x in expected), logits)[0]
        torch.testing.assert_close(got, reference, rtol=1e-10, atol=1e-12)

    def test_float32_edge_and_tie_gradients(self):
        grid = ops.cell_centers(3, 5, (5, 3), device="cpu")
        for k in (4, 32):
            for radius in (0, 2, 4):
                with self.subTest(k=k, radius=radius):
                    logits = torch.zeros(1, 3, 15, requires_grad=True)
                    actual = ops.entmax15_window_readout(
                        logits, grid, (3, 5), radius, k
                    )
                    expected = legacy_readout(logits, grid, (3, 5), radius, k)
                    got = torch.autograd.grad(sum(x.sum() for x in actual), logits)[0]
                    reference = torch.autograd.grad(
                        sum(x.sum() for x in expected), logits
                    )[0]
                    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


def legacy_low_precision_mass(logits, hw, k=512):
    # The original threshold accumulates and centers in the input dtype.
    values, indices = logits.topk(k, dim=-1)
    maximum, index = values[..., :1], indices[..., :1]
    scores = (values - maximum) / 2
    rho = torch.arange(1, k + 1, device=logits.device, dtype=logits.dtype)
    mean = scores.cumsum(-1) / rho
    mean_sq = scores.square().cumsum(-1) / rho
    tau = mean - ((1 - rho * (mean_sq - mean.square())) / rho).clamp_min(0).sqrt()
    tau = tau.gather(-1, (tau <= scores).sum(-1, keepdim=True) - 1)
    h, w = hw
    r = torch.arange(-1, 2, device=logits.device)
    y = (index // w)[..., None] + r[:, None]
    x = (index % w)[..., None] + r[None, :]
    inside = ((y >= 0) & (y < h) & (x >= 0) & (x < w)).flatten(-2)
    neighbor = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).flatten(-2)
    selected = ((logits - maximum) / 2).gather(-1, neighbor)
    return ((selected - tau).clamp_min(0).square() * inside).sum(-1)


def precision_fixture(device, dtype):
    torch.manual_seed(93)
    torch.randn(1, 3, 20)  # Preserve the previously observed CUDA fixture.
    logits = torch.randn(1, 3, 1024) * 0.1 - 5
    logits[0, 0, 0] = 5
    logits[0, 1, -1] = logits[0, 1, -2] = 5
    logits[0, 2, 512] = 5
    return logits.to(device=device, dtype=dtype)


class EntmaxPrecisionTest(unittest.TestCase):
    def test_half_and_bfloat16_against_fp32_accumulation(self):
        for dtype in (torch.float16, torch.bfloat16):
            for hw in ((4, 5), (32, 32)):
                for grid_dtype in (dtype, torch.float32):
                    with self.subTest(dtype=dtype, hw=hw, grid_dtype=grid_dtype):
                        if hw == (32, 32):
                            logits = precision_fixture("cpu", dtype)
                        else:
                            torch.manual_seed(15)
                            logits = (torch.randn(1, 3, 20) * 0.1 - 5).to(dtype)
                            logits[0, 0, 0] = 5
                            logits[0, 1, -1] = logits[0, 1, -2] = 5
                            logits[0, 2, 10] = 5
                        grid = ops.cell_centers(*hw, hw[::-1], device="cpu").to(
                            grid_dtype
                        )
                        actual = ops.entmax15_window_readout(logits, grid, hw)
                        reference = ops.entmax15_window_readout(
                            logits.float(), grid.float(), hw
                        )
                        self.assertEqual(
                            actual[0].dtype, torch.promote_types(dtype, grid_dtype)
                        )
                        self.assertEqual(actual[1].dtype, dtype)
                        for got, ref in zip(actual, reference):
                            self.assertTrue(torch.isfinite(got).all())
                            torch.testing.assert_close(
                                got.float(), ref, rtol=0.01, atol=0.004
                            )
                        self.assertTrue(((actual[1] >= 0) & (actual[1] <= 1.01)).all())

    def test_half_extreme_centering_stays_finite(self):
        logits = torch.tensor([[[65504.0, -65504.0]]], dtype=torch.float16)
        grid = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float16)
        actual = ops.entmax15_window_readout(logits, grid, (1, 2))
        reference = ops.entmax15_window_readout(logits.float(), grid.float(), (1, 2))
        for got, expected in zip(actual, reference):
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got.float(), expected)

    def test_bfloat16_legacy_threshold_has_real_failure(self):
        logits = precision_fixture("cpu", torch.bfloat16)
        grid = ops.cell_centers(32, 32, (32, 32), device="cpu").bfloat16()
        old = legacy_low_precision_mass(logits, (32, 32))
        fixed = ops.entmax15_window_readout(logits, grid, (32, 32))[1]
        self.assertGreater(old.max().item(), 1.05)
        self.assertLessEqual(fixed.max().item(), 1.01)

    def test_low_precision_gradients_against_fp32(self):
        for dtype in (torch.float16, torch.bfloat16):
            for hw in ((4, 5), (32, 32)):
                with self.subTest(dtype=dtype, hw=hw):
                    torch.manual_seed(91)
                    logits = torch.randn(1, 2, hw[0] * hw[1]).to(dtype).requires_grad_()
                    grid = ops.cell_centers(*hw, hw[::-1], device="cpu")
                    actual = ops.entmax15_window_readout(logits, grid, hw)
                    grad = torch.autograd.grad(
                        sum(x.float().sum() for x in actual), logits
                    )[0]
                    reference = logits.detach().float().requires_grad_()
                    out = ops.entmax15_window_readout(reference, grid, hw)
                    expected = torch.autograd.grad(
                        sum(x.sum() for x in out), reference
                    )[0]
                    self.assertTrue(torch.isfinite(grad).all())
                    torch.testing.assert_close(
                        grad.float(), expected, rtol=0.03, atol=0.004
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_original_fp16_cuda_failure_is_fixed(self):
        logits = precision_fixture("cuda", torch.float16)
        grid = ops.cell_centers(32, 32, (32, 32), device="cuda").half()
        old = legacy_low_precision_mass(logits, (32, 32))
        actual = ops.entmax15_window_readout(logits, grid, (32, 32))[1]
        reference = ops.entmax15_window_readout(logits.float(), grid.float(), (32, 32))[
            1
        ]
        self.assertGreater(old.max().item(), 1.05)
        torch.testing.assert_close(actual.float(), reference, rtol=0.001, atol=0.001)


if __name__ == "__main__":
    unittest.main()
