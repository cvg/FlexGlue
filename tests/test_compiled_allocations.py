"""Generated CPU contracts for compiled-path allocation candidates."""
import unittest
from unittest.mock import patch

import torch

from flexglue import FlexGlue


def matcher():
    return FlexGlue(
        weights=None,
        describe=False,
        input_dim=4,
        descriptor_dim=8,
        search_dim=4,
        num_heads=2,
        n_layers=0,
        mp=None,
    ).eval()


class CompiledAllocationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        self.model = matcher()
        self.features = torch.randn(1, 8, 2, 3)
        # Exercise the same dispatch without pretending an identity compiler is CUDA.
        with patch("torch.compile", side_effect=lambda fn, **kw: fn):
            self.model.compile(static_lengths=(4,))

    def test_compiled_buckets_own_queries_and_preserve_outputs(self):
        head = self.model.heads[0]
        real = head.compiled_forward
        for kind in ("exact", "strided", "padded"):
            with self.subTest(kind=kind):
                if kind == "exact":
                    points = torch.rand(1, 4, 2)
                elif kind == "strided":
                    points = torch.rand(1, 4, 4)[..., ::2]
                else:
                    points = torch.rand(1, 3, 2)
                with torch.no_grad():
                    # Compare the same static bucket computation as run_head,
                    # then slice to the original query count. A shorter head
                    # call can select different kernels and rounding.
                    padded = torch.nn.functional.pad(
                        points, (0, 0, 0, 4 - points.shape[1])
                    )
                    expected = tuple(
                        value[:, : points.shape[1]]
                        for value in head(padded, self.features, self.features)
                    )
                    with patch.object(head, "compiled_forward", wraps=real) as call:
                        actual = self.model.run_head(
                            0, points, self.features, self.features
                        )
                used = call.call_args.args[0]
                self.assertTrue(used.is_contiguous())
                self.assertEqual(used.shape[1], 4)
                self.assertNotEqual(used.data_ptr(), points.data_ptr())
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_custom_compiled_head_can_mutate_its_exact_bucket(self):
        def shifted(_head, points, _cf_i, _cf_j):
            points.add_(1)
            return points, points[..., 0]

        model = matcher()
        head = model.heads[0]
        head.forward = shifted.__get__(head)
        real_compile = torch.compile
        with patch(
            "torch.compile",
            side_effect=lambda fn, **kw: real_compile(
                fn, backend="eager", fullgraph=kw["fullgraph"]
            ),
        ):
            model.compile(static_lengths=(4,))
        queries = torch.zeros(1, 4, 2)
        with torch.no_grad():
            positions, confidence = model.run_head(
                0, queries, self.features, self.features
            )
        torch.testing.assert_close(queries, torch.zeros_like(queries), rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.ones_like(positions), rtol=0, atol=0
        )
        torch.testing.assert_close(
            confidence, torch.ones_like(confidence), rtol=0, atol=0
        )

    def test_exact_bucket_preserves_input_and_parameter_gradients(self):
        head = self.model.heads[0]
        p = torch.rand(1, 4, 2, requires_grad=True)
        a = self.features.detach().clone().requires_grad_()
        b = self.features.detach().clone().requires_grad_()
        params = tuple(head.parameters())
        direct = head(p, a, b)
        expected = torch.autograd.grad(
            sum(x.sum() for x in direct), (p, a, b, *params), allow_unused=True
        )
        actual = self.model.run_head(0, p, a, b)
        observed = torch.autograd.grad(
            sum(x.sum() for x in actual), (p, a, b, *params), allow_unused=True
        )
        for x, y in zip(observed, expected):
            self.assertEqual(x is None, y is None)
            if x is not None:
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_real_dynamo_eager_fullgraph_exact_bucket(self):
        model = matcher()
        real_compile = torch.compile
        with patch(
            "torch.compile",
            side_effect=lambda fn, **kw: real_compile(
                fn, backend="eager", fullgraph=kw["fullgraph"]
            ),
        ):
            model.compile(static_lengths=(4,))
        points = torch.rand(1, 4, 2)
        with torch.no_grad():
            expected = model.heads[0](points, self.features, self.features)
            actual = model.run_head(0, points, self.features, self.features)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_existing_hook_empty_oversize_and_output_ownership(self):
        head = self.model.heads[0]
        seen = []
        handle = head.register_forward_pre_hook(
            lambda _m, args: seen.append(args[0].shape[1])
        )
        try:
            with patch.object(head, "compiled_forward", side_effect=AssertionError):
                self.model.run_head(
                    0, torch.rand(1, 4, 2), self.features, self.features
                )
            self.assertEqual(seen, [4])
        finally:
            handle.remove()
        for n in (0, 5):
            with self.subTest(n=n), patch.object(
                head, "compiled_forward", side_effect=AssertionError
            ):
                self.model.run_head(
                    0, torch.rand(1, n, 2), self.features, self.features
                )
        reused = [torch.zeros(1, 4, 2), torch.zeros(1, 4)]

        def replay(points, *_):
            reused[0].fill_(points[0, 0, 0])
            reused[1].fill_(points[0, 0, 0])
            return reused

        with patch.object(head, "compiled_forward", side_effect=replay):
            first = self.model.run_head(
                0, torch.ones(1, 4, 2), self.features, self.features
            )
            self.model.run_head(
                0, torch.full((1, 4, 2), 2.0), self.features, self.features
            )
        self.assertTrue(all(torch.equal(x, torch.ones_like(x)) for x in first))

    def test_pair_context_conversion_gradients_and_ownership(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for static_lengths in (None, (), (4,)):
                compiled = static_lengths is not None
                for strided in (False, True):
                    with self.subTest(
                        dtype=dtype, static_lengths=static_lengths, strided=strided
                    ):
                        self.model.static_lengths = static_lengths
                        width = 16 if strided else 8
                        originals = [
                            torch.randn(1, 6, width, dtype=dtype) for _ in range(2)
                        ]
                        values = [
                            (x[..., ::2] if strided else x).detach().requires_grad_()
                            for x in originals
                        ]
                        old = [
                            x.float().clone() if compiled else x.float() for x in values
                        ]
                        context = self.model.pair_context(
                            *values,
                            ((2, 3), (2, 3)),
                            0,
                            covisibility_score=torch.ones(1),
                        )
                        got = list(context.cf)
                        expected = [x.mT.unflatten(-1, (2, 3)) for x in old]
                        for a, b in zip(got, expected):
                            self.assertEqual(a.dtype, torch.float32)
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        for x, y in zip(values, got):
                            if compiled or dtype != torch.float32:
                                self.assertNotEqual(x.data_ptr(), y.data_ptr())
                            else:
                                self.assertEqual(x.data_ptr(), y.data_ptr())
                        grad_old = torch.autograd.grad(
                            sum(x.square().sum() for x in expected),
                            values,
                            retain_graph=True,
                        )
                        grad_new = torch.autograd.grad(
                            sum(x.square().sum() for x in got), values
                        )
                        for a, b in zip(grad_new, grad_old):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        if compiled:
                            saved = [x.clone() for x in got]
                            with torch.no_grad():
                                for x in values:
                                    x.add_(1)
                            for a, b in zip(got, saved):
                                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
