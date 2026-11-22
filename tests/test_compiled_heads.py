import unittest
from unittest.mock import patch

import torch
from torch import nn

from flexglue import FlexGlue


class CompiledHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
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
        self.features = torch.ones(1, 8, 2, 3)
        self.points = torch.ones(1, 3, 2) * 0.1
        self.head = self.model.heads[0]

    def run_head(self, points=None):
        return self.model.run_head(
            0, self.points if points is None else points, self.features, self.features
        )

    def compile_mock(self, lengths):
        with patch("torch.compile", side_effect=lambda fn, **kwargs: fn) as compile_fn:
            self.model.compile(static_lengths=lengths)
        self.assertIs(compile_fn.call_args.args[0], self.head.__call__.__func__)
        self.assertIs(self.head.compiled_forward.__self__, self.head)

    def test_bucket_selection_and_fallback(self):
        expected = self.run_head()
        self.compile_mock(())
        with patch.object(self.head, "compiled_forward", side_effect=AssertionError):
            actual = self.run_head()
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
        self.compile_mock((2,))
        with patch.object(self.head, "compiled_forward", side_effect=AssertionError):
            actual = self.run_head()
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
        self.compile_mock((8, 4))
        with patch.object(
            self.head, "compiled_forward", wraps=self.head.compiled_forward
        ) as run:
            actual = self.run_head()
        self.assertEqual(run.call_args.args[0].shape[1], 4)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_hooks_added_removed_after_warmup(self):
        self.compile_mock((4,))
        baseline = self.run_head()
        seen = []

        def pre(module, args):
            seen.append(args[0].shape[1])
            return (args[0] * 0, *args[1:])

        def post(module, args, result):
            seen.append(args[0].shape[1])
            return result[0], torch.zeros_like(result[1])

        p = self.head.register_forward_pre_hook(pre)
        h = self.head.register_forward_hook(post)
        try:
            with patch.object(
                self.head, "compiled_forward", side_effect=AssertionError
            ):
                result = self.run_head()
            self.assertEqual(seen, [3, 3])
            self.assertTrue(torch.equal(result[1], torch.zeros_like(result[1])))
            expected = self.head(
                torch.zeros_like(self.points), self.features, self.features
            )
            torch.testing.assert_close(result[0], expected[0])
        finally:
            p.remove()
            h.remove()
        for a, b in zip(self.run_head(), baseline):
            torch.testing.assert_close(a, b)

    def test_child_and_global_hooks_see_unpadded_queries(self):
        self.compile_mock((4,))
        sizes = []

        def child_hook(module, args):
            sizes.append(args[0].shape[1])

        handle = self.head.to_search.register_forward_pre_hook(child_hook)
        try:
            with patch.object(
                self.head, "compiled_forward", side_effect=AssertionError
            ):
                self.run_head()
            self.assertEqual(sizes, [6, 3])
        finally:
            handle.remove()

        def global_hook(module, args):
            if module is self.head:
                sizes.append(args[0].shape[1])

        handle = nn.modules.module.register_module_forward_pre_hook(global_hook)
        try:
            with patch.object(
                self.head, "compiled_forward", side_effect=AssertionError
            ):
                self.run_head()
            self.assertEqual(sizes, [6, 3, 3])
        finally:
            handle.remove()

    def test_backward_hooks_observe_unpadded_queries(self):
        # Global backward hooks wrap every module output; PyTorch disallows
        # inplace ReLU on those views even in eager execution.
        for module in self.head.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.compile_mock((4,))
        for global_hook in (False, True):
            seen = []

            def backward(module, inputs, outputs):
                if module is self.head:
                    seen.append(inputs[0].shape[1])

            handle = (
                nn.modules.module.register_module_full_backward_hook(backward)
                if global_hook
                else self.head.register_full_backward_hook(backward)
            )
            try:
                points = self.points.clone().requires_grad_()
                with patch.object(
                    self.head, "compiled_forward", side_effect=AssertionError
                ):
                    output = self.run_head(points)
                sum(value.sum() for value in output).backward()
                self.assertEqual(seen, [3])
                self.assertIsNotNone(points.grad)
            finally:
                handle.remove()

    def test_empty_queries_use_eager(self):
        self.compile_mock((4,))
        with patch.object(self.head, "compiled_forward", side_effect=AssertionError):
            points, confidence = self.run_head(torch.empty(1, 0, 2))
        self.assertEqual(points.shape, (1, 0, 2))
        self.assertEqual(confidence.shape, (1, 0))

    def test_graph_output_ownership(self):
        self.compile_mock((4,))
        reused = [torch.zeros(1, 4, 2), torch.zeros(1, 4)]

        def replay(points, *features):
            reused[0].fill_(points[0, 0, 0])
            reused[1].fill_(points[0, 0, 0])
            return reused

        with patch.object(self.head, "compiled_forward", side_effect=replay):
            first = self.run_head(torch.ones(1, 3, 2))
            self.run_head(torch.ones(1, 3, 2) * 2)
        self.assertTrue(torch.all(first[0] == 1))
        self.assertTrue(torch.all(first[1] == 1))

    def test_sparse_empty_points(self):
        for n0, n1 in ((0, 3), (3, 0), (0, 0)):
            for gemm in (False, True):
                with self.subTest(n0=n0, n1=n1, gemm=gemm), torch.no_grad():
                    data = [
                        {
                            "features": torch.rand(1, 4, 2, 3),
                            "image_size": torch.tensor([[24.0, 16.0]]),
                            "keypoints": torch.rand(1, n, 2) * 10 + 1,
                        }
                        for n in (n0, n1)
                    ]
                    result = self.model.match(*data, sparse=True, gemm_nn=gemm)
                    for i, n in enumerate((n0, n1)):
                        self.assertEqual(result[f"matches{i}"].shape, (1, n))
                        self.assertTrue(torch.all(result[f"matches{i}"] == -1))
                        self.assertTrue(torch.all(result[f"matching_scores{i}"] == 0))


if __name__ == "__main__":
    unittest.main()
