"""Opt-in Inductor/CUDA ownership checks; no checkpoints or downloads."""
import copy
import os
import unittest

import torch
from test_compiled_ownership import make_matcher, make_refiner

from flexglue.refiner import VGG19BN, RoMav2Refiner
from flexglue.utils.modules import TransformerLayer


@unittest.skipUnless(
    os.environ.get("FLEXGLUE_TEST_COMPILED_CUDA") == "1" and torch.cuda.is_available(),
    "set FLEXGLUE_TEST_COMPILED_CUDA=1 on a CUDA runtime",
)
class CudaCompiledOwnershipTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(901)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_head_copy_reload_and_retained_outputs(self):
        model = make_matcher().cuda().compile(mode="default", static_lengths=(4,))
        features = torch.randn(1, 8, 2, 2, device="cuda")
        points = torch.rand(1, 4, 2, device="cuda") * 2 - 1
        with torch.inference_mode():
            original = model.run_head(1, points, features, features)[0]
            saved = original.clone()
            copied = copy.deepcopy(model)
            copied.heads[1].offset_head.head[-1].bias[:17].add_(
                torch.arange(17, device="cuda")
            )
            expected = copied.heads[1](points, features, features)[0]
            actual = copied.run_head(1, points, features, features)[0]
            torch.testing.assert_close(actual, expected)
            self.assertFalse(torch.allclose(actual, saved))
            torch.testing.assert_close(original, saved, rtol=0, atol=0)

    def test_transformer_copy_follows_dtype_and_parameters(self):
        model = make_matcher().cuda().compile(mode="default", static_lengths=(4,))
        copied = copy.deepcopy(model).double()
        features = torch.randn(1, 8, 2, 2, device="cuda")
        args = tuple(x.double() for x in model.encode_pair(features, features))
        with torch.inference_mode():
            copied.transformers[0].cross_attn.ffn[-1].bias.add_(2)
            expected = TransformerLayer.forward(copied.transformers[0], *args)
            actual = copied.transformers[0](*args)
            for a, e in zip(actual, expected):
                self.assertEqual(a.dtype, torch.float64)
                torch.testing.assert_close(a, e)

    def test_refiner_siblings_use_the_copied_owner(self):
        model = make_matcher().cuda()
        model.refiner = make_refiner().cuda()
        model.compile(mode="default", static_lengths=(4,))
        copied = copy.deepcopy(model)
        image = torch.rand(1, 3, 8, 8, device="cuda")
        fine = {1: torch.rand(1, 8, 8, 64, device="cuda")}
        warp = torch.rand(1, 2, 2, 2, device="cuda") * 2 - 1
        logits = torch.rand(1, 2, 2, device="cuda")
        with torch.inference_mode():
            copied.refiner.features.layers[4].bias.add_(5)
            copied.refiner.refiners["1"].warp_head.bias[0].add_(4)
            # Compile explicit-owner references with the same backend so this
            # tests ownership independently of eager/Inductor rounding changes.
            features_reference = torch.compile(VGG19BN.forward, fullgraph=True)
            expected_features = features_reference(copied.refiner.features, image)
            actual_features = copied.refiner.extract(image, resize=(8, 8))
            for scale in expected_features:
                torch.testing.assert_close(
                    actual_features[scale], expected_features[scale]
                )
            refiner_reference = torch.compile(RoMav2Refiner.forward, fullgraph=True)
            expected = refiner_reference(copied.refiner, fine, fine, warp, logits)
            actual = copied.refiner(fine, fine, warp, logits)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e)


if __name__ == "__main__":
    unittest.main()
