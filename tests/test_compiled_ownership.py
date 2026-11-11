"""Real Dynamo CPU ownership checks, not Inductor/CUDA performance claims."""

import copy
import unittest
from unittest.mock import patch

import torch

from flexglue import FlexGlue
from flexglue.refiner import VGG19BN, ConvRefiner, RoMav2Refiner
from flexglue.utils.modules import TransformerLayer


def make_matcher():
    return FlexGlue(
        weights=None,
        describe=False,
        input_dim=8,
        descriptor_dim=8,
        search_dim=4,
        num_heads=2,
        n_layers=1,
        mp=None,
    ).eval()


def make_refiner():
    features = VGG19BN()
    stage = ConvRefiner(*RoMav2Refiner.configs[1])
    state = {
        **{"refiner_features." + k: v for k, v in features.state_dict().items()},
        **{"refiners.1." + k: v for k, v in stage.state_dict().items()},
    }
    with patch("flexglue.refiner.ops.load_state_dict", return_value=state):
        return RoMav2Refiner(scales=(1,), weights="generated").eval()


def compile_for_cpu(model):
    real_compile = torch.compile
    with patch(
        "torch.compile",
        side_effect=lambda fn, **kw: real_compile(
            fn, backend="eager", fullgraph=kw["fullgraph"]
        ),
    ):
        model.compile(static_lengths=(4,))
    return model


class CompiledOwnershipTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(20261109)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_copied_head_uses_loaded_copy_weights(self):
        model = compile_for_cpu(make_matcher())
        features = torch.randn(1, 8, 2, 2)
        points = torch.rand(1, 4, 2) * 2 - 1
        with torch.no_grad():
            original = model.run_head(1, points, features, features)[0].clone()
            copied = copy.deepcopy(model)
            state = copied.heads[1].state_dict()
            last = len(copied.heads[1].offset_head.head) - 1
            key = f"offset_head.head.{last}.bias"
            state[key] = state[key].clone()
            state[key][:17] += torch.arange(17)
            copied.heads[1].load_state_dict(state)
            expected = copied.heads[1](points, features, features)[0]
            self.assertFalse(torch.equal(expected, original))
            actual = copied.run_head(1, points, features, features)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                model.run_head(1, points, features, features)[0],
                original,
                rtol=0,
                atol=0,
            )

    def test_copied_transformer_uses_copy_parameters(self):
        model = compile_for_cpu(make_matcher())
        copied = copy.deepcopy(model)
        features = torch.randn(1, 8, 2, 2)
        args = model.encode_pair(features, features)
        with torch.no_grad():
            before = model.transformers[0](*args)
            copied.transformers[0].cross_attn.ffn[-1].bias.add_(2)
            expected = TransformerLayer.forward(copied.transformers[0], *args)
            self.assertFalse(torch.equal(before[0], expected[0]))
            actual = copied.transformers[0](*args)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_copied_transformer_follows_dtype_change(self):
        model = compile_for_cpu(make_matcher())
        copied = copy.deepcopy(model).double()
        features = torch.randn(1, 8, 2, 2)
        args = tuple(x.double() for x in model.encode_pair(features, features))
        with torch.no_grad():
            expected = TransformerLayer.forward(copied.transformers[0], *args)
            actual = copied.transformers[0](*args)
        for a, e in zip(actual, expected):
            self.assertEqual(a.dtype, torch.float64)
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_copied_refiner_features_use_copy_parameters(self):
        model = make_matcher()
        model.refiner = make_refiner()
        copied = copy.deepcopy(compile_for_cpu(model))
        image = torch.rand(1, 3, 8, 8)
        with torch.no_grad():
            copied.refiner.features.layers[4].bias.add_(5)
            expected = VGG19BN.forward(copied.refiner.features, image)
            actual = copied.refiner.extract(image, resize=(8, 8))
            for scale in expected:
                torch.testing.assert_close(
                    actual[scale], expected[scale], rtol=0, atol=0
                )

    def test_copied_refiner_uses_copy_parameters(self):
        model = make_matcher()
        model.refiner = make_refiner()
        copied = copy.deepcopy(compile_for_cpu(model))
        fine = {1: torch.rand(1, 8, 8, 64)}
        warp = torch.rand(1, 2, 2, 2) * 2 - 1
        logits = torch.rand(1, 2, 2)
        with torch.no_grad():
            copied.refiner.refiners["1"].warp_head.bias[0].add_(4)
            expected = RoMav2Refiner.forward(copied.refiner, fine, fine, warp, logits)
            actual = copied.refiner(fine, fine, warp, logits)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_public_match_observes_copied_head_reload(self):
        copied = copy.deepcopy(compile_for_cpu(make_matcher()))
        data = {
            "features": torch.randn(1, 8, 2, 2),
            "image_size": torch.tensor([[16.0, 16.0]]),
            "keypoints": torch.tensor(
                [[[2.0, 2.0], [6.0, 6.0], [10.0, 10.0], [13.0, 13.0]]]
            ),
        }
        with torch.no_grad():
            before = copied.match(data, data, max_distance_px=0)["p2d0_n1"].clone()
            copied.heads[1].offset_head.head[-1].bias[:17].add_(torch.arange(17))
            actual = copied.match(data, data, max_distance_px=0)["p2d0_n1"]
            copied.static_lengths = None
            expected = copied.match(data, data, max_distance_px=0)["p2d0_n1"]
            self.assertFalse(torch.equal(expected, before))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
