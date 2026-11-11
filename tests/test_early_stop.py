import unittest
from unittest.mock import patch

import torch
from torch import nn

from flexglue import FlexGlue


class CountingBlock(nn.Module):
    def __init__(self, layer, calls):
        super().__init__()
        self.layer = layer
        self.calls = calls

    def forward(self, x0, x1, enc0, enc1):
        self.calls.append(self.layer)
        return x0, x1


class EarlyStopScheduleTest(unittest.TestCase):
    def setUp(self):
        self.model = FlexGlue(
            weights=None,
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=5,
            mp=None,
        ).eval()
        self.calls = []
        self.model.transformers = nn.ModuleList(
            [CountingBlock(k, self.calls) for k in range(1, 6)]
        )
        self.data = [
            {"features": torch.zeros(2, 4, 2, 2), "image_size": torch.ones(2, 2)}
            for _ in range(2)
        ]

    def decode(self, scores, **overrides):
        self.calls.clear()
        score_calls = []

        def score(x0, x1, layer):
            score_calls.append(layer)
            return torch.tensor(scores.get(layer, [0.9, 0.9]))

        conf = self.model.resolve_conf(
            {"early_stop_decay": 0.5, "min_covisibility": 1.0, **overrides}
        )
        with patch.object(self.model, "covisibility_score", side_effect=score):
            context = self.model.decode_pair(*self.data, conf)
        return context, score_calls, list(self.calls)

    def test_default_skips_first_layer(self):
        context, checks, blocks = self.decode({1: [0.0, 0.0], 2: [0.01, 0.01]})
        self.assertEqual(context.layer, 2)
        self.assertEqual(checks, [2])
        self.assertEqual(blocks, [1, 2])

    def test_default_at_reduced_depths(self):
        for depth, expected in ((0, 0), (1, 1), (2, 2), (3, 2), (5, 2)):
            with self.subTest(depth=depth):
                context, checks, blocks = self.decode(
                    {1: [0.0, 0.0], 2: [0.0, 0.0]}, n_layers=depth
                )
                self.assertEqual(context.layer, expected)
                self.assertEqual(checks, [expected])
                self.assertEqual(blocks, list(range(1, expected + 1)))

    def test_nine_layer_default_checks_only_intermediate_layers_from_two(self):
        self.model.conf.n_layers = 9
        self.model.loaded_heads = set(range(10))
        for depth in (0, 1, 2, 3, 9):
            with self.subTest(depth=depth):
                conf = self.model.resolve_conf({"n_layers": depth})
                self.assertEqual(conf.early_stop_layers, tuple(range(2, depth)))

    def test_first_layer_remains_an_explicit_schedule(self):
        context, checks, blocks = self.decode({1: [0.0, 0.0]}, early_stop_layers=(1,))
        self.assertEqual(context.layer, 1)
        self.assertEqual(checks, [1])
        self.assertEqual(blocks, [1])

    def test_nan_score_does_not_stop(self):
        context, checks, blocks = self.decode({2: [float("nan"), 0.0], 4: [0.0, 0.0]})
        self.assertEqual(context.layer, 4)
        self.assertEqual(checks, [2, 3, 4])
        self.assertEqual(blocks, [1, 2, 3, 4])

    def test_explicit_schedule_and_empty(self):
        context, checks, blocks = self.decode(
            {1: [0.0, 0.0], 2: [0.9, 0.9], 4: [0.0, 0.0]},
            early_stop_layers=[4, 2],
        )
        self.assertEqual(context.layer, 4)
        self.assertEqual(checks, [2, 4])
        self.assertEqual(blocks, [1, 2, 3, 4])
        context, checks, blocks = self.decode({1: [0.0, 0.0]}, early_stop_layers=())
        self.assertEqual(context.layer, 5)
        self.assertEqual(checks, [5])  # final score, no early checks
        self.assertEqual(blocks, [1, 2, 3, 4, 5])

    def test_mixed_batch_requires_every_item_to_stop(self):
        context, checks, blocks = self.decode(
            {2: [0.0, 0.5], 4: [0.0, 0.0]}, early_stop_layers=(2, 4)
        )
        self.assertEqual(context.layer, 4)
        self.assertEqual(checks, [2, 4])
        self.assertEqual(blocks, [1, 2, 3, 4])

    def test_reduced_depth_filters_checks_but_keeps_model_depth_exponent(self):
        # At layer 2 the threshold is 1 * 0.5**(5-2) = 0.125,
        # not 1 * 0.5**(3-2) = 0.5.
        context, checks, blocks = self.decode(
            {2: [0.25, 0.25]}, n_layers=3, early_stop_layers=(2, 4)
        )
        self.assertEqual(context.layer, 3)
        self.assertEqual(checks, [2, 3])
        self.assertEqual(blocks, [1, 2, 3])

    def test_equality_does_not_stop(self):
        context, checks, blocks = self.decode(
            {2: [0.125, 0.125], 4: [0.0, 0.0]}, early_stop_layers=(2, 4)
        )
        self.assertEqual(context.layer, 4)
        self.assertEqual(checks, [2, 4])
        self.assertEqual(blocks, [1, 2, 3, 4])

    def test_only_selected_and_final_heads_are_required(self):
        self.model.loaded_heads = {2, 4, 5}
        conf = self.model.resolve_conf(
            {
                "early_stop_layers": (2, 4),
                "early_stop_decay": 0.5,
                "min_covisibility": 1.0,
            }
        )
        self.assertEqual(conf.early_stop_layers, (2, 4))
        with self.assertRaises(ValueError):
            self.model.resolve_conf(
                {
                    "early_stop_layers": None,
                    "early_stop_decay": 0.5,
                    "min_covisibility": 1.0,
                }
            )
        self.model.loaded_heads = {2, 3}
        self.model.resolve_conf(
            {
                "n_layers": 3,
                "early_stop_layers": (2, 4),
                "early_stop_decay": 0.5,
                "min_covisibility": 1.0,
            }
        )
        self.model.loaded_heads = {2, 4}
        with self.assertRaises(ValueError):
            self.model.resolve_conf(
                {
                    "early_stop_layers": (2, 4),
                    "early_stop_decay": 0.5,
                    "min_covisibility": 1.0,
                }
            )

    def test_invalid_schedules_fail_before_decode(self):
        for schedule in ("2,4", 2, (True,), (1.5,), (0,), (5,), (-1,), (2, 2)):
            with self.subTest(schedule=schedule):
                with patch.object(self.model, "decode_pair") as decode:
                    with self.assertRaises((ValueError, TypeError)):
                        self.model.match_dense(*self.data, early_stop_layers=schedule)
                    decode.assert_not_called()

    def test_native_decay_one_is_not_supported(self):
        with self.assertRaises(ValueError):
            self.model.resolve_conf({"early_stop_decay": 1.0, "min_covisibility": 1.0})


if __name__ == "__main__":
    unittest.main()
