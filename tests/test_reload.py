import unittest
from unittest.mock import patch

import torch
from torch import nn

from flexglue import FlexGlue
from flexglue.utils import ops


class ReloadTest(unittest.TestCase):
    def setUp(self):
        self.model = FlexGlue(
            weights=None,
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=2,
        )
        self.model.descriptor = nn.Linear(3, 2)
        self.model.refiner = nn.Linear(2, 1)
        self.state = {
            k: v.clone()
            for k, v in self.model.state_dict().items()
            if not k.startswith(("descriptor.", "refiner."))
        }
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def load(self, state):
        with patch.object(ops, "_read_state_dict", return_value=state):
            with patch.object(ops, "load_state_dict", return_value=state) as loader:
                self.model.load_weights("checkpoint")
        loader.assert_called_once_with("checkpoint", "matcher.", state_dict=state)

    def owners(self):
        return [id(p) for group in self.optimizer.param_groups for p in group["params"]]

    def test_rejections_do_not_copy_or_change_metadata(self):
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        owners, heads = self.owners(), self.model.loaded_heads.copy()
        key = "input_proj.weight"
        value = self.state[key]
        cases = {}
        missing = dict(self.state)
        del missing[key]
        cases["missing"] = missing
        partial = dict(self.state)
        del partial[next(k for k in partial if k.startswith("heads.1."))]
        cases["partial"] = partial
        cases["unexpected"] = {**self.state, "unexpected": torch.tensor(1)}
        for name, invalid in (
            ("shape", torch.zeros(1)),
            ("type", "invalid"),
            ("meta", torch.empty_like(value, device="meta")),
            ("sparse", value.to_sparse()),
            ("quantized", torch.quantize_per_tensor(value, 0.1, 0, torch.qint8)),
        ):
            state = dict(self.state)
            del state[key]
            state[key] = invalid  # put the invalid entry after valid entries
            cases[name] = state
        for name, state in cases.items():
            with self.subTest(name=name):
                state["input_proj.bias"] = torch.ones_like(
                    self.state["input_proj.bias"]
                )
                with self.assertRaises((KeyError, RuntimeError, TypeError)):
                    self.load(state)
                self.assertEqual(self.model.loaded_heads, heads)
                self.assertEqual(self.owners(), owners)
                for k, v in self.model.state_dict().items():
                    self.assertTrue(torch.equal(v, original[k]), k)

    def test_successful_reload_preserves_owners_and_other_modules(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        owners = self.owners()
        state = dict(self.state)
        # A non-contiguous dense source and different dtype remain valid.
        weight = state["input_proj.weight"]
        state["input_proj.weight"] = torch.ones(weight.shape[::-1]).t().double()
        self.assertFalse(state["input_proj.weight"].is_contiguous())
        state["heads.0.temperature"] = state["heads.0.temperature"].reshape(1)
        for _ in range(2):
            self.load(state)
        self.assertEqual(self.owners(), owners)
        self.assertEqual(self.model.loaded_heads, {0, 1, 2})
        self.assertEqual(self.model.input_proj.weight.dtype, weight.dtype)
        self.assertTrue(
            torch.equal(self.model.input_proj.weight, torch.ones_like(weight))
        )
        for k, v in self.model.state_dict().items():
            if k.startswith(("descriptor.", "refiner.")):
                self.assertTrue(torch.equal(v, before[k]), k)

    def test_whole_head_may_be_absent(self):
        state = {k: v for k, v in self.state.items() if not k.startswith("heads.1.")}
        self.load(state)
        self.assertEqual(self.model.loaded_heads, {0, 2})
        self.load(self.state)
        self.assertEqual(self.model.loaded_heads, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
