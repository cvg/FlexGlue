import inspect
import mmap
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from flexglue import FlexGlue
from flexglue.utils import ops


class WeightLoadingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "weights.pth")
        self.conf = dict(
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=2,
        )
        torch.manual_seed(0)
        self.original = FlexGlue(weights=None, **self.conf)
        self.state = {"matcher." + k: v for k, v in self.original.state_dict().items()}

    def build(self, state=None, **conf):
        torch.save(self.state if state is None else state, self.path)
        return FlexGlue(weights=self.path, **{**self.conf, **conf})

    def test_default_copy_and_opted_local_mapping(self):
        raw = {"matcher.x": torch.tensor([1.0]), "extractor.y": torch.tensor([2.0])}
        torch.save(raw, self.path)
        mapped = ops._mapped_local_checkpoint(self.path)
        with patch.object(torch, "load", wraps=torch.load) as load:
            self.assertTrue(
                torch.equal(
                    ops.load_state_dict(self.path, "matcher.")["x"], raw["matcher.x"]
                )
            )
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )
        with patch.object(torch, "load", wraps=torch.load) as load:
            self.assertTrue(
                torch.equal(
                    ops.load_state_dict(self.path, "matcher.", allow_mmap=True)["x"],
                    raw["matcher.x"],
                )
            )
            kwargs = {"mmap": True} if mapped else {}
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True, **kwargs
            )
        with patch.object(torch, "load", side_effect=AssertionError("unexpected read")):
            self.assertIs(
                ops.load_state_dict(self.path, "extractor.", state_dict=raw)["y"],
                raw["extractor.y"],
            )
        self.assertEqual(set(raw), {"matcher.x", "extractor.y"})

    def test_private_mapping_and_assignment_activate_when_supported(self):
        private = getattr(mmap, "MAP_PRIVATE", None)
        options = getattr(torch.serialization, "get_default_mmap_options", None)
        default_device = getattr(torch, "get_default_device", None)
        if private is None or not callable(options) or not callable(default_device):
            self.skipTest("private mmap policy or default-device API unavailable")
        if options() != private or default_device().type != "cpu":
            self.skipTest("private mmap and default CPU device required")
        if torch.get_default_dtype() != torch.float32:
            self.skipTest("default FP32 dtype required")
        if "assign" not in inspect.signature(nn.Module.load_state_dict).parameters:
            self.skipTest("load_state_dict assignment unavailable")

        original = nn.Linear(4, 3)
        torch.save(original.state_dict(), self.path)
        try:
            torch.load(self.path, map_location="cpu", weights_only=True, mmap=True)
        except (TypeError, RuntimeError) as exc:
            self.skipTest(f"native checkpoint mapping unavailable: {exc}")
        with patch.object(torch, "load", wraps=torch.load) as load:
            checkpoint = ops._read_state_dict(self.path, allow_mmap=True)
        load.assert_called_once_with(
            self.path, map_location="cpu", weights_only=True, mmap=True
        )
        target = nn.Linear(4, 3)
        _, assigned = ops.load_into_module(
            target, checkpoint, self.path, allow_assign=True
        )
        self.assertTrue(assigned)
        self.assertEqual(target.weight.data_ptr(), checkpoint["weight"].data_ptr())
        self.assertEqual(target.bias.data_ptr(), checkpoint["bias"].data_ptr())

    def test_missing_prefix_in_preloaded_mapping(self):
        with self.assertRaisesRegex(KeyError, "No weights with prefix"):
            ops.load_state_dict(self.path, "matcher.", state_dict={})

    def test_legacy_format_uses_copy_loader(self):
        torch.save(self.state, self.path, _use_new_zipfile_serialization=False)
        with patch.object(torch, "load", wraps=torch.load) as load:
            actual = FlexGlue(weights=self.path, **self.conf)
        load.assert_called_once_with(self.path, map_location="cpu", weights_only=True)
        self.assertEqual(actual.loaded_heads, {0, 1, 2})

    def test_exact_module_assignment_preserves_grad_and_storage_lifetime(self):
        original = nn.Linear(4, 3)
        original.weight.requires_grad_(False)
        torch.save(original.state_dict(), self.path)
        checkpoint = ops._read_state_dict(self.path, allow_mmap=True)
        can_assign = ops._mapped_local_checkpoint(self.path) and (
            "assign" in inspect.signature(nn.Module.load_state_dict).parameters
        )
        target = nn.Linear(4, 3)
        target.weight.requires_grad_(False)
        (_, assigned) = ops.load_into_module(
            target, checkpoint, self.path, allow_assign=True
        )
        self.assertEqual(assigned, can_assign)
        self.assertEqual(
            target.weight.data_ptr() == checkpoint["weight"].data_ptr(), can_assign
        )
        self.assertFalse(target.weight.requires_grad)
        self.assertTrue(target.bias.requires_grad)
        del checkpoint
        self.assertTrue(torch.equal(target.weight, original.weight))
        self.assertTrue(torch.equal(target.bias, original.bias))

    def test_alias_or_dtype_mismatch_uses_copy(self):
        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(3))
                self.weight_copy = self.weight

        torch.save({"weight": torch.ones(3), "weight_copy": torch.ones(3)}, self.path)
        tied = Tied()
        before = tied.weight
        (_, assigned) = ops.load_into_module(
            tied,
            ops._read_state_dict(self.path, allow_mmap=True),
            self.path,
            allow_assign=True,
        )
        self.assertFalse(assigned)
        self.assertIs(tied.weight, before)
        self.assertIs(tied.weight_copy, before)

        source = nn.Linear(4, 3)
        torch.save(source.state_dict(), self.path)
        target = nn.Linear(4, 3, dtype=torch.float64)
        (_, assigned) = ops.load_into_module(
            target,
            ops._read_state_dict(self.path, allow_mmap=True),
            self.path,
            allow_assign=True,
        )
        self.assertFalse(assigned)
        self.assertEqual(target.weight.dtype, torch.float64)
        self.assertTrue(torch.equal(target.weight, source.weight.double()))

    def test_nondefault_global_dtype_disables_mapping(self):
        torch.save(self.state, self.path)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with patch.object(torch, "load", wraps=torch.load) as load:
                ops._read_state_dict(self.path, allow_mmap=True)
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )
        finally:
            torch.set_default_dtype(previous)

    def test_nondefault_global_device_disables_mapping(self):
        torch.save(self.state, self.path)
        with patch.object(
            torch, "get_default_device", return_value=torch.device("meta"), create=True
        ):
            with patch.object(torch, "load", wraps=torch.load) as load:
                ops._read_state_dict(self.path, allow_mmap=True)
        load.assert_called_once_with(self.path, map_location="cpu", weights_only=True)

    def test_missing_new_torch_apis_and_legacy_load_signature_use_copy(self):
        expected = nn.Linear(4, 3).state_dict()
        torch.save(expected, self.path)
        for missing in ("get_default_device", "get_default_mmap_options"):
            with self.subTest(missing=missing), patch.object(
                ops, "mmap", SimpleNamespace(MAP_PRIVATE=2)
            ), patch.object(
                torch,
                "get_default_device",
                None
                if missing == "get_default_device"
                else lambda: torch.device("cpu"),
                create=True,
            ), patch.object(
                torch.serialization,
                "get_default_mmap_options",
                None if missing == "get_default_mmap_options" else lambda: 2,
                create=True,
            ), patch.object(
                torch, "load", wraps=torch.load
            ) as load:
                state = ops._read_state_dict(self.path, allow_mmap=True)
                load.assert_called_once_with(
                    self.path, map_location="cpu", weights_only=True
                )
                for name, value in expected.items():
                    self.assertTrue(torch.equal(state[name], value))
                target = nn.Linear(4, 3)
                (_, assigned) = ops.load_into_module(
                    target, state, self.path, allow_assign=True
                )
                self.assertFalse(assigned)
                for name, value in target.state_dict().items():
                    self.assertTrue(torch.equal(value, expected[name]))

        class LegacyLinear(nn.Linear):
            def load_state_dict(self, state_dict, strict=True):
                return super().load_state_dict(state_dict, strict=strict)

        legacy = LegacyLinear(4, 3)
        (_, assigned) = ops.load_into_module(
            legacy,
            ops._read_state_dict(self.path, allow_mmap=True),
            self.path,
            allow_assign=True,
        )
        self.assertFalse(assigned)

    def test_missing_private_mapping_support_uses_ordinary_loader(self):
        torch.save(self.state, self.path)
        with patch.object(ops, "mmap", SimpleNamespace()):
            with patch.object(torch, "load", wraps=torch.load) as load:
                ops._read_state_dict(self.path, allow_mmap=True)
        load.assert_called_once_with(self.path, map_location="cpu", weights_only=True)

    def test_shared_mapping_policy_falls_back_without_mutating_checkpoint(self):
        torch.save({"weight": torch.ones(3)}, self.path)
        original_bytes = Path(self.path).read_bytes()
        # Exercise shared-policy refusal even on hosts without POSIX mmap flags.
        # Only the loader's view is mocked; torch's process-wide policy is unchanged.
        with patch.object(ops, "mmap", SimpleNamespace(MAP_PRIVATE=2)), patch.object(
            torch, "get_default_device", return_value=torch.device("cpu"), create=True
        ), patch.object(
            torch.serialization, "get_default_mmap_options", return_value=1, create=True
        ) as policy:
            with patch.object(torch, "load", wraps=torch.load) as load:
                state = ops._read_state_dict(self.path, allow_mmap=True)
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )

            class OneWeight(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.zeros(3))

            model = OneWeight()
            (_, assigned) = ops.load_into_module(
                model, state, self.path, allow_assign=True
            )
            self.assertFalse(assigned)
            with torch.no_grad():
                model.weight.add_(2)
            self.assertEqual(policy(), 1)
            self.assertEqual(Path(self.path).read_bytes(), original_bytes)
            self.assertTrue(torch.equal(state["weight"], torch.ones(3)))

    def test_url_options(self):
        url = "https://example.invalid/weights.pth"
        with patch.object(
            torch.hub, "load_state_dict_from_url", return_value={"matcher.x": 1}
        ) as load:
            self.assertEqual(ops.load_state_dict(url, "matcher."), {"x": 1})
            load.assert_called_once_with(url, map_location="cpu", weights_only=True)

    def test_standalone_matcher_reads_once(self):
        with patch.object(torch, "load", wraps=torch.load) as load:
            actual = self.build()
            self.assertEqual(load.call_count, 1)
            self.assertNotIn("mmap", load.call_args.kwargs)
        self.assertEqual(actual.loaded_heads, {0, 1, 2})
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, self.original.state_dict()[name]))

    def test_default_constructor_copies_independent_of_checkpoint_rewrite(self):
        actual = self.build()
        before = {k: v.detach().clone() for k, v in actual.state_dict().items()}
        torch.save({k: torch.zeros_like(v) for k, v in self.state.items()}, self.path)
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_opted_constructor_assigns_exact_state(self):
        assigned = []
        original = ops.load_into_module

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            assigned.append(result[1])
            return result

        torch.save(self.state, self.path)
        mapped = ops._mapped_local_checkpoint(self.path)
        can_assign = mapped and (
            "assign" in inspect.signature(nn.Module.load_state_dict).parameters
        )
        with patch.object(ops, "load_into_module", side_effect=observed):
            with patch.object(torch, "load", wraps=torch.load) as load:
                actual = self.build(mmap_weights=True)
        kwargs = {"mmap": True} if mapped else {}
        load.assert_called_once_with(
            self.path, map_location="cpu", weights_only=True, **kwargs
        )
        self.assertEqual(assigned, [can_assign])
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, self.original.state_dict()[name]))

    def test_explicit_reload_preserves_parameter_owners(self):
        actual = self.build(mmap_weights=True)
        owners = {name: id(value) for name, value in actual.named_parameters()}
        optimizer = torch.optim.SGD(actual.parameters(), lr=0.01)
        optimizer_owners = [
            id(value) for group in optimizer.param_groups for value in group["params"]
        ]
        actual.load_weights(self.path)
        self.assertEqual(
            owners, {name: id(value) for name, value in actual.named_parameters()}
        )
        self.assertEqual(
            optimizer_owners,
            [
                id(value)
                for group in optimizer.param_groups
                for value in group["params"]
            ],
        )

    def test_combined_constructor_reads_once(self):
        class Descriptor(nn.Module):
            def __init__(
                self,
                *,
                weights,
                state_dict,
                mp,
                mmap_weights,
                _assign_source_unique,
                _assign_owned_state,
            ):
                super().__init__()
                self.mp = mp
                self.mmap_weights = mmap_weights
                self.assign_source_unique = _assign_source_unique
                self.assign_owned_state = _assign_owned_state
                self.raw = ops.load_state_dict(
                    weights, "extractor.", state_dict=state_dict
                )

        state = {**self.state, "extractor.x": torch.tensor([3.0])}
        with patch("flexglue.flexglue.DINOv3Descriptor", Descriptor):
            with patch.object(torch, "load", wraps=torch.load) as load:
                actual = self.build(state, describe=True)
                self.assertEqual(load.call_count, 1)
        self.assertTrue(torch.equal(actual.descriptor.raw["x"], state["extractor.x"]))
        self.assertIsNone(actual.descriptor.mp)
        self.assertFalse(actual.descriptor.assign_source_unique)
        self.assertTrue(actual.descriptor.assign_owned_state)
        self.assertFalse(actual.descriptor.mmap_weights)
        self.assertIsInstance(actual.conf.weights, str)
        self.assertNotIn("state_dict", vars(actual.conf))

    def test_cross_prefix_checkpoint_alias_disables_assignment(self):
        class Descriptor(nn.Module):
            def __init__(
                self,
                *,
                weights,
                state_dict,
                mp,
                mmap_weights,
                _assign_source_unique,
                _assign_owned_state,
            ):
                super().__init__()
                self.assign_source_unique = _assign_source_unique

        state = {
            **self.state,
            "extractor.x": self.state["matcher.input_proj.weight"],
        }
        with patch("flexglue.flexglue.DINOv3Descriptor", Descriptor):
            actual = self.build(state, describe=True, mmap_weights=True)
        self.assertFalse(actual.descriptor.assign_source_unique)
        self.assertEqual(actual.loaded_heads, {0, 1, 2})

    def test_descriptor_precision_is_fixed_and_passed_at_construction(self):
        class Descriptor(nn.Module):
            def __init__(
                self,
                *,
                weights,
                state_dict,
                mp,
                mmap_weights,
                _assign_source_unique,
                _assign_owned_state,
            ):
                super().__init__()
                self.mp = mp

        with patch("flexglue.flexglue.DINOv3Descriptor", Descriptor):
            actual = FlexGlue(weights=None, describe=True, describe_mp="float16")
        self.assertEqual(actual.descriptor.mp, "float16")
        with self.assertRaisesRegex(KeyError, "Fixed at construction"):
            actual.resolve_conf({"describe_mp": None})
        with self.assertRaisesRegex(KeyError, "Fixed at construction"):
            actual.resolve_conf({"mmap_weights": True})

    def test_mmap_option_rejects_nonboolean(self):
        with self.assertRaisesRegex(ValueError, "mmap_weights"):
            FlexGlue(weights=None, describe=False, mmap_weights="yes")

    def test_descriptor_precision_rejects_unknown_setting(self):
        with self.assertRaisesRegex(ValueError, "describe_mp"):
            FlexGlue(weights=None, describe=False, describe_mp="bfloat16")

    def test_truncated_layers(self):
        actual = self.build(n_layers=1)
        self.assertEqual(actual.loaded_heads, {0, 1})
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, self.original.state_dict()[name]))

    def test_whole_missing_head_allowed(self):
        state = {
            k: v for k, v in self.state.items() if not k.startswith("matcher.heads.1.")
        }
        assigned = []
        original = ops.load_into_module

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            assigned.append(result[1])
            return result

        with patch.object(ops, "load_into_module", side_effect=observed):
            actual = self.build(state)
        self.assertEqual(actual.loaded_heads, {0, 2})
        self.assertEqual(assigned, [False])

    def test_partial_head_rejected(self):
        state = dict(self.state)
        del state[next(k for k in state if k.startswith("matcher.heads.1."))]
        with self.assertRaisesRegex(KeyError, "only partially"):
            self.build(state)

    def test_non_head_missing_rejected(self):
        state = dict(self.state)
        del state["matcher.input_proj.weight"]
        with self.assertRaisesRegex(KeyError, "Missing weights"):
            self.build(state)

    def test_unexpected_key_rejected(self):
        with self.assertRaisesRegex(KeyError, "Unexpected weights"):
            self.build({**self.state, "matcher.unexpected": torch.tensor(1)})

    def test_empty_checkpoint_rejected(self):
        with self.assertRaisesRegex(KeyError, "No weights with prefix 'matcher.'"):
            self.build({})

    def test_reload_validates_before_copy_and_preserves_other_modules(self):
        model = self.build()
        model.descriptor = nn.Linear(3, 2)
        model.refiner = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        owners = [id(p) for group in optimizer.param_groups for p in group["params"]]
        original = {k: v.clone() for k, v in model.state_dict().items()}
        heads = model.loaded_heads.copy()
        valid = {k: v.clone() for k, v in self.state.items()}
        invalid = []
        missing = dict(valid)
        del missing["matcher.input_proj.weight"]
        invalid.append(missing)
        partial = dict(valid)
        del partial[next(k for k in partial if k.startswith("matcher.heads.1."))]
        invalid.append(partial)
        invalid.append({**valid, "matcher.unexpected": torch.tensor(1)})
        wrong_shape = dict(valid)
        wrong_shape["matcher.input_proj.weight"] = torch.zeros(1)
        invalid.append(wrong_shape)
        wrong_type = dict(valid)
        wrong_type["matcher.input_proj.weight"] = "invalid"
        invalid.append(wrong_type)
        late_key = "matcher.heads.2.covisibility.weight"
        late_value = valid[late_key]
        for unsupported in (
            torch.empty_like(late_value, device="meta"),
            late_value.to_sparse(),
            torch.quantize_per_tensor(late_value, 0.1, 0, torch.qint8),
        ):
            invalid.append({**valid, late_key: unsupported})
        for state in invalid:
            # If preflight were absent, this changed tensor could be copied first.
            state["matcher.input_proj.bias"] = torch.ones_like(
                valid["matcher.input_proj.bias"]
            )
            with self.assertRaises((KeyError, RuntimeError, TypeError)):
                model.load_weights(self.path, state_dict=state)
            self.assertEqual(model.loaded_heads, heads)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, original[k]), k)
            self.assertEqual(
                owners,
                [id(p) for group in optimizer.param_groups for p in group["params"]],
            )

        # Native load permits dtype conversion and legacy one-dimensional scalars.
        valid["matcher.input_proj.weight"] = valid["matcher.input_proj.weight"].double()
        valid["matcher.heads.0.temperature"] = valid[
            "matcher.heads.0.temperature"
        ].reshape(1)
        for _ in range(2):
            model.load_weights(self.path, state_dict=valid)
        for prefix in ("descriptor.", "refiner."):
            for k, v in model.state_dict().items():
                if k.startswith(prefix):
                    self.assertTrue(torch.equal(v, original[k]))
        self.assertEqual(
            owners, [id(p) for group in optimizer.param_groups for p in group["params"]]
        )


if __name__ == "__main__":
    unittest.main()
