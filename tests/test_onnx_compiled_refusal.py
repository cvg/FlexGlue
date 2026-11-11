"""Compiled native state must be refused before copying or writing an export."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from onnx_test_utils import load_export

from flexglue import FlexGlue


class CompiledExportRefusalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript", "onnxruntime"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        cls.export = load_export()

    def test_callable_boundary_and_custom_wrapper(self):
        export = self.export
        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=1,
            descriptor_dim=8,
            input_dim=256,
            search_dim=4,
            num_heads=2,
            mp=None,
        ).eval()
        original = model.transformers[0].forward

        # A normal customization is allowed through the guard and is not unwrapped.
        def custom(*args, **kwargs):
            return original(*args, **kwargs)

        model.transformers[0].forward = custom
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "matcher.onnx"
            target.write_bytes(b"existing")
            with patch.object(
                export.copy, "deepcopy", side_effect=RuntimeError("reached copy")
            ):
                with self.assertRaisesRegex(RuntimeError, "reached copy"):
                    export.export_matcher(model, target, (32, 32), 18)
            self.assertIs(model.transformers[0].forward, custom)
            # Independently compiled transformer, or an interrupted compile(), can
            # exist while static_lengths is still None.
            model.transformers[0].forward = torch.compile(custom, backend="eager")
            with patch.object(
                export.copy, "deepcopy", side_effect=AssertionError("copied")
            ):
                with self.assertRaisesRegex(
                    ValueError, "uncompiled matcher transformer"
                ):
                    export.export_matcher(model, target, (32, 32), 18)
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertIsNone(model.static_lengths)

    def test_descriptor_checks_encode_before_eval_or_write(self):
        export = self.export

        class Descriptor(torch.nn.Module):
            def encode(self, x):
                return x

        descriptor = Descriptor().train()
        descriptor.encode = torch.compile(descriptor.encode, backend="eager")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "descriptor.onnx"
            target.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "uncompiled descriptor encode"):
                export.export_descriptor(descriptor, target, (32, 32), 18)
            self.assertTrue(descriptor.training)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_eager_export_survives_refused_compiled_source(self):
        for name in ("onnx", "onnxscript", "onnxruntime"):
            if importlib.util.find_spec(name) is None:
                self.skipTest(f"optional dependency {name} is unavailable")
        import onnxruntime as ort

        export = self.export
        torch.manual_seed(20261110)
        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=1,
            descriptor_dim=8,
            input_dim=256,
            search_dim=4,
            num_heads=2,
            mp=None,
        ).eval()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            target = Path(tmp) / "matcher.onnx"
            export.export_matcher(model, target, (32, 32), 18, fused=False)
            original = target.read_bytes()
            state = {key: value.clone() for key, value in model.state_dict().items()}
            for lengths in ((512, 1024, 2048, 4096), ()):
                model.compile(static_lengths=lengths)
                for fused in (False, True):
                    with self.subTest(lengths=lengths, fused=fused):
                        with patch.object(
                            export.copy,
                            "deepcopy",
                            side_effect=AssertionError("copied compiled source"),
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "uncompiled matcher"
                            ):
                                export.export_matcher(
                                    model, target, (32, 32), 18, fused=fused
                                )
                        self.assertEqual(model.static_lengths, lengths)
                        self.assertEqual(target.read_bytes(), original)
                        self.assertEqual(list(Path(tmp).iterdir()), [target])
                        for key, value in state.items():
                            self.assertTrue(torch.equal(value, model.state_dict()[key]))
            session = ort.InferenceSession(
                str(target), options, providers=["CPUExecutionProvider"]
            )
            for n0, n1 in ((1, 1), (1025, 3), (3, 2049), (4097, 5)):
                with self.subTest(n0=n0, n1=n1):
                    values = list(export.dummy_matcher_inputs((32, 32), n0, n1))
                    values[-3:] = [
                        torch.tensor(0.0),
                        torch.tensor(float("inf")),
                        torch.tensor(0.0),
                    ]
                    feeds = {
                        key: value.numpy()
                        for key, value in zip(export.MATCHER_INPUTS, values)
                    }
                    outputs = session.run(None, feeds)
                    self.assertEqual(outputs[0].shape, (1, n0))
                    self.assertGreater(int((outputs[0] >= 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
