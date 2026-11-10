import importlib.util
import unittest
from unittest.mock import patch

import torch
from onnx_test_utils import load_export
from torch import nn

from flexglue.utils import modules


class TinyMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformers = nn.ModuleList([modules.TransformerLayer(8, 2)])
        self.heads = nn.ModuleList()


class ExportIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        cls.export = load_export()

    def test_fused_context_is_local_and_native_backward_works(self):
        export_model, native = TinyMatcher(), TinyMatcher()
        exported = export_model.transformers[0]
        native_cross = native.transformers[0].cross_attn
        original = modules.attention
        calls = []

        def fake_fused(q, k, v):
            calls.append(1)
            return original(q, k, v)

        with patch.object(self.export, "fused_attention", fake_fused):
            with self.export.exportable(export_model):
                self.assertIs(modules.attention, original)
                self.assertIs(exported.self_attn.attention, fake_fused)
                self.assertIs(exported.cross_attn.attention, fake_fused)
                x = torch.randn(1, 3, 8, requires_grad=True)
                y = torch.randn(1, 4, 8, requires_grad=True)
                outputs = native_cross(x, y)
                sum(t.sum() for t in outputs).backward()
                self.assertIsNotNone(x.grad)
                self.assertEqual(calls, [])
                with torch.no_grad():
                    exported.cross_attn(x.detach(), y.detach())
                self.assertEqual(len(calls), 2)
        self.assertIsNone(exported.self_attn.attention)
        self.assertIsNone(exported.cross_attn.attention)
        self.assertIs(modules.attention, original)

    def test_nested_overlapping_and_exception_restore(self):
        first, second = TinyMatcher(), TinyMatcher()
        a, b = first.transformers[0].cross_attn, second.transformers[0].cross_attn
        native = modules.attention
        with self.export.exportable(first):
            self.assertIs(a.attention, self.export.fused_attention)
            with self.export.exportable(first):
                self.assertIs(a.attention, self.export.fused_attention)
            self.assertIs(a.attention, self.export.fused_attention)
        self.assertIsNone(a.attention)

        outer = self.export.exportable(first)
        inner = self.export.exportable(second)
        outer.__enter__()
        inner.__enter__()
        outer.__exit__(None, None, None)
        self.assertIsNone(a.attention)
        self.assertIs(b.attention, self.export.fused_attention)
        inner.__exit__(None, None, None)
        self.assertIsNone(b.attention)
        self.assertIs(modules.attention, native)

        with self.assertRaisesRegex(RuntimeError, "export failed"):
            with self.export.exportable(first):
                raise RuntimeError("export failed")
        self.assertIsNone(a.attention)
        self.assertIs(modules.attention, native)


if __name__ == "__main__":
    unittest.main()
