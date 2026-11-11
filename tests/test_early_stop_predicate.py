import ast
import re
import types
import unittest
from pathlib import Path


class PredicateReaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "onnx" / "export.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "read_early_stop_layers"
        )

        class Value:
            def __init__(self, v):
                self.v, self.size = v, 1

            def __float__(self):
                return float(self.v)

        onnx = types.SimpleNamespace(
            numpy_helper=types.SimpleNamespace(to_array=lambda v: Value(v))
        )
        helper = types.SimpleNamespace(get_attribute_value=lambda a: a.value)
        namespace = {"re": re, "onnx": onnx, "helper": helper}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        cls.read = staticmethod(namespace["read_early_stop_layers"])

    @staticmethod
    def graph(layer=2, exponent=7.0, threshold_input="min_covisibility"):
        def n(op, inputs, output, value=None):
            attrs = (
                []
                if value is None
                else [types.SimpleNamespace(name="value", value=value)]
            )
            return types.SimpleNamespace(
                op_type=op,
                input=inputs,
                output=[output],
                attribute=attrs,
                name="renamed",
            )

        return types.SimpleNamespace(
            node=[
                n("Constant", [], "one", 1.0),
                n("Constant", [], "exponent", exponent),
                n("Sub", ["one", "early_stop_decay"], "keep"),
                n("Pow", ["keep", "exponent"], "gamma"),
                n("Mul", [threshold_input, "gamma"], "threshold"),
                n(
                    "Less",
                    [f"covisibility_score_layer{layer}", "threshold"],
                    "decision",
                ),
                n("If", ["decision"], "result"),
            ]
        )

    def test_reads_schedule_and_ignores_if_node_name(self):
        self.assertEqual(self.read([self.graph(2, 7), self.graph(4, 5)]), [2, 4])

    def test_refuses_changed_exponent_and_unknown_threshold(self):
        for graph in (self.graph(2, 6), self.graph(2, 7, "other_threshold")):
            with self.assertRaisesRegex(ValueError, "Cannot identify"):
                self.read([graph])

    def test_refuses_duplicate_and_unknown_predicate(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.read([self.graph(), self.graph()])
        graph = self.graph()
        graph.node[-2].op_type = "Greater"
        with self.assertRaisesRegex(ValueError, "Cannot identify"):
            self.read([graph])


if __name__ == "__main__":
    unittest.main()
