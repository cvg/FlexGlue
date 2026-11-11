"""Actual public non-GEMM matcher export, including dynamic runtime execution."""
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from onnx_test_utils import load_export

from flexglue import FlexGlue


class NonGemmMatcherExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript", "onnxruntime"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        cls.export = load_export()

    def test_public_export_matches_non_gemm_pytorch(self):
        import onnxruntime as ort

        torch.manual_seed(17)
        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=1,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
            query_chunk_size=None,
            gemm_nn=False,
        ).eval()
        with torch.no_grad():
            for parameter in model.heads[1].offset_head.parameters():
                parameter.zero_()
            model.heads[1].matchability.weight.zero_()
            model.heads[1].matchability.bias.fill_(2.0)
        size = (32, 32)
        for fused in (False, True):
            with self.subTest(fused=fused), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "matcher.onnx"
                with torch.inference_mode():
                    # Export diagnostics may contain Unicode on ASCII-only consoles.
                    with redirect_stdout(io.StringIO()):
                        self.export.export_matcher(model, path, size, 18, fused=fused)
                    session = ort.InferenceSession(
                        str(path), providers=["CPUExecutionProvider"]
                    )
                    for n0, n1 in ((3, 5), (11, 2)):
                        inputs = list(self.export.dummy_matcher_inputs(size, n0, n1))
                        inputs[4] = inputs[5] = torch.tensor([[32.0, 32.0]])
                        # Identical normalized coordinates guarantee retained pairs.
                        for k, count in ((2, n0), (3, n1)):
                            points = torch.arange(count, dtype=torch.float32) + 3
                            inputs[k] = torch.stack((points, points), -1)[None]
                        inputs[6], inputs[7] = torch.tensor(0.0), torch.tensor(
                            float("inf")
                        )
                        data = [
                            {
                                "features": inputs[k],
                                "keypoints": inputs[2 + k],
                                "image_size": inputs[4 + k],
                            }
                            for k in (0, 1)
                        ]
                        pred = model.match(
                            *data,
                            sparse=True,
                            exact_nn=True,
                            filter_threshold=inputs[6],
                            max_distance_px=inputs[7],
                        )
                        outputs = session.run(
                            None,
                            {
                                key: value.numpy()
                                for key, value in zip(
                                    self.export.MATCHER_INPUTS, inputs
                                )
                            },
                        )
                        self.assertGreater(int((pred["matches0"] >= 0).sum()), 0)
                        self.assertTrue(
                            torch.equal(torch.from_numpy(outputs[0]), pred["matches0"])
                        )
                        for result, expected in zip(
                            outputs[1:],
                            (pred["matching_scores0"], pred["covisibility_score"]),
                        ):
                            torch.testing.assert_close(
                                torch.from_numpy(result), expected, rtol=1e-4, atol=1e-5
                            )
