"""Native refinement configuration must not leak into the coarse ONNX graph."""
import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from onnx_test_utils import load_export

from flexglue import FlexGlue
from flexglue.refiner import VGG19BN, ConvRefiner, RoMav2Refiner


def make_refiner():
    features = VGG19BN()
    stage = ConvRefiner(*RoMav2Refiner.configs[1])
    state = {
        **{"refiner_features." + k: v for k, v in features.state_dict().items()},
        **{"refiners.1." + k: v for k, v in stage.state_dict().items()},
    }
    with patch("flexglue.refiner.ops.load_state_dict", return_value=state):
        return RoMav2Refiner(scales=(1,), weights="generated").eval()


class ExportRefinementContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript", "onnxruntime"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        import onnxruntime as ort

        ort.disable_telemetry_events()
        cls.export = load_export()

    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(20261110)
        self.model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=1,
            input_dim=256,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            mp=None,
        ).eval()
        self.model.refiner = make_refiner()
        self.model.conf.refine = True

    def inputs(self, n0, n1):
        torch.manual_seed(1500 + n0 + n1)
        values = list(self.export.dummy_matcher_inputs((16, 16), n0, n1))
        values[-3:] = [torch.tensor(0.0), torch.tensor(float("inf")), torch.tensor(0.0)]
        return values

    def test_wrapper_matches_explicit_native_coarse_consumer(self):
        values = self.inputs(7, 9)
        calls = []
        handle = self.model.refiner.register_forward_pre_hook(
            lambda *args: calls.append(1)
        )
        self.addCleanup(handle.remove)
        wrapper = self.export.SparseMatcherONNX(self.model, debug_outputs=True)
        with torch.inference_mode():
            actual = wrapper(*values)
            data0, data1 = (
                {
                    "features": values[i],
                    "keypoints": values[2 + i],
                    "image_size": values[4 + i],
                }
                for i in (0, 1)
            )
            reference = self.model.match(
                data0,
                data1,
                sparse=True,
                refine=False,
                exact_nn=True,
                filter_threshold=0.0,
                max_distance_px=float("inf"),
                min_covisibility=0.0,
            )
        expected = (
            reference["matches0"],
            reference["matching_scores0"],
            reference["covisibility_score"],
            *(reference[k] for k in self.export.DEBUG_OUTPUTS),
        )
        self.assertEqual(calls, [])
        self.assertTrue(self.model.conf.refine)
        self.assertFalse(wrapper.conf.refine)
        self.assertGreater(int((actual[0] >= 0).sum()), 0)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_layer_scores_inherits_the_coarse_export_contract(self):
        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=2,
            input_dim=256,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            mp=None,
        ).eval()
        model.refiner = self.model.refiner
        model.conf.refine = True
        wrapper = self.export.LayerScoresMatcherONNX(model, (1,))
        self.assertFalse(wrapper.conf.refine)
        with torch.inference_mode():
            actual = wrapper(*self.inputs(7, 9))
            model.conf.refine = False
            expected = self.export.LayerScoresMatcherONNX(model, (1,))(
                *self.inputs(7, 9)
            )
            model.conf.refine = True
        self.assertEqual(len(actual), 4)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertTrue(model.conf.refine)

    def test_public_standard_and_fused_exports_match_coarse_control(self):
        import onnxruntime as ort

        control = copy.deepcopy(self.model)
        control.conf.refine = False
        modes = {name: module.training for name, module in self.model.named_modules()}
        before = self.model.input_proj.weight.detach().clone()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        with tempfile.TemporaryDirectory() as temporary, torch.inference_mode():
            destination = Path(temporary)
            for fused in (False, True):
                sessions = []
                for label, model in (
                    ("coarse", control),
                    ("refinement_enabled", self.model),
                ):
                    path = destination / f"refinement_contract_{fused}_{label}.onnx"
                    path.write_bytes(b"existing artifact")
                    with contextlib.redirect_stdout(
                        io.StringIO()
                    ), contextlib.redirect_stderr(io.StringIO()):
                        self.export.export_matcher(
                            model,
                            path,
                            (16, 16),
                            18,
                            fused=fused,
                            debug_outputs=True,
                        )
                    sessions.append(
                        ort.InferenceSession(
                            str(path),
                            options,
                            providers=["CPUExecutionProvider"],
                        )
                    )
                for n0, n1 in ((1, 1), (7, 9), (129, 257)):
                    with self.subTest(fused=fused, n0=n0, n1=n1):
                        feeds = {
                            k: v.numpy()
                            for k, v in zip(
                                self.export.MATCHER_INPUTS, self.inputs(n0, n1)
                            )
                        }
                        expected, actual = [
                            session.run(None, feeds) for session in sessions
                        ]
                        self.assertGreater(int((actual[0] >= 0).sum()), 0)
                        for got, want in zip(actual, expected):
                            torch.testing.assert_close(
                                torch.from_numpy(got),
                                torch.from_numpy(want),
                                rtol=0,
                                atol=0,
                            )
        self.assertTrue(self.model.conf.refine)
        self.assertEqual(
            modes,
            {name: module.training for name, module in self.model.named_modules()},
        )
        torch.testing.assert_close(self.model.input_proj.weight, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
