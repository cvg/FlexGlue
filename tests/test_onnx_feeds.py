import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2  # noqa: F401
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "onnx"


def load_pair():
    spec = importlib.util.spec_from_file_location(
        "match_pair_test", PATH / "match_pair.py"
    )
    module = importlib.util.module_from_spec(spec)
    fake_ort = types.SimpleNamespace(InferenceSession=object)
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        spec.loader.exec_module(module)
    return module


class Session:
    def __init__(self, early=False, extra=()):
        names = [
            "features0",
            "features1",
            "kpts0",
            "kpts1",
            "image_size0",
            "image_size1",
            "filter_threshold",
            "max_distance_px",
            "min_covisibility",
        ]
        self.names = names + (["early_stop_decay"] if early else []) + list(extra)
        self.seen = []

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map={})

    def get_inputs(self):
        return [types.SimpleNamespace(name=name) for name in self.names]

    def run(self, outputs, feeds):
        self.seen.append(feeds)
        # Historical graph compares distances in supplied coordinates. Current
        # graphs apply descriptor scaling internally; with descriptor-sized
        # inputs that scaling is the identity for both contracts.
        distance = abs(feeds["kpts0"][0, 0, 1] - feeds["kpts1"][0, 0, 1])
        threshold = float(feeds["max_distance_px"])
        accepted = distance < threshold
        return [
            np.array([[0 if accepted else -1]]),
            np.array([[0.9]], np.float32),
            np.array([0.9], np.float32),
        ]


class MatcherFeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pair = load_pair()

    def test_stride_contract_refuses_conflicting_metadata(self):
        self.assertEqual(self.pair.matcher_stride({}), 8)
        self.assertEqual(self.pair.matcher_stride({"flexglue.stride": "8"}), 8)
        for stride in ("4", "0", "-8", "8.0", "unknown"):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                self.pair.matcher_stride({"flexglue.stride": stride})

    def test_both_graph_contracts_use_descriptor_pixels_and_keep_inputs(self):
        f0 = np.zeros((1, 2, 64, 64), np.float32)
        f1 = np.zeros((1, 2, 32, 64), np.float32)
        k0 = np.array([[200, 100]], np.float32)
        k1 = np.array([[200, 104]], np.float32)
        originals = (k0.copy(), k1.copy())
        for early in (False, True):
            session = Session(early)
            matches, _, _ = self.pair.match(
                session, f0, f1, k0, k1, (1024, 512), (1024, 256), max_distance_px=3.0
            )
            self.assertEqual(len(matches), 0)
            feeds = session.seen[0]
            np.testing.assert_array_equal(feeds["image_size0"], [[512, 512]])
            np.testing.assert_array_equal(feeds["image_size1"], [[512, 256]])
            np.testing.assert_array_equal(feeds["kpts0"], [[[100, 100]]])
            np.testing.assert_array_equal(feeds["kpts1"], [[[100, 104]]])
            self.assertEqual("early_stop_decay" in feeds, early)
        np.testing.assert_array_equal(k0, originals[0])
        np.testing.assert_array_equal(k1, originals[1])

    def test_anisotropic_nearest_neighbor_on_old_and_current_contracts(self):
        class NeighborSession(Session):
            def __init__(self, current):
                super().__init__()
                self.current = current

            def run(self, outputs, feeds):
                self.seen.append(feeds)
                source = feeds["kpts0"][0, 0]
                target = feeds["kpts1"][0]
                size = feeds["image_size1"][0]
                descriptor = np.array([512.0, 512.0])
                # Current graph scales per axis internally. The old graph
                # measures in the supplied coordinates.
                scale = descriptor / size if self.current else np.ones(2)
                distances = np.linalg.norm((target - source) * scale, axis=-1)
                index = int(np.argmin(distances))
                if distances[index] >= float(feeds["max_distance_px"]):
                    index = -1
                return [
                    np.array([[index]]),
                    np.array([[0.9]], np.float32),
                    np.array([0.9], np.float32),
                ]

        f = np.zeros((1, 2, 64, 64), np.float32)
        k0 = np.array([[100.0, 100.0]], np.float32)
        k1 = np.array([[104.0, 100.0], [100.0, 103.0]], np.float32)
        for current in (False, True):
            session = NeighborSession(current)
            matches, _, _ = self.pair.match(
                session, f, f, k0, k1, (1024, 512), (1024, 512), max_distance_px=2.5
            )
            np.testing.assert_array_equal(matches, [[0, 0]])
            # In original coordinates candidate 1 would be nearer (3 versus 4).

    def test_unknown_input_and_invalid_size_refused(self):
        session = Session(extra=("new_required_input",))
        f = np.zeros((1, 2, 64, 64), np.float32)
        k = np.array([[10, 10]], np.float32)
        with self.assertRaisesRegex(ValueError, "Unsupported matcher inputs"):
            self.pair.match(session, f, f, k, k, (1024, 512), (1024, 512))
        self.assertEqual(session.seen, [])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.pair.match(Session(), f, f, k, k, (float("nan"), 512), (1024, 512))

    def test_checker_routes_original_debug_and_adapted_matcher_and_stops(self):
        import torch

        fake_ort = types.SimpleNamespace(InferenceSession=object)
        fake_export = types.SimpleNamespace(
            DEBUG_OUTPUTS=["p2d0_i1", "confidence0", "p2d1_i0", "confidence1"],
            MATCHER_OUTPUTS=["matches0", "mscores0", "covisibility_score"],
            export_matcher=lambda *a, **kw: None,
            read_early_stop_layers=lambda graphs: [],
        )
        spec = importlib.util.spec_from_file_location("check_test", PATH / "check.py")
        check = importlib.util.module_from_spec(spec)
        modules = {
            "onnx": types.ModuleType("onnx"),
            "onnxruntime": fake_ort,
            "export": fake_export,
            "match_pair": self.pair,
        }
        with patch.dict(sys.modules, modules):
            with patch.object(
                sys, "path", [str(Path(check.__file__).parent), *sys.path]
            ):
                spec.loader.exec_module(check)

        class Debug:
            def __init__(self):
                self.seen = []

            def run(self, names, feeds):
                self.seen.append(feeds)
                return [None] * 7

        matcher = Session(early=True)
        debug = Debug()
        sessions = types.SimpleNamespace(matcher=matcher, debug=debug, early_stop=True)
        features = [
            {
                "features": torch.zeros(1, 2, 64, 64),
                "image_size": torch.tensor([[1024.0, 512.0]]),
            }
            for _ in range(2)
        ]
        kpts = [torch.tensor([[[200.0, 100.0]]]), torch.tensor([[[200.0, 104.0]]])]
        with patch.object(check, "match_pytorch", return_value={}), patch.object(
            check, "compare_matches", return_value=(True, "ok")
        ), patch.object(check, "compare_points", return_value=(True, "ok")):
            self.assertTrue(
                check.check_matches(sessions, None, features, kpts, 0.0, {}, False)
            )
            self.assertTrue(
                check.check_matches(sessions, None, features, kpts, 0.8, {}, False)
            )
        np.testing.assert_array_equal(debug.seen[0]["kpts0"], [[[200.0, 100.0]]])
        self.assertEqual(len(matcher.seen), 5)  # two baseline plus three enabled decays
        for feed in matcher.seen:
            np.testing.assert_array_equal(feed["kpts0"], [[[100.0, 100.0]]])
            np.testing.assert_array_equal(feed["image_size0"], [[512.0, 512.0]])

    def test_empty_keypoints_skip_session(self):
        session = Session()
        f = np.zeros((1, 2, 64, 64), np.float32)
        matches, _, score = self.pair.match(
            session, f, f, np.empty((0, 2)), np.ones((1, 2)), (1, 1), (1, 1)
        )
        self.assertEqual(matches.shape, (0, 2))
        self.assertTrue(np.isnan(score))
        self.assertEqual(session.seen, [])


if __name__ == "__main__":
    unittest.main()
