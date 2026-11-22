"""Image boundary contracts for the ONNX-only example."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np


class ONNXExampleIOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("onnxruntime") is None:
            raise unittest.SkipTest("optional dependency onnxruntime is unavailable")
        spec = importlib.util.spec_from_file_location(
            "flexglue_onnx_example_io",
            Path(__file__).resolve().parents[1] / "onnx" / "match_pair.py",
        )
        cls.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.example)

    def setUp(self):
        threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        self.addCleanup(cv2.setNumThreads, threads)

    def test_blank_image_returns_xy_array(self):
        image = np.zeros((16, 24, 3), dtype=np.float32)
        image.setflags(write=False)
        points = self.example.detect_keypoints(image, 32)
        self.assertEqual(points.shape, (0, 2))
        self.assertEqual(points.dtype, np.float32)
        np.testing.assert_array_equal(points[:, 0], np.empty(0, np.float32))
        self.assertFalse(image.flags.writeable)
        self.assertFalse(image.any())

    def test_textured_image_keeps_keypoints_and_input_storage(self):
        rng = np.random.default_rng(20261122)
        backing = rng.integers(0, 256, (256, 320, 3), dtype=np.uint8)
        backing = backing.astype(np.float32) / 255
        image = backing[::2, ::-1]
        image.setflags(write=False)
        before = backing.copy()
        strides = image.strides

        # Reference the unchanged conversion, sorting and half-pixel convention.
        gray = cv2.cvtColor((image * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
        keypoints = cv2.SIFT_create(nfeatures=32).detect(gray, None)
        keypoints = sorted(keypoints, key=lambda point: -point.response)
        expected = np.array([point.pt for point in keypoints], dtype=np.float32)
        self.assertGreater(len(expected), 0)
        _, first = np.unique(expected, axis=0, return_index=True)
        expected = expected[np.sort(first)][:32] + 0.5

        actual = self.example.detect_keypoints(image, 32)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape[1], 2)
        self.assertEqual(image.strides, strides)
        self.assertFalse(image.flags.writeable)
        np.testing.assert_array_equal(backing, before)
        self.assertEqual(self.example.detect_keypoints(image, 0).shape, (0, 2))

    def test_empty_detector_result_does_not_run_matcher(self):
        empty = self.example.detect_keypoints(np.zeros((16, 24, 3), np.float32), 32)
        points = np.array([[1.5, 2.5]], dtype=np.float32)
        features = np.zeros((1, 256, 2, 3), dtype=np.float32)
        for points0, points1 in ((empty, points), (points, empty)):
            with self.subTest(empty_left=len(points0) == 0):
                session = Mock()
                matches, scores, covisibility = self.example.match(
                    session,
                    features,
                    features,
                    points0,
                    points1,
                    (24, 16),
                    (24, 16),
                )
                session.get_inputs.assert_not_called()
                session.run.assert_not_called()
                self.assertEqual(matches.shape, (0, 2))
                self.assertEqual(matches.dtype, np.int64)
                self.assertEqual(scores.shape, (0,))
                self.assertEqual(scores.dtype, np.float32)
                self.assertTrue(np.isnan(covisibility))

    def draw_empty_matches(self, path):
        image0 = np.full((8, 12, 3), 0.25, dtype=np.float32)
        image1 = np.full((12, 6, 3), 0.75, dtype=np.float32)
        self.example.draw_matches(
            image0,
            image1,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.float32),
            path,
        )

    def test_failed_write_does_not_report_success_or_create_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "matches.png"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                with self.assertRaises(OSError) as raised:
                    self.draw_empty_matches(path)
            self.assertIn(str(path), str(raised.exception))
            self.assertEqual(output.getvalue(), "")
            self.assertFalse(path.parent.exists())

    def test_native_write_exception_is_preserved(self):
        error = cv2.error("fixture encoder failure")
        output = io.StringIO()
        with patch.object(cv2, "imwrite", side_effect=error):
            with contextlib.redirect_stdout(output):
                with self.assertRaises(cv2.error) as raised:
                    self.draw_empty_matches(Path("unused.png"))
        self.assertIs(raised.exception, error)
        self.assertEqual(output.getvalue(), "")

    def test_successful_write_reports_saved_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.png"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.draw_empty_matches(path)
            decoded = cv2.imread(str(path))
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, (12, 18, 3))
            np.testing.assert_array_equal(decoded[:8, :12], 64)
            np.testing.assert_array_equal(decoded[:, 12:], 191)
            np.testing.assert_array_equal(decoded[8:, :12], 0)
            self.assertEqual(output.getvalue(), f"Saved the matches to {path}\n")


if __name__ == "__main__":
    unittest.main()
