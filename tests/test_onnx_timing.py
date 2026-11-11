import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class SessionTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript", "onnxruntime"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        path = Path(__file__).resolve().parents[1] / "onnx" / "check.py"
        spec = importlib.util.spec_from_file_location(
            "flexglue_check_timing_test", path
        )
        cls.check = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(path.parent), *sys.path]):
            spec.loader.exec_module(cls.check)

    def test_invalid_counts_refuse_before_running(self):
        for warmup, runs in ((-1, 1), (0, 0), (2, 0), (2, -1)):
            with self.subTest(warmup=warmup, runs=runs):
                session = Mock()
                with self.assertRaises(ValueError):
                    self.check.time_session(session, {}, warmup=warmup, runs=runs)
                session.run.assert_not_called()

    def test_mean_excludes_warmup_and_preserves_feeds(self):
        for warmup in (0, 2):
            with self.subTest(warmup=warmup):
                session, feeds = Mock(), {"input": object()}
                with patch.object(self.check.time, "time", side_effect=(10.0, 10.004)):
                    elapsed = self.check.time_session(
                        session, feeds, warmup=warmup, runs=2
                    )
                self.assertAlmostEqual(elapsed, 2.0)
                self.assertEqual(session.run.call_count, warmup + 2)
                for call in session.run.call_args_list:
                    self.assertIsNone(call.args[0])
                    self.assertIs(call.args[1], feeds)


if __name__ == "__main__":
    unittest.main()
