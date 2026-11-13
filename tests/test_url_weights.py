import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flexglue.utils import ops


class UrlWeightLoadingTest(unittest.TestCase):
    url = "https://example.invalid/weights.pth"

    def cached(self, root, value):
        cache = Path(root) / "checkpoints"
        cache.mkdir()
        torch.save(value, cache / "weights.pth")

    def test_cached_tensor_checkpoint_uses_restricted_loading(self):
        with tempfile.TemporaryDirectory() as root:
            self.cached(root, {"matcher.x": torch.ones(2)})
            with patch.object(torch.hub, "get_dir", return_value=root), patch.object(
                torch.hub,
                "download_url_to_file",
                side_effect=AssertionError("cache miss"),
            ), patch.object(torch, "load", wraps=torch.load) as load:
                actual = ops._read_state_dict(self.url)
            torch.testing.assert_close(actual["matcher.x"], torch.ones(2))
            self.assertEqual(load.call_count, 1)
            self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_rejected_global_is_not_retried_unrestricted(self):
        with tempfile.TemporaryDirectory() as root:
            self.cached(root, SimpleNamespace(value=1))
            with patch.object(torch.hub, "get_dir", return_value=root), patch.object(
                torch.hub,
                "download_url_to_file",
                side_effect=AssertionError("cache miss"),
            ), patch.object(torch, "load", wraps=torch.load) as load:
                with self.assertRaises(pickle.UnpicklingError):
                    ops._read_state_dict(self.url)
            self.assertEqual(load.call_count, 1)
            self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_legacy_url_api_refuses_before_loading_but_local_still_works(self):
        entered = []

        def legacy(url, map_location):
            entered.append(url)
            raise AssertionError("unrestricted loader entered")

        with patch.object(torch.hub, "load_state_dict_from_url", legacy):
            with self.assertRaisesRegex(TypeError, "weights_only"):
                ops._read_state_dict(self.url)
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / "local.pth"
                torch.save({"matcher.x": torch.ones(2)}, path)
                torch.testing.assert_close(
                    ops._read_state_dict(str(path))["matcher.x"], torch.ones(2)
                )
        self.assertEqual(entered, [])


if __name__ == "__main__":
    unittest.main()
