"""Host lattice selection is an explicit RGB8 opt-in."""
import importlib.util
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch
from test_rgb8 import PacketAssertions, forbid_triton, tiny_descriptor, tiny_matcher
from torch.nn import functional as F

from flexglue import FlexGlue
from flexglue.utils import rgb8_torch
from flexglue.utils.preprocess import numpy_image_to_torch


class HostSamplingPortableTest(PacketAssertions, unittest.TestCase):
    def test_descriptor_forward_routes_default_and_explicit_flag(self):
        descriptor = tiny_descriptor()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch.object(descriptor, "_forward_rgb8", return_value={}) as forward:
            descriptor.describe_rgb8(image, backend="torch")
            self.assertIs(forward.call_args.kwargs["host_sampling"], False)
            descriptor.describe_rgb8(image, backend="torch", host_sampling=True)
            self.assertIs(forward.call_args.kwargs["host_sampling"], True)
            descriptor(
                {"image": image, "rgb8_backend": "torch", "rgb8_host_sampling": True}
            )
            self.assertIs(forward.call_args.kwargs["host_sampling"], True)

    def test_matcher_routes_only_with_rgb8_backend_and_honors_override(self):
        matcher = tiny_matcher()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        other = {
            "features": torch.ones(1, 8, 4, 4),
            "image_size": torch.tensor([[32.0, 32.0]]),
        }
        self.assertIs(FlexGlue.default_conf["rgb8_host_sampling"], False)
        with patch.object(
            matcher.descriptor, "forward", wraps=matcher.descriptor.forward
        ) as call:
            conf = matcher.resolve_conf(
                {"describe": True, "rgb8_backend": "torch", "rgb8_host_sampling": True}
            )
            matcher.describe_inputs({"image": image}, other, conf)
            self.assertIs(call.call_args.args[0]["rgb8_host_sampling"], True)
            conf = matcher.resolve_conf(
                {"describe": True, "rgb8_backend": "torch", "rgb8_host_sampling": False}
            )
            matcher.describe_inputs({"image": image}, other, conf)
            self.assertIs(call.call_args.args[0]["rgb8_host_sampling"], False)
            conf = matcher.resolve_conf({"describe": True, "rgb8_host_sampling": True})
            matcher.describe_inputs({"image": torch.zeros(3, 32, 32)}, other, conf)
            self.assertNotIn("rgb8_host_sampling", call.call_args.args[0])

    def test_cpu_fallback_uses_original_image_and_invalid_flag_rejected(self):
        descriptor = tiny_descriptor()
        image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        with forbid_triton(), torch.inference_mode(), patch.object(
            rgb8_torch, "prepare_rgb8", side_effect=AssertionError("CUDA route")
        ):
            expected = descriptor({"image": numpy_image_to_torch(image)})
            actual = descriptor.describe_rgb8(
                image, backend="torch", host_sampling=True
            )
        self.assertPacketEqual(expected, actual)
        for value in (None, 1, "true"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "host_sampling"
            ):
                descriptor.describe_rgb8(image, backend="torch", host_sampling=value)
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "rgb8_host_sampling"
            ):
                tiny_matcher().resolve_conf({"rgb8_host_sampling": value})


@unittest.skipUnless(
    os.environ.get("FLEXGLUE_TEST_RGB8_CUDA") == "1" and torch.cuda.is_available(),
    "optional CUDA checks",
)
class HostSamplingCudaTest(PacketAssertions, unittest.TestCase):
    def test_default_and_optin_pixels_owners_and_stream(self):
        from flexglue.utils import rgb8_triton

        self.assertIsNotNone(importlib.util.find_spec("triton"))
        bgr = (
            np.arange(1024 * 2048 * 3, dtype=np.uint32)
            .astype(np.uint8)
            .reshape(1024, 2048, 3)
        )
        for image in (bgr.copy(), bgr[:, :, ::-1]):
            expected = F.interpolate(
                numpy_image_to_torch(image)[None].cuda(),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
            for port in (rgb8_torch, rgb8_triton):
                stream = torch.cuda.Stream()
                with self.subTest(
                    port=port.__name__, strides=image.strides
                ), torch.cuda.stream(stream):
                    default = port.prepare_rgb8(image)
                    explicit = port.prepare_rgb8(image, host_sampling=True)
                    default.require_current_stream()
                    explicit.require_current_stream()
                    snapshot = explicit.tensor.clone()
                    port.prepare_rgb8(np.bitwise_xor(image, 127), host_sampling=True)
                    consumed = explicit.tensor.clone()
                    self.assertEqual(default.owners[2].numel(), image.nbytes)
                    self.assertLess(explicit.owners[2].numel(), image.nbytes)
                    self.assertIs(explicit.owners[0], image)
                with self.assertRaisesRegex(RuntimeError, "producer stream"):
                    explicit.require_current_stream()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                self.assertPacketEqual(expected, default.tensor)
                self.assertPacketEqual(expected, explicit.tensor)
                self.assertPacketEqual(snapshot, consumed)
                self.assertPacketEqual(
                    explicit.original_size.cpu(), torch.tensor([[2048.0, 1024.0]])
                )


if __name__ == "__main__":
    unittest.main()
