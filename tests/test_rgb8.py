"""Portable RGB8 API checks; no checkpoint, download or Triton is needed.

Run with: python -m unittest discover -s tests -p test_rgb8.py
Optional Linux CUDA checks: set FLEXGLUE_TEST_RGB8_CUDA=1 explicitly.
"""
import builtins
import copy
import functools
import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from flexglue import FlexGlue
from flexglue.descriptor import DINOv3Descriptor
from flexglue.utils.preprocess import numpy_image_to_torch


class TinyStride8Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return [torch.nn.functional.avg_pool2d(self.conv(x), 8)]


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 1)

    def forward(self, features):
        return self.conv(features[0])


def tiny_descriptor(resize=(32, 32), descriptor_cls=DINOv3Descriptor):
    # Supply generated weights at the existing reader boundary.
    torch.manual_seed(7)
    backbone, head, descriptor_head = TinyStride8Backbone(), TinyHead(), nn.Linear(8, 4)
    state = {
        **{"extractor.backbone." + k: v for k, v in backbone.state_dict().items()},
        **{"extractor.head." + k: v for k, v in head.state_dict().items()},
        **{
            "extractor.descriptor_head." + k: v
            for k, v in descriptor_head.state_dict().items()
        },
    }
    with patch("flexglue.descriptor.ConvNeXt", TinyStride8Backbone), patch(
        "flexglue.descriptor.LinearConvHead", TinyHead
    ), patch(
        "flexglue.descriptor.ops.load_state_dict",
        return_value={k.removeprefix("extractor."): v for k, v in state.items()},
    ):
        return descriptor_cls(weights="unused", resize=resize).eval()


def tiny_matcher():
    matcher = FlexGlue(
        weights=None,
        describe=False,
        input_dim=8,
        descriptor_dim=8,
        search_dim=4,
        num_heads=2,
        n_layers=1,
        mp=None,
    ).eval()
    matcher.descriptor = tiny_descriptor()
    return matcher


@contextmanager
def forbid_triton():
    original = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "triton"
            or name.startswith("triton.")
            or "rgb8_triton" in (fromlist or ())
        ):
            raise AssertionError("portable CPU path imported optional Triton")
        return original(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=blocked):
        yield


class PacketAssertions:
    def assertPacketEqual(self, left, right):
        self.assertIs(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.stride(), right.stride())
            self.assertTrue(
                torch.equal(
                    left.detach().cpu().reshape(-1).contiguous().view(torch.uint8),
                    right.detach().cpu().reshape(-1).contiguous().view(torch.uint8),
                )
            )
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assertPacketEqual(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertPacketEqual(a, b)
        else:
            self.assertEqual(left, right)


class RGB8PortableTest(PacketAssertions, unittest.TestCase):
    def setUp(self):
        for target in ("torch.load", "torch.hub.load_state_dict_from_url"):
            guard = patch(
                target, side_effect=AssertionError("checkpoint access is forbidden")
            )
            guard.start()
            self.addCleanup(guard.stop)
        self.image = (
            np.arange(13 * 17 * 3, dtype=np.uint32).astype(np.uint8).reshape(13, 17, 3)
        )
        self.descriptor = tiny_descriptor()

    def test_cpu_layouts_match_original_forward_without_triton(self):
        class ImageSubclass(np.ndarray):
            pass

        layouts = [
            self.image.copy(),
            self.image[:, :, ::-1],
            self.image[:, ::-1, :],
            self.image.view(ImageSubclass),
        ]
        with forbid_triton(), torch.no_grad():
            for image in layouts:
                with self.subTest(strides=image.strides, type=type(image).__name__):
                    expected = self.descriptor({"image": numpy_image_to_torch(image)})
                    actual = self.descriptor.describe_rgb8(image)
                    self.assertPacketEqual(expected, actual)
                    self.assertTrue(
                        torch.equal(actual["image_size"], torch.tensor([[17.0, 13.0]]))
                    )

    def test_keypoints_keep_original_descriptor_values(self):
        keypoints = torch.tensor([[[4.0, 4.0], [12.0, 9.0]]])
        expected = self.descriptor(
            {"image": numpy_image_to_torch(self.image), "keypoints": keypoints}
        )
        expected["keypoints"] = keypoints
        with forbid_triton():
            self.assertPacketEqual(
                expected, self.descriptor.describe_rgb8(self.image, keypoints)
            )

    def test_training_and_grad_fallback_construct_real_autograd_graph(self):
        self.descriptor.train()
        with forbid_triton(), torch.inference_mode(False), torch.enable_grad():
            result = self.descriptor.describe_rgb8(self.image)
            self.assertTrue(result["features"].requires_grad)
            self.assertIsNotNone(result["features"].grad_fn)
            self.assertFalse(torch.is_inference(result["features"]))
            result["features"].sum().backward()
        self.assertIsNotNone(self.descriptor.backbone.conv.weight.grad)
        self.assertTrue(torch.isfinite(self.descriptor.backbone.conv.weight.grad).all())

    def test_compilation_preserves_copied_descriptor_weights(self):
        matcher = tiny_matcher()
        real_compile = torch.compile
        with patch(
            "torch.compile",
            side_effect=lambda fn, **kw: real_compile(
                fn, backend="eager", fullgraph=True
            ),
        ):
            matcher.compile()
        descriptor = matcher.descriptor
        copied = copy.deepcopy(descriptor)
        image = numpy_image_to_torch(self.image)
        with torch.no_grad():
            before = descriptor.describe(image)["features"].clone()
            torch.testing.assert_close(copied.describe(image)["features"], before)
            copied.backbone.conv.weight.add_(1)
            self.assertFalse(torch.equal(copied.describe(image)["features"], before))
            self.assertPacketEqual(
                copied.describe(image)["features"],
                copied.describe_rgb8(self.image)["features"],
            )
            torch.testing.assert_close(descriptor.describe(image)["features"], before)
            torch.testing.assert_close(
                descriptor.describe_rgb8(self.image)["features"], before
            )

    def test_compilation_keeps_overrides_and_repeated_compilation(self):
        observed = []

        class ShiftedDescriptor(DINOv3Descriptor):
            def forward(self, data, resize=None):
                observed.append(type(data["image"]))
                result = super().forward(data, resize)
                return {**result, "features": result["features"] + 1}

        image = numpy_image_to_torch(self.image)
        real_compile = torch.compile
        for override in ("subclass", "instance"):
            matcher = tiny_matcher()
            matcher.descriptor = tiny_descriptor(descriptor_cls=ShiftedDescriptor)
            if override == "instance":
                forward = matcher.descriptor.forward
                matcher.descriptor.forward = lambda *args, **kw: forward(*args, **kw)
            with self.subTest(override=override), torch.no_grad():
                expected = matcher.descriptor({"image": image})
                self.assertPacketEqual(
                    expected["features"], matcher.descriptor.describe(image)["features"]
                )
                with patch(
                    "torch.compile",
                    side_effect=lambda fn, **kw: real_compile(
                        fn, backend="eager", fullgraph=True
                    ),
                ):
                    for _ in range(2):
                        matcher.compile()
                        observed.clear()
                        self.assertPacketEqual(
                            expected, matcher.descriptor.describe_rgb8(self.image)
                        )
                        self.assertEqual(observed, [np.ndarray])
                        conf = matcher.resolve_conf(
                            dict(
                                describe=True,
                                describe_resize=(32, 32),
                                rgb8_backend="torch",
                            )
                        )
                        actual, _ = matcher.describe_inputs(
                            {"image": self.image}, expected, conf
                        )
                        self.assertEqual(observed, [np.ndarray, np.ndarray])
                        self.assertPacketEqual(expected["features"], actual["features"])

    def test_module_dtype_transforms_follow_tensor_reference(self):
        image = numpy_image_to_torch(self.image)
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                self.descriptor.to(dtype=dtype)
                with torch.no_grad():
                    expected = self.descriptor.describe(image.to(dtype=dtype))
                    actual = self.descriptor.describe_rgb8(self.image)
                self.assertPacketEqual(expected["features"], actual["features"])
                self.assertPacketEqual(expected["image_size"], actual["image_size"])

    def test_module_half_uses_typed_tensor_reference(self):
        self.descriptor.half()
        image = numpy_image_to_torch(self.image).half()
        with torch.no_grad():
            expected = self.descriptor.describe(image)
            actual = self.descriptor.describe_rgb8(self.image)
        self.assertGreater(expected["features"].numel(), 0)
        self.assertPacketEqual(expected["features"], actual["features"])
        self.assertPacketEqual(expected["image_size"], actual["image_size"])

    def test_module_dtypes_with_cpu_autocast_follow_typed_tensor_reference(self):
        image = numpy_image_to_torch(self.image)
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            self.descriptor.to(dtype=dtype)
            for enabled in (False, True):
                with self.subTest(dtype=dtype, autocast=enabled), torch.no_grad():
                    autocast_dtype = (
                        torch.float16 if dtype == torch.float16 else torch.bfloat16
                    )
                    with torch.autocast("cpu", dtype=autocast_dtype, enabled=enabled):
                        expected = self.descriptor.describe(image.to(dtype=dtype))
                        actual = self.descriptor.describe_rgb8(self.image)
                self.assertGreater(expected["features"].numel(), 0)
                self.assertPacketEqual(expected["features"], actual["features"])
                self.assertPacketEqual(expected["image_size"], actual["image_size"])

    def test_prepared_input_keeps_resize_and_forward_dispatch(self):
        from flexglue.utils._rgb8 import PreparedDescriptorInput, describe_prepared

        descriptor = tiny_descriptor(resize=(256, 256))
        image = numpy_image_to_torch(self.image)[None]
        pixels = descriptor.preprocess(image, resize=(512, 512))
        original_size = torch.tensor([[image.shape[-1], image.shape[-2]]]).float()
        prepared = PreparedDescriptorInput(pixels, original_size, (pixels,), None)
        calls = []

        def hook(module, args, result):
            calls.append(1)
            return {**result, "features": result["features"] + 1}

        handle = descriptor.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                expected = descriptor.describe(image, resize=(512, 512))
                hooked = descriptor({"image": image}, resize=(512, 512))
                self.assertEqual(calls, [1])
                self.assertFalse(torch.equal(hooked["features"], expected["features"]))
                with patch.object(
                    PreparedDescriptorInput, "require_current_stream"
                ) as stream:
                    actual = describe_prepared(descriptor, prepared)
                fallback = descriptor.describe_rgb8(self.image, resize=(512, 512))
            stream.assert_called_once()
            self.assertEqual(calls, [1])
            self.assertEqual(actual["features"].shape[-2:], (64, 64))
            self.assertPacketEqual(expected["features"], actual["features"])
            self.assertPacketEqual(expected["features"], fallback["features"])
            self.assertPacketEqual(original_size, actual["image_size"])
        finally:
            handle.remove()

    def test_invalid_direct_inputs_are_rejected(self):
        for image in (
            self.image.astype(np.float32),
            self.image[:, :, 0],
            self.image[:, :, :2],
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaisesRegex(ValueError, "HWC RGB8"):
                    self.descriptor.describe_rgb8(image)

    def test_codebook_is_derived_nonpersistent_and_invalidates_on_transforms(self):
        for transform in (
            lambda d: d.to("cpu"),
            lambda d: d.to(dtype=torch.float64),
            lambda d: d.to(dtype=torch.float32),
            lambda d: d.half(),
        ):
            self.descriptor._rgb8_codebook = torch.arange(256, dtype=torch.float32)
            self.assertFalse(
                any("_rgb8_codebook" in key for key in self.descriptor.state_dict())
            )
            self.assertIs(transform(self.descriptor), self.descriptor)
            self.assertIsNone(self.descriptor._rgb8_codebook)
        self.descriptor._rgb8_codebook = torch.arange(256, dtype=torch.float32)

        def fail(tensor):
            raise RuntimeError("test transform failure")

        with self.assertRaisesRegex(RuntimeError, "test transform failure"):
            self.descriptor._apply(fail)
        self.assertIsNone(self.descriptor._rgb8_codebook)

    def test_default_torch_and_backend_validation(self):
        self.assertEqual(FlexGlue.default_conf["rgb8_backend"], "torch")
        matcher = tiny_matcher()
        self.assertEqual(matcher.conf.rgb8_backend, "torch")
        for value in (True, 1, "unknown"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "rgb8_backend"
            ):
                matcher.resolve_conf({"rgb8_backend": value})

    def test_default_rgb8_is_eager_torch_without_triton(self):
        matcher = tiny_matcher()
        conf = matcher.resolve_conf({"describe": True})
        for backend in (None, "torch-compile", "triton"):
            self.assertEqual(
                matcher.resolve_conf({"rgb8_backend": backend}).rgb8_backend, backend
            )
        with forbid_triton(), patch.object(
            self.descriptor, "_forward_rgb8", wraps=self.descriptor._forward_rgb8
        ) as direct, patch.object(torch, "compile") as compile_model, torch.no_grad():
            self.descriptor.describe_rgb8(self.image)
            self.assertEqual(direct.call_args.kwargs["backend"], "torch")
            with patch.object(
                matcher.descriptor,
                "_forward_rgb8",
                wraps=matcher.descriptor._forward_rgb8,
            ) as selected:
                matcher.describe_inputs(
                    {"image": self.image},
                    {
                        "features": torch.ones(1, 8, 4, 4),
                        "image_size": torch.ones(1, 2),
                    },
                    conf,
                )
                self.assertEqual(selected.call_args.kwargs["backend"], "torch")
            compile_model.assert_not_called()

    def test_matcher_byte_inputs_keep_descriptor_module_hooks(self):
        matcher = tiny_matcher()
        tensor = numpy_image_to_torch(self.image)
        calls = []

        def hook(module, args, result):
            calls.append(type(args[0]["image"]))
            return {**result, "features": result["features"] + 1}

        handle = matcher.descriptor.register_forward_hook(hook)
        try:
            conf = matcher.resolve_conf(dict(describe=True, rgb8_backend="torch"))
            with torch.no_grad():
                expected, _ = matcher.describe_inputs(
                    {"image": tensor},
                    {
                        "features": torch.ones(1, 8, 4, 4),
                        "image_size": torch.tensor([[32.0, 32.0]]),
                    },
                    conf,
                )
                actual, _ = matcher.describe_inputs(
                    {"image": self.image},
                    {
                        "features": torch.ones(1, 8, 4, 4),
                        "image_size": torch.tensor([[32.0, 32.0]]),
                    },
                    conf,
                )
            self.assertEqual(calls, [torch.Tensor, np.ndarray])
            for key in ("features", "image_size", "resized_size"):
                self.assertPacketEqual(expected[key], actual[key])
        finally:
            handle.remove()

    def test_byte_detector_receives_one_normalized_chw_image(self):
        matcher = tiny_matcher()
        images = (self.image, self.image[:, :, ::-1])
        observed = []

        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("anchor", torch.zeros(1))

            def extract(self, image):
                observed.append(image.clone())
                return {"keypoints": image.new_tensor([[[4.0, 5.0]]])}

        # Detector placement is independent of matcher placement.
        matcher.to("meta")
        with patch.object(matcher, "match", side_effect=lambda *data, **kw: data):
            data = matcher.match_pair(
                *images, detector=Detector(), rgb8_backend="torch"
            )
        self.assertEqual(len(observed), 2)
        for image, pixels, packet in zip(images, observed, data):
            self.assertEqual(pixels.shape, (3, *image.shape[:2]))
            self.assertPacketEqual(numpy_image_to_torch(image), pixels)
            self.assertIs(packet["image"], image)
            self.assertEqual(packet["keypoints"].shape, (1, 1, 2))

    def test_actual_cpu_match_backends_match_torch_and_feature_packets(self):
        matcher = tiny_matcher()
        images = [self.image, self.image[:, :, ::-1]]
        tensors = [numpy_image_to_torch(image) for image in images]
        policy = dict(
            describe=True,
            describe_resize=(32, 32),
            max_num_samples=4,
            min_covisibility=0.0,
            early_stop_decay=None,
        )

        def run(data, **flags):
            torch.manual_seed(42)
            output = matcher.match(*data, **policy, **flags)
            return output, torch.get_rng_state()

        with forbid_triton(), torch.no_grad():
            reference, rng = run([{"image": tensor} for tensor in tensors])
            actual, actual_rng = run(
                [{"image": image} for image in images], rgb8_backend="triton"
            )
            self.assertEqual(len(reference), 16)
            self.assertPacketEqual(reference, actual)
            self.assertTrue(torch.equal(rng, actual_rng))
            for flags in ({}, {"rgb8_backend": None}):
                default, default_rng = run(
                    [{"image": image} for image in images], **flags
                )
                self.assertPacketEqual(reference, default)
                self.assertTrue(torch.equal(rng, default_rng))
            packets = [
                matcher.descriptor({"image": tensor}, resize=(32, 32))
                for tensor in tensors
            ]
            torch.manual_seed(42)
            expected = matcher.match(
                *packets,
                describe=False,
                max_num_samples=4,
                min_covisibility=0.0,
                early_stop_decay=None,
            )
            torch.manual_seed(42)
            candidate = matcher.match(
                *packets,
                describe=False,
                rgb8_backend="triton",
                max_num_samples=4,
                min_covisibility=0.0,
                early_stop_decay=None,
            )
            self.assertPacketEqual(expected, candidate)

    def test_refiner_description_retains_torch_image(self):
        matcher = tiny_matcher()
        conf = matcher.resolve_conf(
            {"describe": True, "describe_resize": (32, 32), "rgb8_backend": "triton"}
        )
        conf.refine = True  # routing only; no refiner weights or downloads
        with forbid_triton(), torch.no_grad():
            packets = matcher.describe_inputs(
                {"image": self.image}, {"image": self.image}, conf
            )
        for packet in packets:
            self.assertIsInstance(packet["image"], torch.Tensor)
            self.assertPacketEqual(packet["image"], numpy_image_to_torch(self.image))

    def test_cached_rgb8_refinement_normalizes_only_consumed_images(self):
        from flexglue.utils.preprocess import resize_images

        class Extractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def extract(self, image, resize):
                self.calls.append(image)
                return {1: resize_images(image, resize)}

        for descriptor_present in (True, False):
            matcher = tiny_matcher()
            if not descriptor_present:
                matcher.descriptor = None
            matcher.refiner = Extractor()
            expected = resize_images(numpy_image_to_torch(self.image), (32, 32))
            cached_fine = {1: expected}
            for fine_cache in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                for backend in (None, "torch", "torch-compile", "triton"):
                    with self.subTest(
                        descriptor=descriptor_present, fine=fine_cache, backend=backend
                    ), torch.no_grad():
                        matcher.refiner.calls.clear()
                        packets = [
                            {
                                "features": torch.ones(1, 8, 4, 4),
                                "image_size": torch.tensor([[17.0, 13.0]]),
                                "image": self.image,
                                **({"fine_features": cached_fine} if cached else {}),
                            }
                            for cached in fine_cache
                        ]
                        conf = matcher.resolve_conf(
                            dict(
                                describe=False,
                                refine=True,
                                refine_resize=(32, 32),
                                rgb8_backend=backend,
                                early_stop_decay=None,
                            )
                        )
                        described = matcher.describe_inputs(*packets, conf)
                        context = matcher.decode_pair(*described, conf)
                        self.assertEqual(
                            len(matcher.refiner.calls), 2 - sum(fine_cache)
                        )
                        for i, cached in enumerate(fine_cache):
                            self.assertIs(packets[i]["image"], self.image)
                            self.assertPacketEqual(context.fine[i][1], expected)
                            if cached:
                                self.assertIs(context.fine[i], cached_fine)


@unittest.skipUnless(
    os.environ.get("FLEXGLUE_TEST_RGB8_CUDA") == "1",
    "set FLEXGLUE_TEST_RGB8_CUDA=1 for optional Linux CUDA checks",
)
class RGB8CudaTest(PacketAssertions, unittest.TestCase):
    def test_custom_preprocess_uses_original_pixels(self):
        from flexglue.utils import rgb8_torch, rgb8_triton

        stock_preprocess = DINOv3Descriptor.preprocess

        def nonlinear_preprocess(descriptor, images, resize=None):
            return stock_preprocess(descriptor, images.square(), resize)

        class ContrastDescriptor(DINOv3Descriptor):
            preprocess = nonlinear_preprocess

        rgb = np.random.default_rng(19).integers(0, 256, (513, 769, 3), dtype=np.uint8)
        for kind in ("subclass", "instance", "class"):
            descriptor = tiny_descriptor(
                (512, 512),
                descriptor_cls=(
                    ContrastDescriptor if kind == "subclass" else DINOv3Descriptor
                ),
            ).cuda()
            if kind == "instance":
                descriptor.preprocess = nonlinear_preprocess.__get__(descriptor)
            override = (
                patch.object(DINOv3Descriptor, "preprocess", nonlinear_preprocess)
                if kind == "class"
                else nullcontext()
            )
            with override, torch.inference_mode():
                expected = descriptor({"image": numpy_image_to_torch(rgb).cuda()})
                for backend in ("torch", "torch-compile", "triton"):
                    with self.subTest(kind=kind, backend=backend), patch.object(
                        rgb8_torch,
                        "prepare_rgb8",
                        side_effect=AssertionError("fast path"),
                    ), patch.object(
                        rgb8_triton,
                        "prepare_rgb8",
                        side_effect=AssertionError("fast path"),
                    ):
                        actual = descriptor.describe_rgb8(rgb, backend=backend)
                    self.assertPacketEqual(expected["features"], actual["features"])
                    self.assertPacketEqual(expected["image_size"], actual["image_size"])
                    self.assertIsNone(descriptor._rgb8_codebook)

    def setUp(self):
        if not sys.platform.startswith("linux") or not torch.cuda.is_available():
            self.skipTest("Linux CUDA is required for the optional Triton path")
        if importlib.util.find_spec("triton") is None:
            self.skipTest("optional Triton is not installed")
        self.descriptor = tiny_descriptor((512, 512)).cuda()

    def test_accelerated_pixels_features_and_original_size(self):
        from flexglue.utils import rgb8_triton

        for height, width in ((512, 512), (513, 769)):
            self.descriptor = tiny_descriptor((512, 512)).cuda()
            rgb = (
                np.arange(height * width * 3, dtype=np.uint32)
                .astype(np.uint8)
                .reshape(height, width, 3)[:, :, ::-1]
            )
            with self.subTest(shape=rgb.shape), torch.inference_mode():
                image = numpy_image_to_torch(rgb).cuda()
                expected_pixels = self.descriptor.preprocess(image, (512, 512))
                actual = self.descriptor.describe_rgb8(rgb)
                prepared = rgb8_triton.prepare_rgb8(
                    rgb,
                    codebook_device=self.descriptor._rgb8_codebook,
                )
                self.assertPacketEqual(expected_pixels, prepared.tensor)
                self.assertPacketEqual(self.descriptor({"image": image}), actual)
                torch.cuda.synchronize()

    def test_accelerated_module_dtype_follows_tensor_reference(self):
        from flexglue.utils import rgb8_triton

        rgb = np.arange(512 * 512 * 3, dtype=np.uint32).astype(np.uint8)
        rgb = rgb.reshape(512, 512, 3)
        tensor = numpy_image_to_torch(rgb).cuda()
        self.descriptor.conf.resize = (512, 512)
        for dtype in (torch.float64, torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype), torch.inference_mode():
                self.descriptor.to(dtype=dtype)
                expected = self.descriptor.describe(tensor.to(dtype=dtype))
                with patch.object(
                    rgb8_triton, "prepare_rgb8", wraps=rgb8_triton.prepare_rgb8
                ) as prepare:
                    actual = self.descriptor.describe_rgb8(rgb, backend="triton")
                if dtype == torch.float32:
                    prepare.assert_called_once()
                else:
                    prepare.assert_not_called()
                self.assertPacketEqual(expected["features"], actual["features"])
                self.assertPacketEqual(expected["image_size"], actual["image_size"])

    def test_byte_detector_with_independent_device_matches(self):
        matcher = tiny_matcher().cuda()
        rgb = np.arange(32 * 32 * 3, dtype=np.uint32).astype(np.uint8)
        rgb = rgb.reshape(32, 32, 3)
        observed = []

        class Detector(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.register_buffer("anchor", torch.zeros(1, device=device))

            def extract(self, image):
                observed.append(image.device)
                if image.device != self.anchor.device:
                    raise AssertionError("detector input device differs")
                return {
                    "keypoints": self.anchor.new_tensor([[[4.0, 4.0], [16.0, 16.0]]])
                }

        outputs = []
        with torch.inference_mode():
            for device in ("cuda", "cpu"):
                torch.manual_seed(42)
                outputs.append(
                    matcher.match_pair(
                        rgb,
                        rgb,
                        detector=Detector(device),
                        describe=True,
                        rgb8_backend="torch",
                        describe_resize=(32, 32),
                        min_covisibility=0.0,
                        early_stop_decay=None,
                    )
                )
        self.assertEqual(
            observed, [torch.device("cuda:0")] * 2 + [torch.device("cpu")] * 2
        )
        self.assertGreater(len(outputs[0]), 0)
        self.assertPacketEqual(*outputs)

    def test_matcher_hook_wraps_accelerated_byte_input(self):
        from flexglue.utils import rgb8_torch

        matcher = tiny_matcher().cuda()
        rgb = np.arange(512 * 512 * 3, dtype=np.uint32).astype(np.uint8)
        rgb = rgb.reshape(512, 512, 3)
        calls = []

        def hook(module, args, result):
            calls.append(type(args[0]["image"]))
            return {**result, "features": result["features"] + 1}

        handle = matcher.descriptor.register_forward_hook(hook)
        try:
            conf = matcher.resolve_conf(
                dict(describe=True, describe_resize=(512, 512), rgb8_backend="torch")
            )
            with torch.inference_mode():
                tensor = {"image": numpy_image_to_torch(rgb).cuda()}
                expected = matcher.describe_inputs(tensor, tensor, conf)
                with patch.object(
                    rgb8_torch, "prepare_rgb8", wraps=rgb8_torch.prepare_rgb8
                ) as prepare:
                    actual = matcher.describe_inputs(
                        {"image": rgb}, {"image": rgb}, conf
                    )
                self.assertEqual(prepare.call_count, 2)
            self.assertEqual(calls, [torch.Tensor] * 2 + [np.ndarray] * 2)
            for left, right in zip(expected, actual):
                for key in ("features", "image_size", "resized_size"):
                    self.assertPacketEqual(left[key], right[key])
        finally:
            handle.remove()

    def test_accelerated_resize_override(self):
        rgb = np.arange(320 * 480 * 3, dtype=np.uint32).astype(np.uint8)
        rgb = rgb.reshape(320, 480, 3)
        descriptor = tiny_descriptor(resize=(256, 256)).cuda()
        tensor = numpy_image_to_torch(rgb).cuda()
        with torch.inference_mode():
            expected = descriptor.describe(tensor, resize=(512, 512))
            for backend in ("torch", "triton"):
                with self.subTest(backend=backend):
                    actual = descriptor.describe_rgb8(
                        rgb, resize=(512, 512), backend=backend
                    )
                    self.assertEqual(actual["features"].shape[-2:], (64, 64))
                    self.assertPacketEqual(expected["features"], actual["features"])
                    self.assertPacketEqual(expected["image_size"], actual["image_size"])

    def test_descriptor_compilation_covers_both_input_routes(self):
        from flexglue.utils import rgb8_triton

        matcher = tiny_matcher().cuda()
        descriptor = matcher.descriptor
        descriptor.conf.resize = (512, 512)
        encoding = compiling = 0
        compiled_encoding = False
        original_encode = descriptor.encode

        def observe_encode(receiver, *args, **kwargs):
            nonlocal encoding, compiled_encoding
            compiled_encoding |= compiling > 0
            encoding += 1
            try:
                return original_encode(*args, **kwargs)
            finally:
                encoding -= 1

        descriptor.encode = observe_encode.__get__(descriptor)

        def track_compile(fn, **kwargs):
            @functools.wraps(fn)
            def run(*args, **kw):
                nonlocal compiling, compiled_encoding
                compiled_encoding |= encoding > 0
                compiling += 1
                try:
                    return fn(*args, **kw)
                finally:
                    compiling -= 1

            return run

        with patch("torch.compile", side_effect=track_compile):
            matcher.compile()
        rgb = np.arange(513 * 769 * 3, dtype=np.uint32).astype(np.uint8)
        rgb = rgb.reshape(513, 769, 3)
        with torch.inference_mode():
            descriptor.describe(numpy_image_to_torch(rgb).cuda())
            tensor_compiled = compiled_encoding
            compiled_encoding = False
            with patch.object(
                rgb8_triton, "prepare_rgb8", wraps=rgb8_triton.prepare_rgb8
            ) as prepare:
                descriptor.describe_rgb8(rgb, backend="triton")
            prepare.assert_called_once()
            # Both routes must enter the shared compiled tensor forward.
            self.assertTrue(tensor_compiled)
            self.assertTrue(compiled_encoding)

    def test_cached_features_survive_subsequent_images(self):
        matcher = tiny_matcher().cuda().compile()
        descriptor = matcher.descriptor
        descriptor.conf.resize = (512, 512)
        rgb = np.arange(513 * 769 * 3, dtype=np.uint32).astype(np.uint8)
        rgb = rgb.reshape(513, 769, 3)
        with torch.inference_mode():
            first = descriptor.describe_rgb8(rgb)["features"]
            saved = first.clone()
            for later in (rgb[:, ::-1, :], np.bitwise_xor(rgb, 127)):
                torch.compiler.cudagraph_mark_step_begin()
                descriptor.describe_rgb8(later)
                self.assertPacketEqual(first, saved)

    def test_disabled_sparse_description_preserves_keypoints(self):
        from flexglue.utils import rgb8_triton

        self.descriptor.conf.describe = False
        rgb = np.arange(512 * 512 * 3, dtype=np.uint32).astype(np.uint8)
        rgb = rgb.reshape(512, 512, 3)
        keypoints = torch.tensor([[[4.0, 4.0], [12.0, 9.0]]], device="cuda")
        with torch.inference_mode(), patch.object(
            self.descriptor, "sample_descriptors", side_effect=AssertionError("sampled")
        ), patch.object(
            rgb8_triton, "prepare_rgb8", wraps=rgb8_triton.prepare_rgb8
        ) as prepare:
            image = numpy_image_to_torch(rgb).cuda()
            expected = self.descriptor({"image": image, "keypoints": keypoints})
            expected["keypoints"] = keypoints
            actual = self.descriptor.describe_rgb8(rgb, keypoints, backend="triton")
            prepare.assert_called_once()
            self.assertPacketEqual(expected, actual)

    def test_deterministic_mode_uses_reference_preprocess(self):
        from flexglue.utils import rgb8_triton

        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            descriptor = tiny_descriptor((512, 512)).cuda()
            rgb = np.arange(513 * 769 * 3, dtype=np.uint32).astype(np.uint8)
            rgb = rgb.reshape(513, 769, 3)[:, :, ::-1]
            with torch.inference_mode(), patch.object(
                rgb8_triton, "prepare_rgb8", side_effect=AssertionError("fast path")
            ):
                image = numpy_image_to_torch(rgb).cuda()
                expected = descriptor({"image": image})
                actual = descriptor.describe_rgb8(rgb, backend="triton")
                self.assertPacketEqual(expected, actual)
                self.assertIsNone(descriptor._rgb8_codebook)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)


if __name__ == "__main__":
    unittest.main()
