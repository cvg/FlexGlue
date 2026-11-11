"""Torch RGB8 contracts; generated inputs, no checkpoints or downloads.

CUDA checks require FLEXGLUE_TEST_RGB8_CUDA=1; eager Torch needs no Triton.
"""
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch
from test_rgb8 import PacketAssertions, forbid_triton, tiny_descriptor, tiny_matcher
from torch.nn import functional as F

from flexglue import FlexGlue
from flexglue.utils import rgb8_torch
from flexglue.utils._rgb8 import (
    fp32_codebook,
    select_resize_samples,
    select_upload_pixels,
)
from flexglue.utils.preprocess import numpy_image_to_torch


def reference(source, table, target_hw, bgr):
    image = table[source.to(torch.int32)].permute(2, 0, 1)[None].contiguous()
    if bgr:
        image = image.flip(1)
    if source.shape[:2] != target_hw:
        image = F.interpolate(
            image, size=target_hw, mode="bilinear", align_corners=False
        )
    return image


def tables(device):
    canonical = torch.from_numpy(fp32_codebook()).to(device)
    signed = torch.linspace(-17, 23, 256, device=device)
    exceptional = signed.clone()
    exceptional[:5] = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf")], device=device
    )
    return canonical, signed, exceptional


def exceptional_neighbours(test, device, compiled):
    table = torch.ones(256, device=device)
    for bits in (0x7F800000, 0x7FC01234, -0x3FEDCC, -0x80000000):
        table[1] = torch.tensor(bits, dtype=torch.int32, device=device).view(
            torch.float32
        )
        for axis in (0, 1):
            source = torch.zeros((25, 35, 3), dtype=torch.uint8, device=device)
            if axis == 0:
                source[3::5] = 1
            else:
                source[:, 3::5] = 1
            with test.subTest(bits=bits, axis=axis, compiled=compiled):
                test.assertPacketEqual(
                    reference(source, table, (5, 7), False),
                    rgb8_torch._resize_rgb8(source, table, (5, 7), False, compiled),
                )


class TorchRGB8PortableTest(PacketAssertions, unittest.TestCase):
    def test_host_selection_shares_the_tensor_lattice(self):
        for sy, sx in ((1, 1), (2, 3), (3, 5), (4, 6), (5, 8), (31, 32)):
            for th, tw in ((1, 1), (5, 7)):
                shape = (th * sy, tw * sx, 3)
                image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
                selected = select_resize_samples(image, (th, tw))
                tensor = select_resize_samples(torch.from_numpy(image), (th, tw))
                np.testing.assert_array_equal(selected, tensor.numpy())
                for table in tables("cpu"):
                    expected = reference(
                        torch.from_numpy(image), table, (th, tw), False
                    )
                    actual = reference(
                        torch.from_numpy(selected.copy()), table, (th, tw), False
                    )
                    self.assertPacketEqual(expected, actual)
        for shape, target in (
            ((9, 13, 3), (5, 7)),
            ((1, 1, 3), (5, 7)),
            ((5, 7, 3), (5, 7)),
            ((10, 21, 3), (5, 7)),
        ):
            image = np.zeros(shape, dtype=np.uint8)
            tensor = torch.from_numpy(image)
            self.assertIs(select_resize_samples(image, target), image)
            self.assertIs(select_resize_samples(tensor, target), tensor)

    def test_large_integer_factors_and_exceptional_neighbours(self):
        for source_hw, target_hw in (
            ((155, 224), (5, 7)),
            ((65535, 7), (1, 7)),
            ((512, 217), (512, 7)),
        ):
            source = (
                torch.arange(source_hw[0] * source_hw[1] * 3)
                .to(torch.uint8)
                .reshape(*source_hw, 3)
            )
            table = tables("cpu")[0]
            self.assertPacketEqual(
                reference(source, table, target_hw, False),
                rgb8_torch._resize_rgb8(source, table, target_hw, False),
            )
        exceptional_neighbours(self, "cpu", False)

    def test_integer_lattice_and_unchanged_grids_are_bitwise(self):
        # Includes odd/even integer factors, a single-pixel axis, same size,
        # upscaling and noninteger scaling. LUTs test zero-weight neighbours.
        for height, width in (
            (1, 1),
            (5, 7),
            (10, 14),
            (15, 21),
            (20, 28),
            (25, 42),
            (30, 35),
            (9, 13),
        ):
            source = (
                torch.arange(height * width * 3)
                .to(torch.uint8)
                .reshape(height, width, 3)
            )
            for table in tables("cpu"):
                for bgr in (False, True):
                    with self.subTest(shape=source.shape, bgr=bgr), forbid_triton():
                        actual = rgb8_torch._resize_rgb8(source, table, (5, 7), bgr)
                        self.assertPacketEqual(
                            reference(source, table, (5, 7), bgr), actual
                        )

    def test_eager_does_not_invoke_compiler(self):
        source = torch.zeros((20, 28, 3), dtype=torch.uint8)
        with forbid_triton(), patch(
            "torch.compile", side_effect=AssertionError("compiler")
        ):
            result = rgb8_torch._resize_rgb8(source, tables("cpu")[0], (5, 7), False)
        self.assertEqual(result.shape, (1, 3, 5, 7))

    def test_compilation_is_lazy_cached_and_errors_propagate(self):
        rgb8_torch._compiled_converter.cache_clear()
        self.addCleanup(rgb8_torch._compiled_converter.cache_clear)
        with patch("torch.compile", return_value=rgb8_torch._convert_rgb8) as compile:
            first = rgb8_torch._compiled_converter()
            self.assertIs(first, rgb8_torch._compiled_converter())
            compile.assert_called_once_with(rgb8_torch._convert_rgb8, dynamic=True)
        rgb8_torch._compiled_converter.cache_clear()
        with patch("torch.compile", side_effect=RuntimeError("compiler unavailable")):
            with self.assertRaisesRegex(RuntimeError, "compiler unavailable"):
                rgb8_torch._compiled_converter()

    def test_upload_layouts_retain_pixels(self):
        rgb = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
        for image in (rgb, rgb[:, :, ::-1], rgb[:, ::-1], rgb[::2, ::2]):
            with self.subTest(strides=image.strides):
                upload, bgr = select_upload_pixels(image)
                self.assertTrue(upload.flags.c_contiguous)
                np.testing.assert_array_equal(
                    upload[:, :, ::-1] if bgr else upload, image
                )
                if image is rgb or image.strides[-1] == -1:
                    self.assertTrue(np.shares_memory(upload, image))

    def test_validation_precedes_cuda_access(self):
        image = np.zeros((7, 9, 3), dtype=np.uint8)
        with patch(
            "torch.cuda.current_device", side_effect=AssertionError("CUDA access")
        ):
            for invalid in (image.astype(np.float32), image[:, :, :2], image[:0]):
                with self.assertRaises(ValueError):
                    rgb8_torch.prepare_rgb8(invalid)
            for grid in ((0, 1), (True, 1), (4097, 1), [5, 7]):
                with self.assertRaises(ValueError):
                    rgb8_torch.prepare_rgb8(image, target_hw=grid)
            with self.assertRaisesRegex(ValueError, "CUDA device"):
                rgb8_torch.prepare_rgb8(image, device="cpu")

    def test_backend_validation_and_default(self):
        self.assertEqual(FlexGlue.default_conf["rgb8_backend"], "torch")
        matcher = tiny_matcher()
        for invalid in (False, "auto", "cuda", 1):
            with self.assertRaisesRegex(ValueError, "rgb8_backend"):
                matcher.resolve_conf({"rgb8_backend": invalid})
            with self.assertRaisesRegex(ValueError, "rgb8_backend"):
                FlexGlue(weights=None, describe=False, rgb8_backend=invalid)
        with self.assertRaisesRegex(ValueError, "backend"):
            tiny_descriptor().describe_rgb8(
                np.zeros((7, 9, 3), dtype=np.uint8), backend="auto"
            )

    def test_all_backend_cpu_fallbacks_and_refiner_routing(self):
        matcher = tiny_matcher()
        image = np.arange(13 * 17 * 3, dtype=np.uint8).reshape(13, 17, 3)
        tensor = numpy_image_to_torch(image)
        with forbid_triton(), torch.no_grad():
            expected = matcher.descriptor({"image": tensor})
            for backend in ("torch", "torch-compile", "triton"):
                with self.subTest(backend=backend), patch.object(
                    rgb8_torch,
                    "prepare_rgb8",
                    side_effect=AssertionError("CUDA preparation"),
                ):
                    self.assertPacketEqual(
                        expected,
                        matcher.descriptor.describe_rgb8(image, backend=backend),
                    )
                    conf = matcher.resolve_conf(
                        dict(
                            describe=True,
                            describe_resize=(32, 32),
                            rgb8_backend=backend,
                        )
                    )
                    packets = matcher.describe_inputs(
                        {"image": image}, {"image": image}, conf
                    )
                    for packet in packets:
                        self.assertPacketEqual(expected["features"], packet["features"])
                    conf.refine = True
                    packets = matcher.describe_inputs(
                        {"image": image}, {"image": image}, conf
                    )
                    for packet in packets:
                        self.assertPacketEqual(tensor, packet["image"])

    def test_explicit_backend_reaches_descriptor(self):
        matcher = tiny_matcher()
        image = np.zeros((13, 17, 3), dtype=np.uint8)
        conf = matcher.resolve_conf(dict(describe=True, rgb8_backend="torch"))
        with torch.no_grad(), patch.object(
            matcher.descriptor, "_forward_rgb8", wraps=matcher.descriptor._forward_rgb8
        ) as describe:
            matcher.describe_inputs({"image": image}, {"image": image}, conf)
        self.assertEqual(describe.call_count, 2)
        self.assertTrue(
            all(call.kwargs["backend"] == "torch" for call in describe.call_args_list)
        )


@unittest.skipUnless(
    os.environ.get("FLEXGLUE_TEST_RGB8_CUDA") == "1",
    "set FLEXGLUE_TEST_RGB8_CUDA=1 for CUDA checks",
)
class TorchRGB8CudaTest(PacketAssertions, unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        # Test geometry families separately rather than exhaust Dynamo's cache
        # and accidentally validate its eager fallback as compiled execution.
        torch.compiler.reset()
        rgb8_torch._compiled_converter.cache_clear()
        self.addCleanup(torch.compiler.reset)
        self.addCleanup(rgb8_torch._compiled_converter.cache_clear)

    def test_large_factors_and_exceptional_neighbours(self):
        for compiled in (False, True):
            for source_hw, target_hw in (
                ((155, 224), (5, 7)),
                ((65535, 7), (1, 7)),
                ((512, 217), (512, 7)),
            ):
                if compiled:
                    torch.compiler.reset()
                source = (
                    torch.arange(source_hw[0] * source_hw[1] * 3, device="cuda")
                    .to(torch.uint8)
                    .reshape(*source_hw, 3)
                )
                table = tables("cuda")[0]
                self.assertPacketEqual(
                    reference(source, table, target_hw, False),
                    rgb8_torch._resize_rgb8(source, table, target_hw, False, compiled),
                )
            if compiled:
                torch.compiler.reset()
            exceptional_neighbours(self, "cuda", compiled)

    def test_conversion_matches_native_with_custom_tables(self):
        for height, width in (
            (5, 7),
            (10, 14),
            (15, 21),
            (20, 28),
            (25, 42),
            (30, 35),
            (9, 13),
            (1, 1),
        ):
            torch.compiler.reset()
            source = (
                torch.arange(height * width * 3, device="cuda")
                .to(torch.uint8)
                .reshape(height, width, 3)
            )
            for table in tables("cuda"):
                for bgr in (False, True):
                    expected = reference(source, table, (5, 7), bgr)
                    for compiled in (False, True):
                        with self.subTest(
                            shape=source.shape, bgr=bgr, compiled=compiled
                        ):
                            self.assertPacketEqual(
                                expected,
                                rgb8_torch._resize_rgb8(
                                    source, table, (5, 7), bgr, compiled
                                ),
                            )

    def test_pixels_features_keypoints_and_metadata(self):
        descriptor = tiny_descriptor((512, 512)).cuda()
        keypoints = torch.tensor([[[4.0, 4.0], [12.0, 9.0]]], device="cuda")
        for height, width in ((512, 512), (513, 769), (2048, 3072)):
            rgb = (
                np.arange(height * width * 3, dtype=np.uint32)
                .astype(np.uint8)
                .reshape(height, width, 3)[:, :, ::-1]
            )
            with torch.inference_mode():
                image = numpy_image_to_torch(rgb).cuda()
                expected = descriptor({"image": image, "keypoints": keypoints})
                expected["keypoints"] = keypoints
                pixels = descriptor.preprocess(image)
                for backend in ("torch", "torch-compile"):
                    with self.subTest(shape=rgb.shape, backend=backend):
                        actual = descriptor.describe_rgb8(
                            rgb, keypoints, backend=backend
                        )
                        self.assertPacketEqual(expected, actual)
                        prepared = rgb8_torch.prepare_rgb8(
                            rgb, compile_conversion=backend == "torch-compile"
                        )
                        self.assertPacketEqual(pixels, prepared.tensor)

    def test_eager_cuda_without_triton_and_deterministic_fallback(self):
        descriptor = tiny_descriptor((512, 512)).cuda()
        rgb = np.zeros((1024, 2048, 3), dtype=np.uint8)
        with forbid_triton(), torch.inference_mode():
            descriptor.describe_rgb8(rgb, backend="torch")
        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with torch.inference_mode(), patch.object(
                rgb8_torch,
                "prepare_rgb8",
                side_effect=AssertionError("accelerated route"),
            ):
                expected = descriptor({"image": numpy_image_to_torch(rgb).cuda()})
                self.assertPacketEqual(
                    expected, descriptor.describe_rgb8(rgb, backend="torch-compile")
                )
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)

    def test_stream_ownership_and_repeated_outputs(self):
        rgb = (
            np.arange(1024 * 2048 * 3, dtype=np.uint32)
            .astype(np.uint8)
            .reshape(1024, 2048, 3)
        )
        expected = F.interpolate(
            numpy_image_to_torch(rgb)[None].cuda(),
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )
        for compiled in (False, True):
            stream = torch.cuda.Stream()
            with torch.inference_mode(), torch.cuda.stream(stream):
                prepared = rgb8_torch.prepare_rgb8(rgb, compile_conversion=compiled)
                prepared.require_current_stream()
                saved = prepared.tensor.clone()
                rgb8_torch.prepare_rgb8(
                    np.bitwise_xor(rgb, 127), compile_conversion=compiled
                )
            with self.assertRaisesRegex(RuntimeError, "producer stream"):
                prepared.require_current_stream()
            torch.cuda.current_stream().wait_stream(stream)
            self.assertPacketEqual(expected, prepared.tensor)
            self.assertPacketEqual(saved, prepared.tensor)

    def test_invalid_codebooks_are_rejected(self):
        rgb = np.zeros((7, 9, 3), dtype=np.uint8)
        for table in (
            torch.zeros(256),
            torch.zeros(255, device="cuda"),
            torch.zeros(256, device="cuda", dtype=torch.float16),
            torch.zeros(512, device="cuda")[::2],
        ):
            with self.assertRaisesRegex(ValueError, "codebook"):
                rgb8_torch.prepare_rgb8(rgb, codebook_device=table)


if __name__ == "__main__":
    unittest.main()
