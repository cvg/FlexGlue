import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from flexglue.descriptor import DINOv3Descriptor
from flexglue.utils import ops


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return [self.conv(x)]


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 1)

    def forward(self, features):
        return self.conv(features[0])


class DescriptorPrecisionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        backbone = TinyBackbone()
        head = TinyHead()
        descriptor_head = nn.Linear(8, 4)
        self.state = {
            **{"extractor.backbone." + k: v for k, v in backbone.state_dict().items()},
            **{"extractor.head." + k: v for k, v in head.state_dict().items()},
            **{
                "extractor.descriptor_head." + k: v
                for k, v in descriptor_head.state_dict().items()
            },
        }

    def build(self, mp=None):
        with patch("flexglue.descriptor.ConvNeXt", TinyBackbone), patch(
            "flexglue.descriptor.LinearConvHead", TinyHead
        ):
            return DINOv3Descriptor(
                weights="unused", state_dict=self.state, resize=(32, 32), mp=mp
            ).eval()

    def test_default_remains_float32(self):
        descriptor = self.build()
        image = torch.rand(1, 3, 32, 32)
        keypoints = torch.tensor([[[8.0, 8.0], [24.0, 24.0]]])
        pred = descriptor.describe(image, keypoints=keypoints)
        self.assertEqual(descriptor.backbone.conv.weight.dtype, torch.float32)
        self.assertEqual(descriptor.head.conv.weight.dtype, torch.float32)
        self.assertEqual(pred["features"].dtype, torch.float32)
        self.assertEqual(pred["descriptors"].dtype, torch.float32)

    def test_resident_half_keeps_normalization_and_sparse_output_float32(self):
        descriptor = self.build("float16")
        image = torch.rand(1, 3, 32, 32)
        keypoints = torch.tensor([[[8.0, 8.0], [24.0, 24.0]]])
        pred = descriptor.describe(image, keypoints=keypoints)
        self.assertEqual(descriptor.backbone.conv.weight.dtype, torch.float16)
        self.assertEqual(descriptor.head.conv.weight.dtype, torch.float16)
        self.assertEqual(descriptor.mean.dtype, torch.float32)
        self.assertEqual(descriptor.std.dtype, torch.float32)
        self.assertEqual(descriptor.descriptor_head.weight.dtype, torch.float32)
        self.assertEqual(pred["features"].dtype, torch.float32)
        self.assertEqual(pred["descriptors"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(pred["descriptors"]).all())

    def test_unknown_precision_rejected(self):
        with self.assertRaisesRegex(ValueError, "descriptor mp"):
            self.build("bfloat16")

    def test_encode_preserves_legacy_precision_path(self):
        image = torch.rand(1, 3, 32, 32)
        for mp in (None, "float16"):
            with self.subTest(mp=mp):
                descriptor = self.build(mp)
                normalized = (image - descriptor.mean) / descriptor.std
                if mp == "float16":
                    normalized = normalized.to(torch.float16)
                expected = descriptor.head(descriptor.backbone(normalized)).float()
                actual = descriptor.encode(descriptor.preprocess(image))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(actual, expected))

    def test_external_autocast_preserves_legacy_encode(self):
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                probe = torch.nn.functional.conv2d(
                    torch.zeros(1, 3, 2, 2), torch.zeros(4, 3, 1, 1)
                )
        except (RuntimeError, NotImplementedError) as exc:
            self.skipTest(f"CPU bfloat16 autocast convolution unavailable: {exc}")
        if probe.dtype != torch.bfloat16:
            self.skipTest("CPU convolution did not activate bfloat16 autocast")

        image = torch.rand(1, 3, 32, 32)
        for mp in (None, "float16"):
            with self.subTest(mp=mp):
                descriptor = self.build(mp)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    normalized = (image - descriptor.mean) / descriptor.std
                    if mp == "float16":
                        normalized = normalized.to(torch.float16)
                    expected = descriptor.head(descriptor.backbone(normalized))
                    if mp == "float16":
                        expected = expected.float()
                    actual = descriptor.encode(image)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(torch.equal(actual, expected))

    def test_precision_follows_module_transforms(self):
        image = torch.rand(1, 3, 32, 32)
        for transform, dtype in (("float", torch.float32), ("double", torch.float64)):
            with self.subTest(transform=transform):
                descriptor = getattr(self.build("float16"), transform)()
                image_for_model = image.to(dtype)
                keypoints = torch.tensor([[[8.0, 8.0], [24.0, 24.0]]], dtype=dtype)
                normalized = (image_for_model - descriptor.mean) / descriptor.std
                expected = descriptor.head(descriptor.backbone(normalized)).to(
                    descriptor.descriptor_head.weight.dtype
                )
                grid = (keypoints / image_for_model.new_tensor([32, 32])) * 2 - 1
                sampled = torch.nn.functional.grid_sample(
                    expected,
                    grid[:, None],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )[:, :, 0].transpose(1, 2)
                sparse = torch.nn.functional.normalize(
                    descriptor.descriptor_head(sampled), dim=-1
                )
                pred = descriptor.describe(image_for_model, keypoints=keypoints)
                self.assertEqual(pred["features"].dtype, dtype)
                self.assertEqual(pred["descriptors"].dtype, dtype)
                self.assertTrue(torch.equal(pred["features"], expected))
                self.assertTrue(torch.equal(pred["descriptors"], sparse))

    def test_device_only_transform_preserves_resident_precision(self):
        descriptor = self.build("float16").to(device="cpu")
        image = torch.rand(1, 3, 32, 32)
        normalized = ((image - descriptor.mean) / descriptor.std).to(
            descriptor.backbone.conv.weight.dtype
        )
        expected = descriptor.head(descriptor.backbone(normalized)).to(
            descriptor.descriptor_head.weight.dtype
        )
        self.assertEqual(descriptor.backbone.conv.weight.dtype, torch.float16)
        self.assertEqual(descriptor.mean.dtype, torch.float32)
        self.assertTrue(torch.equal(descriptor.encode(image), expected))

    def test_half_transform_encode_when_cpu_convolution_supported(self):
        try:
            torch.nn.functional.conv2d(
                torch.zeros(1, 3, 2, 2, dtype=torch.float16),
                torch.zeros(4, 3, 1, 1, dtype=torch.float16),
            )
        except RuntimeError as exc:
            self.skipTest(f"CPU float16 convolution unavailable: {exc}")
        descriptor = self.build("float16").half()
        image = torch.rand(1, 3, 32, 32, dtype=torch.float16)
        normalized = (image - descriptor.mean) / descriptor.std
        expected = descriptor.head(descriptor.backbone(normalized)).to(
            descriptor.descriptor_head.weight.dtype
        )
        actual = descriptor.encode(image)
        self.assertEqual(actual.dtype, torch.float16)
        self.assertTrue(torch.equal(actual, expected))

    def test_forward_encodes_once(self):
        descriptor = self.build()
        image = torch.rand(1, 3, 32, 32)
        with patch.object(descriptor, "encode", wraps=descriptor.encode) as encode:
            descriptor.forward({"image": image})
        encode.assert_called_once()

    def test_caller_supplied_state_remains_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extractor.pth"
            torch.save(self.state, path)
            with patch("flexglue.descriptor.ConvNeXt", TinyBackbone), patch(
                "flexglue.descriptor.LinearConvHead", TinyHead
            ):
                model = DINOv3Descriptor(
                    weights=str(path),
                    state_dict=self.state,
                    resize=(32, 32),
                    mmap_weights=True,
                )
            before = model.backbone.conv.weight.detach().clone()
            with torch.no_grad():
                self.state["extractor.backbone.conv.weight"].add_(1)
            self.assertTrue(torch.equal(model.backbone.conv.weight, before))

    def test_owned_standalone_checkpoint_assigns_exact_submodules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extractor.pth"
            torch.save(self.state, path)
            assigned = []
            can_assign = ops._mapped_local_checkpoint(str(path)) and (
                "assign" in inspect.signature(nn.Module.load_state_dict).parameters
            )
            original = ops.load_into_module

            def observed(*args, **kwargs):
                result = original(*args, **kwargs)
                assigned.append(result[1])
                return result

            with patch("flexglue.descriptor.ConvNeXt", TinyBackbone), patch(
                "flexglue.descriptor.LinearConvHead", TinyHead
            ), patch.object(ops, "load_into_module", side_effect=observed):
                default = DINOv3Descriptor(weights=str(path), resize=(32, 32))
                self.assertEqual(assigned, [False, False, False])
                assigned.clear()
                model = DINOv3Descriptor(
                    weights=str(path), resize=(32, 32), mmap_weights=True
                )
            self.assertEqual(assigned, [can_assign] * 3)
            self.assertTrue(
                torch.equal(default.backbone.conv.weight, model.backbone.conv.weight)
            )
            self.assertTrue(
                torch.equal(
                    model.backbone.conv.weight,
                    self.state["extractor.backbone.conv.weight"],
                )
            )


if __name__ == "__main__":
    unittest.main()
