import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flexglue import FlexGlue


class DescriptorFrameTest(unittest.TestCase):
    def test_to_resized_preserves_frame_precision_for_point_dtypes(self):
        for width in (8192, 8193):
            data = {
                "features": torch.zeros(1, 1, 1, width),
                "image_size": torch.tensor([[65536.0, 8.0]], dtype=torch.float32),
            }
            size = self.model.descriptor_size(data)
            original_size = data["image_size"].clone()
            original_frame = size.clone()
            for dtype in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
                torch.int64,
            ):
                with self.subTest(width=width, dtype=dtype):
                    points = torch.tensor([[[2, 4]]], dtype=dtype)
                    if dtype.is_floating_point:
                        points.requires_grad_()
                    expected = points * (size / data["image_size"])[:, None, :]
                    for supplied in (None, size):
                        actual = self.model.to_resized(points, data, supplied)
                        self.assertEqual(
                            actual.dtype, torch.promote_types(dtype, torch.float32)
                        )
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        if dtype.is_floating_point:
                            actual_grad = torch.autograd.grad(
                                actual.sum(), points, retain_graph=True
                            )[0]
                            expected_grad = torch.autograd.grad(
                                expected.sum(), points, retain_graph=True
                            )[0]
                            torch.testing.assert_close(
                                actual_grad, expected_grad, atol=0, rtol=0
                            )
            self.assertTrue(torch.equal(data["image_size"], original_size))
            self.assertTrue(torch.equal(size, original_frame))

    def test_to_resized_retains_double_division_precision(self):
        data = {
            "features": torch.zeros(1, 1, 1, 8192),
            "image_size": torch.tensor([[65535.0, 7.0]], dtype=torch.float32),
        }
        points = torch.tensor([[[2.0, 4.0]]], dtype=torch.float64)
        frame = self.model.descriptor_size(data)
        expected = points * (frame.double() / data["image_size"])[:, None, :]
        actual = self.model.to_resized(points, data)
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        rounded = points * (frame / data["image_size"])[:, None, :]
        self.assertFalse(torch.equal(actual, rounded))

    def test_geometry_precision_is_independent_of_feature_precision(self):
        points = torch.tensor([[[319.25, 271.125]]], dtype=torch.float64)
        size = torch.tensor([[1536, 1024]])
        expected = points * torch.tensor([32.0 / 1536, 16.0 / 1024], dtype=points.dtype)
        for dtype in (torch.float16, torch.float32, torch.float64):
            data = {
                "features": torch.zeros(1, 4, 2, 4, dtype=dtype),
                "image_size": size,
            }
            for descriptor_size in (None, self.model.descriptor_size(data)):
                actual = self.model.to_resized(points, data, descriptor_size)
                self.assertEqual(actual.dtype, points.dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def setUp(self):
        self.model = FlexGlue(
            weights=None,
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=0,
            mp=None,
        ).eval()
        self.data0 = {
            "features": torch.zeros(1, 4, 2, 4),
            "image_size": torch.tensor([[128, 32]]),
            "keypoints": torch.tensor([[[32.0, 16.0]]]),
        }
        self.data1 = {
            "features": torch.zeros(1, 4, 4, 2),
            "image_size": torch.tensor([[32, 128]]),
            "keypoints": torch.tensor([[[16.0, 32.0]]]),
        }
        self.context = SimpleNamespace(covisibility_score=torch.ones(1))

    def test_describe_inputs_restores_the_overridable_resized_size(self):
        class CustomFrame(FlexGlue):
            def resized_size(self, features, image_size):
                return super().resized_size(features, image_size) + 8

        model = CustomFrame(**vars(self.model.conf)).eval()
        incoming = torch.ones(1, 2)
        data = {
            **self.data0,
            "image_size": self.data0["image_size"].half(),
            "resized_size": incoming,
        }
        a, b = model.describe_inputs(data, self.data1, model.conf)
        torch.testing.assert_close(a["resized_size"], torch.tensor([[40.0, 24.0]]))
        torch.testing.assert_close(b["resized_size"], torch.tensor([[24.0, 40.0]]))
        self.assertIs(data["resized_size"], incoming)
        self.assertNotIn("resized_size", self.data1)

    def test_explicit_geometry_packet_and_two_argument_conversion(self):
        size0, size1 = torch.tensor([[16.0, 8.0]]), torch.tensor([[8.0, 16.0]])
        data0 = {**self.data0, "resized_size": size0}
        data1 = {**self.data1, "resized_size": size1}
        pred = {
            "confidence0": torch.ones(1, 1),
            "confidence1": torch.ones(1, 1),
            "p2d0_i1": data1["keypoints"],
            "p2d1_i0": data0["keypoints"],
        }
        dense = {
            "dense_p2d0_n1": torch.zeros(1, 1, 1, 2),
            "dense_p2d0_i1": data1["keypoints"][:, None],
            "dense_confidence0": torch.ones(1, 1, 1),
        }
        conf = self.model.resolve_conf(
            {"max_distance_px": 0.0, "remove_border_px": 0.0}
        )
        convert = self.model.to_resized
        received = []

        def legacy_convert(points, packet):
            received.append(id(packet))
            return convert(points, packet)

        with patch.object(self.model, "to_resized", new=legacy_convert):
            gated = self.model.gate_points(pred, self.context, data0, data1, conf)
            self.assertTrue(gated["valid0"].item())
            self.assertTrue(gated["valid1"].item())
            sparse = self.model.sparse_matches(gated, data0, data1, conf)
            self.assertEqual(sparse["matches0"].item(), 0)
            grid = self.model.gate_dense(dense, self.context, data0, data1, conf)
            self.assertTrue(grid["dense_valid0"].item())
        a, b = id(data0), id(data1)
        self.assertEqual(received, [a, b, b, a, a, b, b, a, b])

        packet = {"image_size": data0["image_size"], "resized_size": size0}
        actual = convert(data0["keypoints"], packet)
        torch.testing.assert_close(actual, torch.tensor([[[4.0, 4.0]]]))

    def test_public_matching_keeps_two_argument_resize_override(self):
        frames = []

        class LegacyConversion(FlexGlue):
            def to_resized(self, points, data_i):
                frames.append(data_i["resized_size"])
                return super().to_resized(points, data_i)

        model = LegacyConversion(**vars(self.model.conf)).eval()
        model.load_state_dict(self.model.state_dict())
        for name, overrides, conversions in (
            ("match", {"sparse": False}, 4),
            ("match", {"sparse": True}, 8),
            ("match_dense", {}, 2),
        ):
            with self.subTest(name=name, **overrides), torch.no_grad():
                frames.clear()
                conf = {"max_distance_px": 0.0, **overrides}
                expected = getattr(self.model, name)(self.data0, self.data1, **conf)
                actual = getattr(model, name)(self.data0, self.data1, **conf)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(len(frames), conversions)
                self.assertEqual(len({id(size) for size in frames}), 2)

    def test_cached_geometry_and_integer_size_border(self):
        for dtype in (torch.int64, torch.float32):
            data = {**self.data0, "image_size": self.data0["image_size"].to(dtype)}
            self.assertTrue(
                torch.equal(
                    self.model.descriptor_size(data), torch.tensor([[32.0, 16.0]])
                )
            )
            # 0.5 descriptor pixels is two original pixels horizontally.
            data["keypoints"] = torch.tensor([[[1.6, 16.0], [2.4, 16.0]]])
            pred = {
                "confidence0": torch.ones(1, 2),
                "p2d0_i1": torch.tensor([[[16.0, 32.0], [16.0, 32.0]]]),
            }
            conf = SimpleNamespace(
                min_covisibility=0.0, remove_border_px=0.5, max_distance_px=0.0
            )
            result = self.model.gate_points(pred, self.context, data, self.data1, conf)
            self.assertTrue(
                torch.equal(result["valid0"], torch.tensor([[False, True]]))
            )

    def test_cycle_error_uses_descriptor_axes(self):
        # An x offset of one eighth of normalized width is two descriptor pixels.
        points = torch.tensor([[[0.0, 0.0]]])
        target = torch.tensor([[[0.0, 0.0]]])
        context = SimpleNamespace(layer=0, direction=lambda i: (None, None))
        with patch.object(
            self.model,
            "run_head",
            return_value=(torch.tensor([[[0.125, 0.25]]]), torch.ones(1, 1)),
        ):
            error, _ = self.model.cycle_error(
                context, 0, points, target, self.model.descriptor_size(self.data0)
            )
        self.assertTrue(torch.allclose(error, torch.tensor([[2.0**0.5 * 2]])))

    def test_sparse_metric_and_strict_threshold(self):
        pred = {
            "p2d0_i1": torch.tensor([[[20.0, 32.0]]]),
            "p2d1_i0": torch.tensor([[[32.0, 16.0]]]),
            "confidence0": torch.ones(1, 1),
            "confidence1": torch.ones(1, 1),
            "valid0": torch.ones(1, 1, dtype=torch.bool),
            "valid1": torch.ones(1, 1, dtype=torch.bool),
        }
        for gemm, exact in ((True, False), (True, True), (False, False)):
            for threshold, expected in (
                (None, 0),
                (0.0, 0),
                (-1.0, 0),
                (float("nan"), 0),
                (1.0, -1),
                (1.1, 0),
                (torch.tensor(0.0), -1),
                (torch.tensor(-1.0), -1),
                (torch.tensor(float("nan")), -1),
                (torch.tensor(float("inf")), 0),
            ):
                with self.subTest(gemm=gemm, exact=exact, threshold=threshold):
                    conf = SimpleNamespace(
                        gemm_nn=gemm,
                        exact_nn=exact,
                        filter_threshold=0.0,
                        max_distance_px=threshold,
                    )
                    result = self.model.sparse_matches(
                        pred, self.data0, self.data1, conf
                    )
                    self.assertEqual(result["matches0"].item(), expected)
                    self.assertEqual(result["matches1"].item(), expected)
                    for i in (0, 1):
                        self.assertEqual(
                            result[f"matching_scores{i}"].item(), float(expected >= 0)
                        )

    def test_square_frames_with_unequal_image_scales(self):
        data0 = {
            **self.data0,
            "features": torch.zeros(1, 4, 4, 4),
            "image_size": torch.tensor([[128, 128]]),
        }
        data1 = {
            **self.data1,
            "features": torch.zeros(1, 4, 4, 4),
            "image_size": torch.tensor([[32, 32]]),
        }
        pred = {
            "p2d0_i1": data1["keypoints"] + torch.tensor([[[4.0, 0.0]]]),
            "p2d1_i0": data0["keypoints"],
            "confidence0": torch.ones(1, 1),
            "confidence1": torch.ones(1, 1),
            "valid0": torch.ones(1, 1, dtype=torch.bool),
            "valid1": torch.ones(1, 1, dtype=torch.bool),
        }
        # Descriptor mean error 2 px: original-image scale averaging would
        # incorrectly accept it at threshold 2 (original error 2 < scaled 5).
        for gemm in (False, True):
            for threshold, expected in ((2.0, -1), (2.1, 0)):
                conf = SimpleNamespace(
                    gemm_nn=gemm,
                    exact_nn=True,
                    filter_threshold=0.0,
                    max_distance_px=threshold,
                )
                result = self.model.sparse_matches(pred, data0, data1, conf)
                self.assertEqual(result["matches0"].item(), expected)

    def test_dense_border_uses_target_frame(self):
        pred = {
            "dense_p2d0_n1": torch.zeros(1, 1, 2, 2),
            "dense_p2d0_i1": torch.tensor([[[[0.8, 32.0], [1.2, 32.0]]]]),
            "dense_confidence0": torch.ones(1, 1, 2),
        }
        conf = SimpleNamespace(
            min_covisibility=0.0,
            remove_border_px=0.5,
            max_distance_px=0.0,
            filter_threshold=0.0,
        )
        out = self.model.gate_dense(pred, self.context, self.data0, self.data1, conf)
        self.assertTrue(
            torch.equal(out["dense_valid0"], torch.tensor([[[False, True]]]))
        )

    def test_dense_cycle_threshold_uses_each_source_frame(self):
        # Opposite anisotropic frames: both return errors are (2, 2) pixels.
        for threshold, expected in ((2.8, False), (2.9, True)):
            pred = {
                "dense_p2d0_n1": torch.tensor([[[[0.25, 0.125]]]]),
                "dense_p2d1_n0": torch.tensor([[[[0.125, 0.25]]]]),
                "dense_p2d0_i1": self.data1["keypoints"][:, None],
                "dense_p2d1_i0": self.data0["keypoints"][:, None],
                "dense_confidence0": torch.ones(1, 1, 1),
                "dense_confidence1": torch.ones(1, 1, 1),
            }
            conf = SimpleNamespace(
                min_covisibility=0.0,
                remove_border_px=0.0,
                max_distance_px=threshold,
                filter_threshold=0.0,
            )
            out = self.model.gate_dense(
                pred, self.context, self.data0, self.data1, conf
            )
            for i in (0, 1):
                with self.subTest(threshold=threshold, direction=i):
                    self.assertEqual(out[f"dense_valid{i}"].item(), expected)

    def test_bidirectional_gates_reuse_dimensions(self):
        pred = {
            "confidence0": torch.ones(1, 1),
            "confidence1": torch.ones(1, 1),
            "p2d0_i1": self.data1["keypoints"],
            "p2d1_i0": self.data0["keypoints"],
        }
        conf = SimpleNamespace(
            min_covisibility=0.0, remove_border_px=0.0, max_distance_px=0.0
        )
        with patch.object(
            self.model, "descriptor_size", wraps=self.model.descriptor_size
        ) as size:
            self.model.gate_points(pred, self.context, self.data0, self.data1, conf)
            self.assertEqual(size.call_count, 2)
            size.reset_mock()
            data0, data1 = self.model.describe_inputs(self.data0, self.data1, conf)
            self.model.gate_points(pred, self.context, data0, data1, conf)
            self.assertEqual(size.call_count, 2)

    def test_batch_scales_original_sizes_independently(self):
        data = {
            "features": torch.zeros(2, 4, 2, 4),
            "image_size": torch.tensor([[128, 32], [64, 64]]),
        }
        points = torch.tensor([[[64.0, 16.0]], [[32.0, 32.0]]])
        expected = torch.tensor([[[16.0, 8.0]], [[16.0, 8.0]]])
        self.assertTrue(torch.equal(self.model.to_resized(points, data), expected))

    def test_square_uniform_frame_has_original_gate(self):
        data = {
            "features": torch.zeros(1, 4, 4, 4),
            "image_size": torch.tensor([[128.0, 128.0]]),
            "keypoints": torch.tensor([[[8.0, 8.0], [4.0, 4.0]]]),
        }
        pred = {"confidence0": torch.ones(1, 2), "p2d0_i1": data["keypoints"]}
        conf = SimpleNamespace(
            min_covisibility=0.0, remove_border_px=1.0, max_distance_px=0.0
        )
        out = self.model.gate_points(pred, self.context, data, data, conf)
        self.assertTrue(torch.equal(out["valid0"], torch.tensor([[True, False]])))


if __name__ == "__main__":
    unittest.main()
