import unittest
from unittest import mock

import cv2
import numpy as np
import torch

from flexglue.utils.preprocess import (
    check_resize,
    numpy_image_to_torch,
    resize_image,
    resize_images,
    resize_tensor,
)


def legacy_image_to_torch(image):
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))
    elif image.ndim == 2:
        image = image[None]
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


class ImageConversionTest(unittest.TestCase):
    def assert_legacy_equal(self, image):
        before = image.copy()
        expected = legacy_image_to_torch(image)
        actual = numpy_image_to_torch(image)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        np.testing.assert_array_equal(image, before)
        self.assertFalse(np.shares_memory(actual.numpy(), image))
        return actual

    def test_every_byte_value_gray_and_rgb(self):
        codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
        self.assert_legacy_equal(codes)
        rgb = np.stack((codes, np.flipud(codes), np.fliplr(codes)), axis=-1)
        self.assert_legacy_equal(rgb)
        self.assert_legacy_equal(rgb[..., ::-1])

    def test_strided_readonly_input_has_independent_output(self):
        source = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)
        view = source[1::2, 2::3, ::-1]
        view.setflags(write=False)
        converted = self.assert_legacy_equal(view)
        saved = converted.clone()
        source.fill(0)
        self.assertTrue(torch.equal(converted, saved))
        converted.fill_(0.5)
        self.assertEqual(int(source.sum()), 0)

    def test_empty_and_single_pixel(self):
        self.assert_legacy_equal(np.empty((0, 0), dtype=np.uint8))
        self.assert_legacy_equal(np.array([[255]], dtype=np.uint8))

    def test_non_byte_inputs_keep_legacy_behavior(self):
        for dtype in (np.uint16, np.int16, np.float32, np.float64, np.bool_):
            with self.subTest(dtype=dtype):
                self.assert_legacy_equal(np.arange(36).reshape(3, 4, 3).astype(dtype))

    def test_ndarray_subclass_uses_legacy_division(self):
        class ImageSubclass(np.ndarray):
            divisions = 0

            def __truediv__(self, value):
                type(self).divisions += 1
                return super().__truediv__(value)

        image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3).view(ImageSubclass)
        expected = legacy_image_to_torch(image)
        ImageSubclass.divisions = 0
        actual = numpy_image_to_torch(image)
        self.assertEqual(ImageSubclass.divisions, 1)
        self.assertTrue(torch.equal(actual, expected))

    def test_default_device_context(self):
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        rgb = np.stack((gray, np.flipud(gray), np.fliplr(gray)), axis=-1)[..., ::-1]
        for image in (gray, rgb):
            with self.subTest(shape=image.shape), torch.device("meta"):
                expected = legacy_image_to_torch(image)
                actual = numpy_image_to_torch(image)
                self.assertEqual(actual.device.type, "meta")
                self.assertEqual(actual.device, expected.device)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.stride(), expected.stride())

    def test_empty_resize_and_nonpositive_targets_are_rejected(self):
        for shape in ((0, 0), (0, 3), (3, 0)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    resize_tensor(torch.empty(1, 3, *shape), 32)
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    resize_image(np.empty((*shape, 3), dtype=np.uint8), 32)
        for target in (0, -32, (0, 32), (32, -1)):
            with self.assertRaises(ValueError):
                check_resize(target)
            with self.subTest(image_target=target):
                with self.assertRaisesRegex(ValueError, "positive"):
                    resize_image(np.ones((4, 6, 3), dtype=np.uint8), target)
                with self.assertRaises(ValueError):
                    resize_tensor(torch.ones(1, 3, 4, 6), target)

    def test_scalar_resize_keeps_extreme_aspect_dimensions_positive(self):
        for shape in ((1, 4096), (4096, 1), (1, 4096, 3), (4096, 1, 3)):
            image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
            before = image.copy()
            for target in (1, 512):
                with self.subTest(shape=shape, target=target):
                    actual, scale = resize_image(image, target)
                    height = 1 if shape[0] == 1 else target
                    width = 1 if shape[1] == 1 else target
                    expected = cv2.resize(
                        image, (width, height), interpolation=cv2.INTER_AREA
                    )
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.shape[:2], (height, width))
                    self.assertEqual(scale, (width / shape[1], height / shape[0]))
                    np.testing.assert_array_equal(image, before)

    def test_valid_resize_pixels_and_geometry(self):
        image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        for size, target in ((3, (2, 3)), ((3, 5), (3, 5)), ([3, 5], (3, 5))):
            with self.subTest(size=size):
                actual, scale = resize_image(image, size)
                h, w = target
                expected = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(scale, (w / 6, h / 4))
        tensor = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
        for size, target in ((4, (2, 4)), ((3, 5), (3, 5))):
            with self.subTest(tensor_size=size):
                actual = resize_tensor(tensor, size)
                expected = torch.nn.functional.interpolate(
                    tensor, target, mode="bilinear", align_corners=False
                )
                self.assertTrue(torch.equal(actual, expected))

    def test_invalid_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Not an image"):
            numpy_image_to_torch(np.zeros((2, 2, 2, 2), dtype=np.uint8))


class ResizeImagesTest(unittest.TestCase):
    def test_singleton_matches_cat_values_gradients_and_storage(self):
        for batch in (1, 2):
            for target in ((4, 6), (3, 5)):
                with self.subTest(batch=batch, target=target):
                    base = torch.arange(batch * 3 * 4 * 6, dtype=torch.float32)
                    source = base.reshape(batch, 3, 4, 6).requires_grad_()
                    expected = torch.cat(
                        [
                            torch.nn.functional.interpolate(
                                source, target, mode="bilinear", align_corners=False
                            )
                        ]
                    )
                    with mock.patch("torch.cat", wraps=torch.cat) as cat:
                        actual = resize_images(source, target)
                    cat.assert_not_called()
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertNotEqual(
                        actual.untyped_storage().data_ptr(),
                        source.untyped_storage().data_ptr(),
                    )
                    weights = torch.arange(actual.numel()).reshape(actual.shape)
                    ga = torch.autograd.grad(
                        (actual * weights).sum(), source, retain_graph=True
                    )[0]
                    ge = torch.autograd.grad((expected * weights).sum(), source)[0]
                    self.assertTrue(torch.equal(ga, ge))
                    before = source.detach().clone()
                    actual.detach().fill_(0)
                    self.assertTrue(torch.equal(source.detach(), before))

    def test_list_batching_and_empty_behavior(self):
        images = [torch.ones(3, 4, 6), torch.full((3, 4, 6), 2.0)]
        expected = torch.cat(
            [
                torch.nn.functional.interpolate(
                    x[None], (3, 5), mode="bilinear", align_corners=False
                )
                for x in images
            ]
        )
        with mock.patch("torch.cat", wraps=torch.cat) as cat:
            actual = resize_images(images, (3, 5))
        cat.assert_called_once()
        self.assertTrue(torch.equal(actual, expected))
        with self.assertRaises((RuntimeError, ValueError)):
            resize_images([], (3, 5))
        with self.assertRaisesRegex(ValueError, "different sizes"):
            resize_images([torch.ones(3, 4, 6), torch.ones(3, 6, 4)], 6)

    def test_same_size_still_interpolates_nonfinite_values(self):
        source = torch.tensor(
            [[[[float("inf"), -float("inf")], [float("nan"), -0.0]]]]
        ).expand(1, 3, 2, 2)
        expected = torch.cat(
            [
                torch.nn.functional.interpolate(
                    source, (2, 2), mode="bilinear", align_corners=False
                )
            ]
        )
        with mock.patch(
            "flexglue.utils.preprocess.F.interpolate",
            wraps=torch.nn.functional.interpolate,
        ) as interpolate:
            actual = resize_images([source], (2, 2))
        interpolate.assert_called_once()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertNotEqual(
            actual.untyped_storage().data_ptr(), source.untyped_storage().data_ptr()
        )


if __name__ == "__main__":
    unittest.main()
