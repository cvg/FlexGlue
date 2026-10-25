import collections.abc as collections
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

Resize = Union[int, Tuple[int, int]]
Images = Union[torch.Tensor, List[torch.Tensor]]  # [B x] C x H x W, or a list


def batch_images(images: Images) -> List[torch.Tensor]:
    """Images as a list of B x 3 x H x W tensors, gray expanded to RGB."""
    if isinstance(images, torch.Tensor):
        images = [images]
    images = [im[None] if im.dim() == 3 else im for im in images]
    if any(im.dim() != 4 for im in images):
        shapes = [tuple(im.shape) for im in images]
        raise ValueError(f"Expected [B x] C x H x W images, got {shapes}.")
    return [im.expand(-1, 3, -1, -1) for im in images]


def image_sizes(images: Images) -> torch.Tensor:
    """Sizes (w, h) of the images, B x 2."""
    sizes = [  # new_full fills on device, a tensor from a list would sync
        torch.stack([im.new_full((len(im),), s) for s in im.shape[-2:][::-1]], -1)
        for im in batch_images(images)
    ]
    return torch.cat(sizes)


def resize_images(
    images: Images, resize: Resize, edge_divisible_by: Optional[int] = None
) -> torch.Tensor:
    """Resize and stack images to B x 3 x h x w, raise if their new sizes differ."""
    resized = [
        resize_tensor(im, resize, edge_divisible_by) for im in batch_images(images)
    ]
    if len({tuple(im.shape[-2:]) for im in resized}) > 1:
        raise ValueError("Images of different sizes need a (h, w) resize to batch.")
    return torch.cat(resized)


def check_resize(resize: Resize, edge_divisible_by: Optional[int] = None) -> Resize:
    """Raise unless resize is an int (long side) or (h, w), divisible if given."""
    d = edge_divisible_by or 1
    sides = (resize,) if isinstance(resize, int) else tuple(resize)
    valid = isinstance(resize, int) or len(sides) == 2
    if not valid or any(not isinstance(s, int) or s % d for s in sides):
        raise ValueError(
            f"resize must be an int or (h, w), divisible by {d}, got {resize}."
        )
    return resize


def resize_tensor(
    img: torch.Tensor, resize: Resize, edge_divisible_by: Optional[int] = None
) -> torch.Tensor:
    """Bilinear resize of B x C x H x W: int keeps the aspect, (h, w) squashes."""
    h, w = img.shape[-2:]
    d = edge_divisible_by or 1
    if isinstance(check_resize(resize, d), int):  # long side, sides floored to d
        size = tuple(max(d, int(x * resize / max(h, w)) // d * d) for x in (h, w))
    else:
        size = tuple(resize)
    return F.interpolate(img, size, mode="bilinear", align_corners=False)


def map_tensor(input_, func: Callable):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, torch.Tensor):
        return func(input_)
    else:
        return input_


def batch_to_device(batch: dict, device: str = "cpu"):
    """Move batch (dict) to device"""

    def _func(tensor):
        # blocking: a non-blocking device-to-host copy can be read before it lands
        return tensor.to(device=device).detach()

    return map_tensor(batch, _func)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)
