import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .utils import ops
from .utils import preprocess as pp

WEIGHTS_URL = "https://github.com/cvg/FlexGlue/releases/download/v0.1/flexglue.pth"


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of a [B x C x H x W] tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.to_chw(super().forward(ops.to_hwc(x)))


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(ops.to_hwc(self.dwconv(x)))
        y = self.gamma * self.pwconv2(self.act(self.pwconv1(y)))
        return x + ops.to_chw(y)


class ConvNeXt(nn.Module):
    """DINOv3 ConvNeXt backbone, returns the unnormalized features of all 4 stages."""

    def __init__(self, depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024)) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, dims[0], 4, stride=4), LayerNorm2d(dims[0], 1e-6)
                )
            ]
            + [
                nn.Sequential(
                    LayerNorm2d(dims[i], 1e-6), nn.Conv2d(dims[i], dims[i + 1], 2, 2)
                )
                for i in range(3)
            ]
        )
        self.stages = nn.ModuleList(
            nn.Sequential(*[ConvNeXtBlock(d) for _ in range(n)])
            for n, d in zip(depths, dims)
        )

    def forward(self, x: torch.Tensor) -> list:
        features = []
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            features.append(x)
        return features


class LinearConvHead(nn.Module):
    """Stride-8 features: unshuffled conv stem on stride 4 plus linear strides 8-32."""

    def __init__(self, in_dims=(128, 256, 512, 1024), dim=512, out_dim=256) -> None:
        super().__init__()
        fine_in, *coarse_in = in_dims
        c4 = dim // 4
        self.stem = nn.Sequential(
            LayerNorm2d(fine_in),
            nn.Conv2d(fine_in, c4, 3, padding=1),
            LayerNorm2d(c4),
            nn.GELU(),
        )
        self.norms = nn.ModuleList(LayerNorm2d(c) for c in coarse_in)
        self.fine_proj = nn.Conv2d(4 * c4, dim, 1)
        self.projs = nn.ModuleList(nn.Conv2d(c, dim, 1, bias=False) for c in coarse_in)
        self.mlp = nn.Sequential(
            LayerNorm2d(dim), nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        self.down = nn.Conv2d(dim, out_dim, 1)
        self.out_norm = LayerNorm2d(out_dim)

    def forward(self, features: list) -> torch.Tensor:
        fine, *coarse = features
        h, w = coarse[0].shape[-2:]
        x = self.fine_proj(F.pixel_unshuffle(self.stem(fine), 2))
        for norm, proj, c in zip(self.norms, self.projs, coarse):
            c = proj(norm(c))  # 1x1 commutes with bilinear, so project first
            if c.shape[-2:] != (h, w):
                c = F.interpolate(c, (h, w), mode="bilinear", align_corners=False)
            x = x + c
        return self.out_norm(self.down(x + self.mlp(x)))


class DINOv3Descriptor(nn.Module):
    """Describes images densely (features) and keypoints sparsely (descriptors)."""

    default_conf = {
        "resize": (512, 512),  # int: long side, aspect kept, (h, w): squashed
        "describe": True,  # layer-0 descriptors at the keypoints, if there are any
        "weights": WEIGHTS_URL,
    }
    edge_divisible_by = 32  # backbone stride

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        pp.check_resize(conf.resize, self.edge_divisible_by)
        self.backbone = ConvNeXt()
        self.head = LinearConvHead()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[:, None, None])

        self.register_buffer("_rgb8_codebook", None, persistent=False)
        state_dict = ops.load_state_dict(conf.weights, "extractor.")
        sub = lambda p: {  # noqa: E731
            k.removeprefix(p): v for k, v in state_dict.items() if k.startswith(p)
        }
        self.backbone.load_state_dict(sub("backbone."))
        self.head.load_state_dict(sub("head."))
        # layer-0 matcher head, input_proj and to_search fused into one linear map
        descriptor_head = sub("descriptor_head.")
        self.descriptor_head = nn.Linear(*descriptor_head["weight"].shape[::-1])
        self.descriptor_head.load_state_dict(descriptor_head)

    def _apply(self, fn, *args, **kwargs):
        # A half->float round trip can corrupt a codebook even when its final
        # dtype is FP32. Any module transform invalidates this derived cache.
        self._rgb8_codebook = None
        return super()._apply(fn, *args, **kwargs)

    def forward(self, data: dict, resize: Optional[pp.Resize] = None) -> dict:
        """Features of images, with optional byte preparation before tensor work."""
        if "rgb8_backend" in data and isinstance(data["image"], np.ndarray):
            return self._forward_rgb8(
                data["image"],
                data.get("keypoints"),
                resize,
                backend=data["rgb8_backend"],
                host_sampling=data.get("rgb8_host_sampling", False),
            )
        return self._forward(data, resize)

    def _forward(self, data: dict, resize: Optional[pp.Resize] = None) -> dict:
        """Shared tensor forward, including the compiled descriptor path."""
        x = self.preprocess(data["image"], resize)
        pred = {
            "features": self.encode(x),
            "image_size": pp.image_sizes(data["image"]).to(x),
        }
        if self.conf.describe and "keypoints" in data:
            pred["descriptors"] = self.sample_descriptors(
                pred["features"], data["keypoints"], pred["image_size"]
            )
        return pred

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Dense features (B, C, h / 8, w / 8) of images already at the input size."""
        return self.head(self.backbone((x - self.mean) / self.std))

    def preprocess(
        self, img: pp.Images, resize: Optional[pp.Resize] = None
    ) -> torch.Tensor:
        """Resize and stack images to the input resolution, conf.resize if None."""
        resize = self.conf.resize if resize is None else resize
        return pp.resize_images(img, resize, self.edge_divisible_by)

    _stock_preprocess = preprocess

    @property
    def pixel_dtype(self) -> torch.dtype:
        """Input precision before resize, matching the normalization buffers."""
        return self.mean.dtype

    def rgb8_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Normalize byte pixels on the descriptor's device and input precision."""
        return pp.numpy_image_to_torch(image).to(
            device=self.mean.device, dtype=self.pixel_dtype
        )

    def describe_rgb8(
        self,
        image: np.ndarray,
        keypoints: Optional[torch.Tensor] = None,
        resize: Optional[pp.Resize] = None,
        *,
        backend: str = "torch",
        host_sampling: bool = False,
    ) -> dict:
        """Describe HWC byte input through the same public forward dispatch."""
        data = {
            "image": image,
            "rgb8_backend": backend,
            "rgb8_host_sampling": host_sampling,
        }
        if keypoints is not None:
            data["keypoints"] = keypoints
        return self.forward(data, resize)

    def _forward_rgb8(
        self,
        image: np.ndarray,
        keypoints: Optional[torch.Tensor] = None,
        resize: Optional[pp.Resize] = None,
        *,
        backend: str = "torch",
        host_sampling: bool = False,
    ) -> dict:
        """Describe HWC RGB8 with an optional CUDA preparation backend.

        The caller must not mutate image during the blocking upload. Unsupported
        inputs and inference settings use the original Torch preprocessing.
        Byte input uses the normalization buffers' dtype before resizing.
        """
        if not isinstance(host_sampling, bool):
            raise ValueError("host_sampling must be a bool")
        if backend not in ("torch", "torch-compile", "triton"):
            raise ValueError("backend must be 'torch', 'torch-compile' or 'triton'.")
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("describe_rgb8 expects an HWC RGB8 NumPy image")
        resize = self.conf.resize if resize is None else resize
        device = self.mean.device
        pixel_dtype = self.pixel_dtype
        supported = (
            pixel_dtype == torch.float32
            and getattr(self.preprocess, "__func__", None)
            is DINOv3Descriptor._stock_preprocess
            and type(image) is np.ndarray
            and resize == (512, 512)
            and 1 <= image.shape[0] <= 65535
            and 1 <= image.shape[1] <= 65535
            and image.shape[0] * image.shape[1] <= 64_000_000
            and device.type == "cuda"
            and (backend != "triton" or sys.platform.startswith("linux"))
            and not self.training
            and not torch.is_grad_enabled()
            and not torch.are_deterministic_algorithms_enabled()
            and device.index == torch.cuda.current_device()
        )
        port = None
        if supported:
            if backend == "triton":
                try:
                    from .utils import rgb8_triton as port
                except ModuleNotFoundError as error:
                    if error.name != "triton":
                        raise
            else:
                from .utils import rgb8_torch as port
        if port is not None:
            table = self._rgb8_codebook
            if (
                table is None
                or table.device != device
                or table.dtype != torch.float32
                or tuple(table.shape) != (256,)
                or not table.is_contiguous()
            ):
                table = torch.from_numpy(port.fp32_codebook()).to(device)
                self._rgb8_codebook = table
            options = {"compile_conversion": True} if backend == "torch-compile" else {}
            prepared = port.prepare_rgb8(
                image,
                target_hw=(512, 512),
                device=device,
                codebook_device=table,
                host_sampling=host_sampling,
                **options,
            )
            return port.describe_prepared(self, prepared, keypoints=keypoints)
        # CPU, gradients, other layouts/platforms/resizes or absent Triton retain
        # the original normalization, interpolation and descriptor forward path.
        tensor = self.rgb8_to_tensor(image)
        data = {"image": tensor}
        if keypoints is not None:
            data["keypoints"] = keypoints
        pred = self._forward(data, resize)
        if keypoints is not None:
            pred["keypoints"] = keypoints
        return pred

    def sample_descriptors(
        self, features: torch.Tensor, keypoints: torch.Tensor, image_size: torch.Tensor
    ) -> torch.Tensor:
        """L2-normalized layer-0 descriptors (B, N, D) at keypoints in pixels."""
        kpts_n = ops.normalize_keypoints(keypoints, image_size)
        descriptors = ops.sample_features(features, kpts_n)
        descriptors = self.descriptor_head(descriptors)
        return F.normalize(descriptors, dim=-1)

    @torch.no_grad()
    def describe(
        self,
        img: pp.Images,
        keypoints: Optional[torch.Tensor] = None,
        resize: Optional[pp.Resize] = None,
    ) -> dict:
        """Dense features, the image and its size (w, h), descriptors if keypoints."""
        if isinstance(img, torch.Tensor) and img.dim() == 3:
            img = img[None]  # add batch dim
        data = {"image": img}
        if keypoints is not None:
            data["keypoints"] = keypoints
        pred = self.forward(data, resize)
        pred["image"] = img  # the refiner reads its own resolution from it
        if keypoints is not None:
            pred["keypoints"] = keypoints
        return pred
