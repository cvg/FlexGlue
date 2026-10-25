from types import SimpleNamespace
from typing import Optional

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

    def forward(self, data: dict, resize: Optional[pp.Resize] = None) -> dict:
        """Features of data["image"] (Images), descriptors at data["keypoints"]."""
        x = self.preprocess(data["image"], resize)
        pred = {
            "features": self.head(self.backbone((x - self.mean) / self.std)),
            "image_size": pp.image_sizes(data["image"]).to(x),
        }
        if self.conf.describe and "keypoints" in data:
            pred["descriptors"] = self.sample_descriptors(
                pred["features"], data["keypoints"], pred["image_size"]
            )
        return pred

    def preprocess(
        self, img: pp.Images, resize: Optional[pp.Resize] = None
    ) -> torch.Tensor:
        """Resize and stack images to the input resolution, conf.resize if None."""
        resize = self.conf.resize if resize is None else resize
        return pp.resize_images(img, resize, self.edge_divisible_by)

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
