"""FlexGlue: Sparse-to-Dense Matching and Visual Disambiguation."""

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .descriptor import DINOv3Descriptor
from .refiner import RoMav2Refiner
from .utils import ops
from .utils import preprocess as pp
from .utils.modules import (
    LearnableFourierPositionalEncoding,
    MatchHead,
    PairContext,
    TransformerLayer,
)

WEIGHTS_URL = "https://github.com/cvg/FlexGlue/releases/download/v0.1/flexglue.pth"


class FlexGlue(nn.Module):
    default_conf = {
        "input_dim": 256,
        "descriptor_dim": 512,
        "search_dim": 64,
        "n_layers": 9,
        "num_heads": 8,
        "filter_threshold": 0.1,  # min match confidence
        "max_distance_px": 3.0,  # max cycle error, pixels of the resized image
        "min_covisibility": 0.0,  # invalidate all matches if covisibility_score is below
        "early_stop_decay": None,  # in [0, 1): decay per earlier layer (0: no decay)
        "remove_border_px": 0.0,  # border margin, pixels of the resized image
        "mp": "float16",  # transformer dtype: float16, bfloat16 or None (fp32), readout in fp32
        "describe": True,  # build a DINOv3Descriptor for inputs without features
        "describe_resize": (512, 512),  # descriptor input, as refine_resize
        "refine": False,  # refine with RoMa v2's dense refiners
        "refine_resize": (512, 512),  # refiner input, int: long side, (h, w): squash
        "refine_scales": (2, 1),  # RoMa refiner strides, coarse to fine
        "coarse_stride": 8,  # coarse grid pitch, pixels of the resized image (8: cells)
        "max_num_samples": 5000,  # matches sampled from the grids without keypoints
        "kde_balance": True,  # RoMa v2's kde balancing of the sampled matches
        "gemm_nn": False,  # sparse: mutual NN on summed squared distances, much faster
        "query_chunk_size": 4096,  # points per correspondence-head call, bounds memory
        "weights": WEIGHTS_URL,
    }
    fixed_conf = (
        "weights",
        "refine_scales",
        "query_chunk_size",
        "input_dim",
        "descriptor_dim",
        "search_dim",
        "num_heads",
    )
    stride = 8  # descriptor input pixels per feature cell

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        d, h = conf.descriptor_dim, conf.num_heads
        self.input_proj = nn.Linear(conf.input_dim, d)
        self.posenc = LearnableFourierPositionalEncoding(2, d // h)
        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h) for _ in range(conf.n_layers)]
        )
        # heads[k] reads the features after k transformer layers, heads[0] the raw ones
        self.heads = nn.ModuleList(
            [
                MatchHead(d, conf.search_dim, conf.query_chunk_size, cross=k > 0)
                for k in range(conf.n_layers + 1)
            ]
        )
        self.loaded_heads = set(range(conf.n_layers + 1))
        if conf.weights is not None:
            self.load_weights(conf.weights)
        # the descriptor weights live in the same file, under "extractor."
        self.descriptor = (
            DINOv3Descriptor(weights=conf.weights) if conf.describe else None
        )
        self.refiner = RoMav2Refiner(conf.refine_scales) if conf.refine else None
        self.static_lengths: Optional[Tuple[int, ...]] = None  # set by compile

    def compile(
        self,
        mode: str = "reduce-overhead",
        static_lengths: Tuple[int, ...] = (512, 1024, 2048, 4096),
    ) -> "FlexGlue":
        """Compile the transformer layers and heads, as LightGlue's compile."""
        for block in self.transformers:  # dense tokens: fixed size, no masks
            block.forward = torch.compile(block.forward, mode=mode, fullgraph=True)
        for head in self.heads:  # queries padded to static_lengths, eager beyond
            head.compiled_forward = torch.compile(
                head.forward, mode=mode, fullgraph=True
            )
        self.static_lengths = tuple(sorted(static_lengths))
        return self

    def forward(self, data: dict, **conf) -> dict:
        """LightGlue drop-in: data = {"image0": data0, "image1": data1}."""
        return self.match(data["image0"], data["image1"], **conf)

    def match(self, data0: dict, data1: dict, sparse: bool = False, **conf) -> dict:
        """Match two images, the mode depends on which images have keypoints."""
        conf = self.resolve_conf(conf)
        data0, data1 = self.describe_inputs(data0, data1, conf)
        dirs = self.directions(data0, data1)
        context = self.decode_pair(data0, data1, conf)
        if not dirs:  # neither has keypoints: sample correspondences
            return self.match_sampled(context, data0, data1, conf)
        sparse = sparse and len(dirs) == 2
        # sparse: max_distance_px bounds the mutual NN distance, no cycle readouts
        pred = self.keypoint_correspondences(
            context, data0, data1, dirs, conf, cycle=not sparse
        )
        if sparse:  # both: mutual nearest neighbors
            pred.update(self.sparse_matches(pred, data0, data1, conf))
            pred["keypoints0"], pred["keypoints1"] = (
                data0["keypoints"],
                data1["keypoints"],
            )
        else:  # one: matches from i to j, both: both directions concatenated
            pred.update(self.dense_matches(pred, data0, data1, dirs, conf))
        return pred

    def keypoint_correspondences(
        self,
        context: PairContext,
        data0: dict,
        data1: dict,
        dirs: List[tuple],
        conf,
        cycle: bool = True,
    ) -> dict:
        """Predicted point, confidence and validity of the keypoints, per direction."""
        data = (data0, data1)
        pred = {
            "covisibility_score": context.covisibility_score,
            "stop_layer": context.layer,
        }
        dense = self.prepare_dense(context, data0, data1, dirs, conf)
        for i, j in dirs:
            size_i, size_j = data[i]["image_size"], data[j]["image_size"]
            kpts_ni = ops.normalize_keypoints(data[i]["keypoints"], size_i)
            if conf.refine:
                p2di_nj, confidence = self.sample_correspondences(dense[i], kpts_ni)
            else:
                p2di_nj, confidence = self.run_head(
                    context.layer, kpts_ni, *context.direction(i)
                )
            pred[f"p2d{i}_n{j}"] = p2di_nj
            pred[f"p2d{i}_i{j}"] = ops.denormalize_keypoints(p2di_nj, size_j)
            if conf.max_distance_px and cycle:
                error, back_confidence = self.cycle_error(
                    context, i, kpts_ni, p2di_nj, size_i, dense.get(j)
                )
                pred[f"cycle_error{i}"] = error
                # geometric mean of the forward and backward confidence
                confidence = (confidence * back_confidence).sqrt()
            pred[f"confidence{i}"] = confidence
        return self.gate_points(pred, context, data0, data1, conf)

    def match_pair(
        self,
        image0: pp.Images,
        image1: pp.Images,
        detector: Optional[nn.Module] = None,
        sparse: bool = False,
        **conf,
    ) -> dict:
        """Match images (Images), keypoints from detector if given."""
        data = [
            {
                "image": (
                    im[None] if isinstance(im, torch.Tensor) and im.dim() == 3 else im
                )
            }
            for im in (image0, image1)
        ]
        if detector is not None:
            for data_i in data:
                feats = [detector.extract(im) for im in data_i["image"]]
                if len({f["keypoints"].shape[1] for f in feats}) > 1:
                    raise ValueError(
                        "Batching needs the same number of keypoints in each image."
                    )
                data_i.update({k: torch.cat([f[k] for f in feats]) for k in feats[0]})
        return self.match(*data, sparse=sparse, **conf)

    def match_dense(
        self, data0: dict, data1: dict, symmetric: bool = True, **conf
    ) -> dict:
        """Dense grid 0 -> 1, and 1 -> 0 if symmetric, without matching."""
        conf = self.resolve_conf(conf)
        if conf.max_distance_px and not symmetric:
            raise ValueError("max_distance_px needs symmetric=True.")
        data0, data1 = self.describe_inputs(data0, data1, conf)
        context = self.decode_pair(data0, data1, conf)
        dirs = ((0, 1), (1, 0)) if symmetric else ((0, 1),)
        pred = {
            "covisibility_score": context.covisibility_score,
            "stop_layer": context.layer,
        }
        pred.update(self.dense_correspondences(context, data0, data1, dirs, conf))
        return self.gate_dense(pred, context, data0, data1, conf)

    def describe_inputs(self, data0: dict, data1: dict, conf) -> Tuple[dict, dict]:
        """Copies of the inputs, with the features of those that only have an image."""
        out = []
        for data_i in (data0, data1):
            if "features" not in data_i:
                if not conf.describe:
                    raise KeyError(
                        "No features: pass them or build FlexGlue(describe=True)."
                    )
                described = self.descriptor(
                    {"image": data_i["image"]}, resize=conf.describe_resize
                )
                data_i = {**described, **data_i}
            out.append(data_i)
        return tuple(out)

    def prepare_dense(
        self, context: PairContext, data0: dict, data1: dict, dirs: List[tuple], conf
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Refined grids by source image, both directions if the cycle check needs them."""
        if not conf.refine:
            return {}
        if conf.max_distance_px:
            dirs = ((0, 1), (1, 0))
        data = (data0, data1)
        return {
            i: self.grid_correspondences(context, i, data[i], data[j], conf)
            for i, j in dirs
        }

    def match_sampled(
        self, context: PairContext, data0: dict, data1: dict, conf
    ) -> dict:
        """No keypoints: up to max_num_samples matches drawn from both dense grids."""
        pred = {
            "covisibility_score": context.covisibility_score,
            "stop_layer": context.layer,
        }
        dirs = ((0, 1), (1, 0))
        pred.update(self.dense_correspondences(context, data0, data1, dirs, conf))
        pred = self.gate_dense(pred, context, data0, data1, conf)
        pred.update(self.sample_matches(pred, data0, data1, conf))
        return pred

    def directions(self, data0: dict, data1: dict) -> List[tuple]:
        """Directions i -> j to run: those whose image i has keypoints."""
        return [(i, 1 - i) for i, d in enumerate((data0, data1)) if "keypoints" in d]

    def covisibility_score(
        self, x0: torch.Tensor, x1: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """covisibility_score (B,) of the tokens after layer, 1 for the raw features."""
        covisibility = self.heads[layer].covisibility
        if covisibility is None:  # raw features: no covisibility evidence
            return x0.new_ones(x0.shape[0])
        tokens = torch.cat([x0, x1], 1).float()
        return torch.sigmoid(covisibility(tokens)).mean((1, 2))

    def decode_pair(
        self, data0: dict, data1: dict, conf: SimpleNamespace
    ) -> PairContext:
        """Cross features and covisibility_score of the pair, refiner features if refine."""
        if self.static_lengths is not None:  # new CUDA graph step per pair
            torch.compiler.cudagraph_mark_step_begin()
        f0, f1 = data0["features"], data1["features"]
        hw0, hw1 = f0.shape[-2:], f1.shape[-2:]
        enc0, enc1 = (
            self.posenc(ops.cell_centers(*hw, scale=(max(hw),) * 2, device=f0.device))
            for hw in (hw0, hw1)
        )
        x0, x1 = self.input_proj(f0.flatten(2).mT), self.input_proj(f1.flatten(2).mT)
        layer, covisibility_score = conf.n_layers, None
        dtype = torch.float16 if conf.mp is None else getattr(torch, conf.mp)
        with torch.autocast(f0.device.type, dtype, enabled=conf.mp is not None):
            for k, block in enumerate(self.transformers[: conf.n_layers], 1):
                x0, x1 = block(x0, x1, enc0, enc1)
                if conf.early_stop_decay is None or k == conf.n_layers:
                    continue
                # stop once the whole batch is a confident non-pair (host sync)
                score = self.covisibility_score(x0, x1, k)
                gamma = (1 - conf.early_stop_decay) ** (len(self.transformers) - k)
                if (score < conf.min_covisibility * gamma).all():
                    layer, covisibility_score = k, score
                    break
        x0, x1 = x0.float(), x1.float()
        if self.static_lengths is not None:  # own the tokens, not the graph's buffers
            x0, x1 = x0.clone(), x1.clone()
        if covisibility_score is None:
            covisibility_score = self.covisibility_score(x0, x1, layer)
        cf = (x0.mT.unflatten(-1, hw0), x1.mT.unflatten(-1, hw1))
        context = PairContext(layer, covisibility_score, cf)
        if conf.refine:
            fine = tuple(
                self.refiner.extract(d["image"], resize=conf.refine_resize)
                for d in (data0, data1)
            )
            context = context._replace(fine=fine)
        return context

    def run_head(
        self,
        layer: int,
        points_ni: torch.Tensor,
        cf_i: torch.Tensor,
        cf_j: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """heads[layer] on points (B, N, 2), padded to a static length if compiled."""
        n = points_ni.shape[1]
        if self.static_lengths is None or n > self.static_lengths[-1]:
            return self.heads[layer](points_ni, cf_i, cf_j)
        length = next(s for s in self.static_lengths if s >= n)
        points_ni = F.pad(points_ni, (0, 0, 0, length - n))  # rows are independent
        p2di_nj, confidence = self.heads[layer].compiled_forward(points_ni, cf_i, cf_j)
        # clone: CUDA graph outputs are overwritten by the next replay
        return p2di_nj[:, :n].clone(), confidence[:, :n].clone()

    def grid_correspondences(
        self, context: PairContext, i: int, data_i: dict, data_j: dict, conf
    ) -> Dict[str, torch.Tensor]:
        """Grid i -> j at coarse_stride: the head on all points, refined if refine."""
        cf_i, cf_j = context.direction(i)
        b, (h_r, w_r) = cf_i.shape[0], [self.stride * s for s in cf_i.shape[-2:]]
        if h_r % conf.coarse_stride or w_r % conf.coarse_stride:
            raise ValueError(
                f"coarse_stride {conf.coarse_stride} must divide the resized size "
                f"{(h_r, w_r)}."
            )
        h, w = h_r // conf.coarse_stride, w_r // conf.coarse_stride
        centers = ops.cell_centers(h, w, scale=(w, h), device=cf_i.device)
        dense_p2di_nj, dense_confidence = self.run_head(
            context.layer, centers.expand(b, -1, -1), cf_i, cf_j
        )
        dense_p2di_nj = dense_p2di_nj.unflatten(1, (h, w))
        dense_confidence = dense_confidence.unflatten(1, (h, w))
        size_j = data_j["image_size"]
        if not conf.refine:
            return {
                "dense_p2di_nj": dense_p2di_nj,
                "dense_p2di_ij": ops.denormalize_keypoints(dense_p2di_nj, size_j),
                "dense_confidencei": dense_confidence,
            }
        fine_i, fine_j = context.fine_direction(i)
        return self.refiner.refine(
            dense_p2di_nj,
            torch.logit(dense_confidence, eps=1e-6),
            fine_i,
            fine_j,
            size_j,
        )

    def dense_correspondences(
        self, context: PairContext, data0: dict, data1: dict, dirs: tuple, conf
    ) -> Dict[str, torch.Tensor]:
        """grid_correspondences for each direction, under the output keys."""
        data, pred = (data0, data1), {}
        for i, j in dirs:
            dense = self.grid_correspondences(context, i, data[i], data[j], conf)
            pred[f"dense_p2d{i}_n{j}"] = dense["dense_p2di_nj"]
            pred[f"dense_p2d{i}_i{j}"] = dense["dense_p2di_ij"]
            pred[f"dense_confidence{i}"] = dense["dense_confidencei"]
            if "dense_covi_ij" in dense:
                pred[f"dense_cov{i}_i{j}"] = dense["dense_covi_ij"]
        return pred

    def sample_correspondences(
        self, dense: Dict[str, torch.Tensor], points_ni: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read a grid at normalized points (B, N, 2): predicted points and confidence."""
        p2di_nj = ops.sample_features(ops.to_chw(dense["dense_p2di_nj"]), points_ni)
        confidence = ops.sample_features(dense["dense_confidencei"][:, None], points_ni)
        return p2di_nj, confidence[..., 0]

    def cycle_error(
        self,
        context: PairContext,
        i: int,
        points_ni: torch.Tensor,
        p2di_nj: torch.Tensor,
        size_i: torch.Tensor,
        dense_j: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map p2di_nj back to i with the grid j -> i or the head: error in px, confidence."""
        if dense_j is None:
            cf_i, cf_j = context.direction(i)
            back_ni, back_confidence = self.run_head(context.layer, p2di_nj, cf_j, cf_i)
        else:
            back_ni, back_confidence = self.sample_correspondences(dense_j, p2di_nj)
        back_ii = ops.denormalize_keypoints(back_ni, size_i)
        error = (back_ii - ops.denormalize_keypoints(points_ni, size_i)).norm(dim=-1)
        return error, back_confidence

    def pixel_scale(self, data_i: dict) -> torch.Tensor:
        """Image pixels per resized pixel (B,), long side of image / resized image."""
        resized_side = self.stride * max(data_i["features"].shape[-2:])
        return data_i["image_size"].max(-1).values / resized_side

    def gate_points(
        self, pred: dict, context: PairContext, data0: dict, data1: dict, conf
    ) -> dict:
        """Pair, cycle and border checks on keypoints as valid{i}."""
        data = (data0, data1)
        scale = (self.pixel_scale(data0), self.pixel_scale(data1))
        is_covisible = context.covisibility_score >= conf.min_covisibility
        if conf.min_covisibility > 0:
            pred["is_covisible"] = is_covisible
        for i, j in ((0, 1), (1, 0)):
            if f"confidence{i}" not in pred:
                continue
            size_i, size_j = data[i]["image_size"], data[j]["image_size"]
            border_i = conf.remove_border_px * scale[i]
            border_j = conf.remove_border_px * scale[j]
            valid = is_covisible[:, None]
            valid = valid & ops.inside_border(data[i]["keypoints"], size_i, border_i)
            valid &= ops.inside_border(pred[f"p2d{i}_i{j}"], size_j, border_j)
            if f"cycle_error{i}" in pred:
                max_error = conf.max_distance_px * scale[i][:, None]
                valid &= pred[f"cycle_error{i}"] < max_error
            pred[f"valid{i}"] = valid
        return pred

    def gate_dense(
        self, pred: dict, context: PairContext, data0: dict, data1: dict, conf
    ) -> dict:
        """Pair, filter, cycle and border gates on grid confidence."""
        data = (data0, data1)
        scale = (self.pixel_scale(data0), self.pixel_scale(data1))
        is_covisible = context.covisibility_score >= conf.min_covisibility
        if conf.min_covisibility > 0:
            pred["is_covisible"] = is_covisible
        for i, j in ((0, 1), (1, 0)):
            if f"dense_confidence{i}" not in pred:
                continue
            size_i, size_j = data[i]["image_size"], data[j]["image_size"]
            border_i = conf.remove_border_px * scale[i]
            border_j = conf.remove_border_px * scale[j]
            dense_p2di_nj = pred[f"dense_p2d{i}_n{j}"]
            dense_confidence = pred[f"dense_confidence{i}"]
            dense_p2di_ni = ops.grid_sources(dense_p2di_nj)
            dense_p2di_ii = ops.denormalize_keypoints(dense_p2di_ni, size_i)
            valid = is_covisible[:, None, None]
            valid = valid & (dense_confidence >= conf.filter_threshold)
            valid &= ops.inside_border(dense_p2di_ii, size_i, border_i)
            valid &= ops.inside_border(pred[f"dense_p2d{i}_i{j}"], size_j, border_j)
            if conf.max_distance_px:
                dense_j = {
                    "dense_p2di_nj": pred[f"dense_p2d{j}_n{i}"],
                    "dense_confidencei": pred[f"dense_confidence{j}"],
                }
                err = self.cycle_error(
                    context,
                    i,
                    dense_p2di_ni.flatten(1, 2),
                    dense_p2di_nj.flatten(1, 2),
                    size_i,
                    dense_j,
                )[0]
                max_error = conf.max_distance_px * scale[i][:, None, None]
                valid &= err.view_as(dense_confidence) < max_error
            pred[f"dense_valid{i}"] = valid
            pred[f"dense_confidence{i}"] = torch.where(valid, dense_confidence, 0)
        return pred

    def sparse_matches(self, pred: dict, data0: dict, data1: dict, conf) -> dict:
        """Mutual nearest neighbours of keypoints and predicted points."""
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        p2d0_i1, p2d1_i0 = pred["p2d0_i1"], pred["p2d1_i0"]
        if conf.gemm_nn:  # centered on each image for float32 precision
            c0, c1 = (d["image_size"][:, None] / 2 for d in (data0, data1))
            x = torch.cat([p2d0_i1 - c1, kpts0 - c0], -1)
            y = torch.cat([kpts1 - c1, p2d1_i0 - c0], -1)
            m0, m1 = ops.mutual_nn(ops.sq_dist_gemm(x, y))

            # exact mean distance of the selected pairs only
            def take(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
                return t.gather(1, m.clamp(0)[..., None].expand(-1, -1, 2))

            dist0 = 0.5 * (
                (p2d0_i1 - take(kpts1, m0)).norm(dim=-1)
                + (kpts0 - take(p2d1_i0, m0)).norm(dim=-1)
            )
            dist1 = 0.5 * (
                (take(p2d0_i1, m1) - kpts1).norm(dim=-1)
                + (take(kpts0, m1) - p2d1_i0).norm(dim=-1)
            )
        else:
            dist = 0.5 * (torch.cdist(p2d0_i1, kpts1) + torch.cdist(kpts0, p2d1_i0))
            m0, m1 = ops.mutual_nn(dist)
            dist0, dist1 = dist.min(2).values, dist.min(1).values
        # geometric mean of both confidences, as the cycle score
        scores0 = (
            pred["confidence0"] * pred["confidence1"].gather(1, m0.clamp(0))
        ).sqrt()
        scores1 = (
            pred["confidence1"] * pred["confidence0"].gather(1, m1.clamp(0))
        ).sqrt()
        valid0 = (m0 > -1) & (scores0 >= conf.filter_threshold)
        valid1 = (m1 > -1) & (scores1 >= conf.filter_threshold)
        valid0 &= pred["valid0"] & pred["valid1"].gather(1, m0.clamp(0))
        valid1 &= pred["valid1"] & pred["valid0"].gather(1, m1.clamp(0))
        if conf.max_distance_px:  # dist averages both images, so does the scale
            scale = (self.pixel_scale(data0) + self.pixel_scale(data1)) / 2
            max_dist = conf.max_distance_px * scale[:, None]
            valid0 &= dist0 < max_dist
            valid1 &= dist1 < max_dist
        return {
            "matches0": torch.where(valid0, m0, -1),
            "matches1": torch.where(valid1, m1, -1),
            "matching_scores0": torch.where(valid0, scores0, 0),
            "matching_scores1": torch.where(valid1, scores1, 0),
        }

    def dense_matches(
        self, pred: dict, data0: dict, data1: dict, dirs: List[tuple], conf
    ) -> dict:
        """Each keypoint with its predicted point, concatenated over directions."""
        data, kpts0, kpts1, scores = (data0, data1), [], [], []
        for i, j in dirs:
            pair = [data[i]["keypoints"], pred[f"p2d{i}_i{j}"]]
            kpts0.append(pair[i])
            kpts1.append(pair[j])
            scores.append(torch.where(pred[f"valid{i}"], pred[f"confidence{i}"], 0))
        return ops.identity_matches(
            torch.cat(kpts0, 1),
            torch.cat(kpts1, 1),
            torch.cat(scores, 1),
            conf.filter_threshold,
        )

    def sample_matches(self, pred: dict, data0: dict, data1: dict, conf) -> dict:
        """max_num_samples cells of both gated grids, by confidence or kde balance."""
        data, kpts0, kpts1, scores = (data0, data1), [], [], []
        for i, j in ((0, 1), (1, 0)):
            dense_p2di_ij = pred[f"dense_p2d{i}_i{j}"]
            dense_p2di_ii = ops.denormalize_keypoints(
                ops.grid_sources(dense_p2di_ij), data[i]["image_size"]
            )
            pair = [dense_p2di_ii.flatten(1, 2), dense_p2di_ij.flatten(1, 2)]
            kpts0.append(pair[i])
            kpts1.append(pair[j])
            scores.append(pred[f"dense_confidence{i}"].flatten(1))  # 0 where gated out
        kpts0, kpts1, scores = (torch.cat(x, 1) for x in (kpts0, kpts1, scores))
        weights = scores
        if conf.kde_balance:
            weights = ops.kde_weights(
                pred["dense_p2d0_n1"],
                pred["dense_p2d1_n0"],
                scores,
                conf.max_num_samples,
            )
        idx, idx_xy = ops.sample_cells(weights, conf.max_num_samples)
        return ops.identity_matches(
            kpts0.gather(1, idx_xy),
            kpts1.gather(1, idx_xy),
            scores.gather(1, idx),
            conf.filter_threshold,
        )

    def load_weights(self, weights: str) -> None:
        """Strict for everything except whole per-layer heads, which may be absent."""
        loaded = ops.load_state_dict(weights, "matcher.")
        depth = len({k.split(".")[1] for k in loaded if k.startswith("transformers.")})
        if self.conf.n_layers > depth:
            raise ValueError(f"n_layers={self.conf.n_layers}, weights have {depth}.")
        state_dict = {}
        for k, v in loaded.items():
            if k.startswith("head."):  # single-head export: the last layer's head
                k = f"heads.{depth}." + k.removeprefix("head.")
            k = k.replace(".disambiguate.", ".covisibility.")  # exported head name
            parts = k.split(".")
            limit = self.conf.n_layers + (parts[0] == "heads")
            if parts[0] in ("transformers", "heads") and int(parts[1]) >= limit:
                continue  # truncated model: layers past n_layers are not built
            state_dict[k] = v
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise KeyError(f"Unexpected weights: {unexpected}")
        if any(not k.startswith("heads.") for k in missing):
            raise KeyError(f"Missing weights: {missing}")
        partial = {int(k.split(".")[1]) for k in missing}
        for layer in partial:
            prefix = f"heads.{layer}."
            if any(k.startswith(prefix) for k in state_dict):
                raise KeyError(f"Head {layer} is only partially in the weights.")
        self.loaded_heads = set(range(self.conf.n_layers + 1)) - partial

    def resolve_conf(self, conf: dict) -> SimpleNamespace:
        """Merge per-call overrides with the construction conf."""
        if unknown := conf.keys() - self.default_conf.keys():
            raise KeyError(f"Unknown conf: {sorted(unknown)}")
        if fixed := conf.keys() & set(self.fixed_conf):
            raise KeyError(f"Fixed at construction: {sorted(fixed)}")
        c = SimpleNamespace(**{**vars(self.conf), **conf})
        if not 0 <= c.n_layers <= self.conf.n_layers:
            raise ValueError(f"n_layers must be in [0, {self.conf.n_layers}].")
        if c.n_layers not in self.loaded_heads:
            raise ValueError(f"No weights for the head of layer {c.n_layers}.")
        if c.describe and self.descriptor is None:
            raise ValueError("describe=True needs FlexGlue(describe=True) at init.")
        if c.refine and self.refiner is None:
            raise ValueError("refine=True needs FlexGlue(refine=True) at init.")
        if c.mp not in (None, "float16", "bfloat16"):
            raise ValueError(f"mp must be 'float16', 'bfloat16' or None: {c.mp!r}")
        if not isinstance(c.coarse_stride, int) or c.coarse_stride <= 0:
            raise ValueError(f"coarse_stride must be a positive int: {c.coarse_stride}")
        if c.early_stop_decay is not None:
            if not 0 <= c.early_stop_decay < 1:
                raise ValueError(
                    f"early_stop_decay must be in [0, 1): {c.early_stop_decay}"
                )
            if c.min_covisibility <= 0:
                raise ValueError("early_stop_decay needs min_covisibility > 0.")
            if missing := set(range(1, c.n_layers + 1)) - self.loaded_heads:
                raise ValueError(
                    f"early_stop_decay needs all heads, missing {sorted(missing)}."
                )
        return c
