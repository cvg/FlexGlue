"""Export FlexGlue to ONNX for COLMAP (onnx/check.py checks the models).

Two graphs are exported:

- flexglue_descriptor.onnx
    image [1, 3, S, S] float32 in [0, 1] -> features [1, 256, S/8, S/8]
  The image is resized to S x S (squashed, bilinear, no antialiasing) by the
  caller, as DINOv3Descriptor does with a (h, w) resize.

- flexglue_matcher.onnx (sparse mode, batch size 1)
    features0, features1          [1, 256, S/8, S/8] float32
    kpts0, kpts1                  [1, N0, 2], [1, N1, 2] float32, pixels with
                                  the origin at the top-left image corner,
                                  N0, N1 >= 1
    image_size0, image_size1      [1, 2] float32, (w, h) of the original image
    filter_threshold              scalar float32
    max_distance_px               scalar float32, inf disables the check and
                                  0 rejects all matches (in PyTorch, 0 or None
                                  disables it)
    min_covisibility              scalar float32, 0 disables the check
  ->
    matches0                      [1, N0] int64, index into kpts1 or -1
    mscores0                      [1, N0] float32, 0 where unmatched
    covisibility_score            [1] float32
  All matches are rejected (matches0 = -1) if
  covisibility_score < min_covisibility.

The graphs are traced from the PyTorch implementation itself
(DINOv3Descriptor.encode, FlexGlue.encode_pair, pair_context,
keypoint_correspondences and sparse_matches, as FlexGlue.match(...,
sparse=True)), with:
- the thresholds as graph inputs, so that they can be tuned at runtime;
- all transformer layers (no early stopping) and no refinement;
- exact_nn: the squared distances of the mutual nearest neighbor search are
  computed from coordinate differences, since ONNX Runtime runs float32 matrix
  multiplications in TF32 on CUDA;
- no chunking of the keypoints in the correspondence heads;
- q, k and v as contiguous blocks (SelfBlock.use_blocked_qkv), which ONNX
  Runtime splits much faster than the interleaved checkpoint layout.

By default, the attention (modules.attention) is exported as the fused
com.microsoft::MultiHeadAttention contrib operator of ONNX Runtime (flash /
memory-efficient attention on CUDA) instead of being decomposed into
MatMul/Softmax/MatMul, which materializes the attention matrices. Such models
require ONNX Runtime; use --no_fused_attention for standard ONNX operators.
Likewise, the GELU activations of the matcher are exported as the
com.microsoft::Gelu operator, a single kernel instead of five element-wise
operators (Div, Erf, Add, Mul, Mul) that ONNX Runtime does not fuse.

With --early_stop_layers (e.g. 2,4,6,8), the matcher stops early on pairs
that are confidently not covisible, as FlexGlue.match(..., early_stop_decay=...):
after each listed layer k, the pair is rejected without running the remaining
layers and the readout if its covisibility score is below
min_covisibility * (1 - early_stop_decay) ** (n_layers - k). The checks are ONNX
If nodes built from per-layer covisibility-score outputs of the traced graph,
and early_stop_decay is an additional scalar input (1 never stops early).

Requires onnx and onnxscript.

Usage (from the root of the repository):
    python onnx/export.py                         # descriptor + fp32 matcher
    python onnx/export.py --fp16 --matcher_only   # fp16 matcher
The models are written to onnx/models/ by default. onnx/check.py checks them
against PyTorch, and onnx/match_pair.py is an example of inference with ONNX
Runtime.
"""

import argparse
import contextlib
import copy
import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import onnx
import onnxscript
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper
from onnx.defs import OpSchema
from torch import nn

# Also run from a clone of the repository without installing it.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flexglue import DINOv3Descriptor, FlexGlue  # noqa: E402
from flexglue.flexglue import WEIGHTS_URL  # noqa: E402
from flexglue.utils import modules  # noqa: E402

# Minimal schema of ONNX Runtime's contrib MultiHeadAttention operator (only
# the query/key/value inputs and the output), so that the exporter can emit it.
onnx.defs.register_schema(
    OpSchema(
        "MultiHeadAttention",
        "com.microsoft",
        1,
        inputs=[
            OpSchema.FormalParameter("query", "T"),
            OpSchema.FormalParameter("key", "T"),
            OpSchema.FormalParameter("value", "T"),
        ],
        outputs=[OpSchema.FormalParameter("output", "T")],
        type_constraints=[("T", ["tensor(float)", "tensor(float16)"], "")],
        attributes=[OpSchema.Attribute("num_heads", OpSchema.AttrType.INT, "")],
    )
)
MSFT_OPSET = onnxscript.values.Opset("com.microsoft", 1)


@torch.library.custom_op("flexglue::multi_head_attention", mutates_args=())
def multi_head_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Attention on (B, N, H*D) tensors, as com.microsoft::MultiHeadAttention."""
    b, n, e = query.shape
    d = e // num_heads
    q, k, v = (
        t.view(b, t.shape[1], num_heads, d).transpose(1, 2) for t in (query, key, value)
    )
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, n, e)


@multi_head_attention.register_fake
def _(query, key, value, num_heads):
    return torch.empty_like(query)


def multi_head_attention_onnx(query, key, value, num_heads: int):
    return MSFT_OPSET.MultiHeadAttention(query, key, value, num_heads=num_heads)


# Schema of ONNX Runtime's contrib Gelu operator (exact, erf-based).
onnx.defs.register_schema(
    OpSchema(
        "Gelu",
        "com.microsoft",
        1,
        inputs=[OpSchema.FormalParameter("X", "T")],
        outputs=[OpSchema.FormalParameter("Y", "T")],
        type_constraints=[("T", ["tensor(float)", "tensor(float16)"], "")],
    )
)


@torch.library.custom_op("flexglue::gelu", mutates_args=())
def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU, as com.microsoft::Gelu and nn.GELU()."""
    return F.gelu(x)


@gelu.register_fake
def _(x):
    return torch.empty_like(x)


def gelu_onnx(x):
    return MSFT_OPSET.Gelu(x)


class FusedGELU(nn.Module):
    """nn.GELU with the fused operator."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.flexglue.gelu(x)


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """modules.attention on (B, N, H, D) tensors with the fused operator."""
    return torch.ops.flexglue.multi_head_attention(
        q.flatten(-2), k.flatten(-2), v.flatten(-2), q.shape[-2]
    )


@contextlib.contextmanager
def exportable(matcher: FlexGlue, fused: bool = True):
    """Configure the matcher (in place) and the attention for the export."""
    for block in matcher.transformers:
        block.self_attn.use_blocked_qkv()
    for head in matcher.heads:
        head.query_chunk_size = None  # cannot split a dynamic number of keypoints
    if fused:
        for layer in matcher.transformers:
            for ffn in (layer.self_attn.ffn, layer.cross_attn.ffn):
                gelus = [i for i, m in enumerate(ffn) if isinstance(m, nn.GELU)]
                for i in gelus:
                    ffn[i] = FusedGELU()
    original = modules.attention
    if fused:
        modules.attention = fused_attention
    try:
        yield
    finally:
        modules.attention = original


class DescriptorONNX(nn.Module):
    """Dense features of an image already resized to the input resolution."""

    def __init__(self, descriptor: DINOv3Descriptor):
        super().__init__()
        self.descriptor = descriptor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.descriptor.encode(image)


# Per-keypoint predictions of the debug graph used by onnx/check.py.
DEBUG_OUTPUTS = ["p2d0_i1", "p2d1_i0", "confidence0", "confidence1"]


class SparseMatcherONNX(nn.Module):
    """FlexGlue.match(..., sparse=True) with the thresholds as inputs (B=1)."""

    def __init__(
        self, matcher: FlexGlue, fp16: bool = False, debug_outputs: bool = False
    ):
        super().__init__()
        self.matcher = matcher
        self.n_layers = matcher.conf.n_layers
        self.conf = matcher.resolve_conf({"mp": None, "exact_nn": True})
        self.debug_outputs = debug_outputs
        # Cast the transformer layers to float16, including their residual
        # stream and LayerNorms, unlike the autocast of FlexGlue(mp="float16")
        # which keeps these in float32. The readout always runs in float32.
        self.fp16 = fp16
        if fp16:
            for block in matcher.transformers:
                block.half()

    def tokens(self, features0: torch.Tensor, features1: torch.Tensor):
        """Input tokens and positional encodings, as FlexGlue.encode_pair."""
        x0, x1, enc0, enc1 = self.matcher.encode_pair(features0, features1)
        if self.fp16:
            x0, x1, enc0, enc1 = x0.half(), x1.half(), enc0.half(), enc1.half()
        return x0, x1, enc0, enc1

    def forward(
        self,
        features0: torch.Tensor,
        features1: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
        filter_threshold: torch.Tensor,
        max_distance_px: torch.Tensor,
        min_covisibility: torch.Tensor,
    ):
        x0, x1, enc0, enc1 = self.tokens(features0, features1)
        for block in self.matcher.transformers[: self.n_layers]:
            x0, x1 = block(x0, x1, enc0, enc1)
        return self.match_tokens(
            x0,
            x1,
            features0,
            features1,
            kpts0,
            kpts1,
            image_size0,
            image_size1,
            filter_threshold,
            max_distance_px,
            min_covisibility,
        )

    def match_tokens(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        features0: torch.Tensor,
        features1: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
        filter_threshold: torch.Tensor,
        max_distance_px: torch.Tensor,
        min_covisibility: torch.Tensor,
    ):
        """Outputs from the tokens after the last layer, as FlexGlue.match."""
        m = self.matcher
        hw = (features0.shape[-2:], features1.shape[-2:])
        context = m.pair_context(x0, x1, hw, self.n_layers)
        conf = SimpleNamespace(
            **{
                **vars(self.conf),
                "filter_threshold": filter_threshold,
                "max_distance_px": max_distance_px,
                "min_covisibility": min_covisibility,
            }
        )
        data0 = {"features": features0, "keypoints": kpts0, "image_size": image_size0}
        data1 = {"features": features1, "keypoints": kpts1, "image_size": image_size1}
        dirs = [(0, 1), (1, 0)]
        pred = m.keypoint_correspondences(context, data0, data1, dirs, conf, False)
        pred.update(m.sparse_matches(pred, data0, data1, conf))
        outputs = (
            pred["matches0"],
            pred["matching_scores0"],
            pred["covisibility_score"],
        )
        if self.debug_outputs:
            outputs += tuple(pred[k] for k in DEBUG_OUTPUTS)
        return outputs


class LayerScoresMatcherONNX(SparseMatcherONNX):
    """SparseMatcherONNX that also outputs the covisibility scores after layers.

    Additional outputs covisibility_score_layer{k} for k in check_layers, used
    to build the early-stopping graph (see add_early_stopping).
    """

    def __init__(self, matcher: FlexGlue, check_layers, fp16: bool = False):
        super().__init__(matcher, fp16)
        self.check_layers = tuple(check_layers)

    def forward(
        self,
        features0: torch.Tensor,
        features1: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
        filter_threshold: torch.Tensor,
        max_distance_px: torch.Tensor,
        min_covisibility: torch.Tensor,
    ):
        x0, x1, enc0, enc1 = self.tokens(features0, features1)
        scores = []
        for k, block in enumerate(self.matcher.transformers[: self.n_layers], 1):
            x0, x1 = block(x0, x1, enc0, enc1)
            if k in self.check_layers:
                scores.append(self.matcher.covisibility_score(x0, x1, k))
        outputs = self.match_tokens(
            x0,
            x1,
            features0,
            features1,
            kpts0,
            kpts1,
            image_size0,
            image_size1,
            filter_threshold,
            max_distance_px,
            min_covisibility,
        )
        return (*outputs, *scores)


def stage_node_indices(nodes, stage_outputs):
    """Split the nodes into stages that compute each list of outputs in turn.

    Stage i contains the indices of the nodes needed for stage_outputs[i] that
    are not in an earlier stage, in topological order.
    """
    producer = {out: i for i, n in enumerate(nodes) for out in n.output}
    stages, done = [], set()
    for names in stage_outputs:
        stage, stack = set(), [producer[n] for n in names if n in producer]
        while stack:
            i = stack.pop()
            if i in stage or i in done:  # the ancestors of done are done
                continue
            stage.add(i)
            stack.extend(producer[x] for x in nodes[i].input if x in producer)
        done |= stage
        stages.append(sorted(stage))
    return stages


def stage_initializers(graph, nodes, stages):
    """Initializers of each stage: those first used by a node of the stage."""
    first_stage = {}
    for s, stage in enumerate(stages):
        for i in stage:
            for x in nodes[i].input:
                first_stage.setdefault(x, s)
    inits = [[] for _ in stages]
    for init in graph.initializer:
        if init.name in first_stage:
            inits[first_stage[init.name]].append(init)
    return inits


def matcher_output_info(names):
    """Value infos of outputs like MATCHER_OUTPUTS, with the given names."""
    types = [TensorProto.INT64, TensorProto.FLOAT, TensorProto.FLOAT]
    shapes = [[1, "num_keypoints0"], [1, "num_keypoints0"], [1]]
    return [
        helper.make_tensor_value_info(n, t, s) for n, t, s in zip(names, types, shapes)
    ]


def rename_values(nodes, rename):
    """Rename the inputs and outputs of the nodes (in place)."""
    for node in nodes:
        for values in (node.input, node.output):
            for j, x in enumerate(values):
                values[j] = rename.get(x, x)
    return nodes


def reject_branch(prefix: str, score: str):
    """Branch of a stop: no matches, and the score after the last layer run."""
    shape = f"{prefix}_shape"
    matches, mscores, covis = (f"{prefix}_{n}" for n in MATCHER_OUTPUTS)
    nodes = [
        helper.make_node("Shape", ["kpts0"], [shape], start=0, end=2),
        helper.make_node(
            "ConstantOfShape",
            [shape],
            [matches],
            value=helper.make_tensor("v", TensorProto.INT64, [1], [-1]),
        ),
        helper.make_node(
            "ConstantOfShape",
            [shape],
            [mscores],
            value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0]),
        ),
        helper.make_node("Identity", [score], [covis]),
    ]
    return helper.make_graph(
        nodes, f"{prefix}_reject", [], matcher_output_info([matches, mscores, covis])
    )


def stop_condition(prefix: str, score: str, exponent: int):
    """Nodes computing {prefix}_stop: whether the score is below the threshold
    min_covisibility * (1 - early_stop_decay) ** exponent."""

    def constant(name, value):
        tensor = helper.make_tensor("v", TensorProto.FLOAT, [], [value])
        return helper.make_node("Constant", [], [f"{prefix}_{name}"], value=tensor)

    return [
        constant("one", 1.0),
        constant("exponent", float(exponent)),
        helper.make_node(
            "Sub", [f"{prefix}_one", "early_stop_decay"], [f"{prefix}_keep"]
        ),
        helper.make_node(
            "Pow", [f"{prefix}_keep", f"{prefix}_exponent"], [f"{prefix}_gamma"]
        ),
        helper.make_node(
            "Mul", ["min_covisibility", f"{prefix}_gamma"], [f"{prefix}_threshold"]
        ),
        helper.make_node("Less", [score, f"{prefix}_threshold"], [f"{prefix}_stop"]),
    ]


def early_stop_check(k: int, n_layers: int, outputs, continue_branch):
    """Nodes that stop after layer k if its covisibility score is low, or else
    run continue_branch (the remaining layers). Both produce the outputs."""
    prefix, score = f"early_stop_layer{k}", f"covisibility_score_layer{k}"
    if_node = helper.make_node(
        "If",
        [f"{prefix}_stop"],
        outputs,
        name=f"{prefix}_if",
        then_branch=reject_branch(prefix, score),
        else_branch=continue_branch,
    )
    return stop_condition(prefix, score, n_layers - k) + [if_node]


def add_early_stopping(path: Path, check_layers, n_layers: int):
    """Turn the graph of LayerScoresMatcherONNX into nested If nodes.

    After each layer k in check_layers, the pair is rejected without running
    the remaining layers and the readout if its covisibility score after layer
    k is below min_covisibility * (1 - early_stop_decay) ** (n_layers - k), as
    FlexGlue.match(..., early_stop_decay=...). The nodes are split into stages
    (the nodes needed for the score after the first checked layer, those
    needed in addition for the next one, ..., and the rest for the outputs),
    and each stage after a check goes into the else-branch of an If node.
    Adds the scalar input early_stop_decay: 1 never stops early (nor does
    min_covisibility = 0). When stopping early, matches0 is -1, mscores0 is 0
    and covisibility_score is the score of the last layer run.
    """
    model = onnx.load(str(path))
    graph = model.graph
    nodes = list(graph.node)
    score_names = [f"covisibility_score_layer{k}" for k in check_layers]
    final_names = [o.name for o in graph.output if o.name not in score_names]
    assert final_names == MATCHER_OUTPUTS, final_names
    stages = stage_node_indices(nodes, [[n] for n in score_names] + [final_names])
    inits = stage_initializers(graph, nodes, stages)

    # Build the nested Ifs from the innermost check to the outermost. The last
    # stage produces renamed outputs, since the If node around it does.
    outputs = [f"{n}_full" for n in final_names]
    stage_nodes = rename_values(
        [nodes[i] for i in stages[-1]], dict(zip(final_names, outputs))
    )
    graph_name = "layers_after_last_check"
    for c in reversed(range(len(check_layers))):
        body = helper.make_graph(
            stage_nodes,
            graph_name,
            [],
            matcher_output_info(outputs),
            initializer=inits[c + 1],
        )
        k = check_layers[c]
        outputs = (
            final_names
            if c == 0
            else [f"{n}_after_layer{check_layers[c - 1]}" for n in final_names]
        )
        stage_nodes = [nodes[i] for i in stages[c]]
        stage_nodes += early_stop_check(k, n_layers, outputs, body)
        graph_name = f"layers_after_check{c}"

    del graph.node[:]
    graph.node.extend(stage_nodes)
    del graph.initializer[:]
    graph.initializer.extend(inits[0])
    del graph.output[:]
    graph.output.extend(matcher_output_info(final_names))
    graph.input.append(
        helper.make_tensor_value_info("early_stop_decay", TensorProto.FLOAT, [])
    )
    del graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_descriptor(descriptor, path: Path, size, opset: int):
    model = DescriptorONNX(descriptor).eval()
    image = torch.rand(1, 3, *size)
    torch.onnx.export(
        model,
        (image,),
        str(path),
        input_names=["image"],
        output_names=["features"],
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )


def dummy_matcher_inputs(size, n0=1000, n1=1200, device="cpu"):
    h, w = size[0] // 8, size[1] // 8
    image_size0 = torch.tensor([[1024.0, 768.0]], device=device)
    image_size1 = torch.tensor([[800.0, 1000.0]], device=device)
    return (
        torch.randn(1, 256, h, w, device=device),
        torch.randn(1, 256, h, w, device=device),
        torch.rand(1, n0, 2, device=device) * image_size0[:, None],
        torch.rand(1, n1, 2, device=device) * image_size1[:, None],
        image_size0,
        image_size1,
        torch.tensor(0.1, device=device),
        torch.tensor(3.0, device=device),
        torch.tensor(0.0, device=device),
    )


MATCHER_INPUTS = [
    "features0",
    "features1",
    "kpts0",
    "kpts1",
    "image_size0",
    "image_size1",
    "filter_threshold",
    "max_distance_px",
    "min_covisibility",
]
MATCHER_OUTPUTS = ["matches0", "mscores0", "covisibility_score"]
KEYPOINT_DIMS = {
    "kpts0": "num_keypoints0",
    "kpts1": "num_keypoints1",
    "matches0": "num_keypoints0",
    "mscores0": "num_keypoints0",
    "p2d0_i1": "num_keypoints0",
    "confidence0": "num_keypoints0",
    "p2d1_i0": "num_keypoints1",
    "confidence1": "num_keypoints1",
}


def make_keypoint_dims_dynamic(path: Path):
    """Remove shapes specialized to the dummy number of keypoints.

    The exporter may annotate intermediate values (value_info) and outputs with
    the concrete number of keypoints of the dummy inputs. ONNX Runtime uses
    these annotations in graph optimizations (e.g. MatMul+Add -> Gemm fusion
    with static reshapes), which then fail for other numbers of keypoints.
    Clear the intermediate annotations and name the dynamic dimensions.
    """
    model = onnx.load(str(path))
    del model.graph.value_info[:]
    for value in list(model.graph.input) + list(model.graph.output):
        if value.name in KEYPOINT_DIMS:
            dim = value.type.tensor_type.shape.dim[1]
            dim.Clear()
            dim.dim_param = KEYPOINT_DIMS[value.name]
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def export_matcher(
    matcher,
    path: Path,
    size,
    opset: int,
    fp16: bool = False,
    fused: bool = True,
    debug_outputs: bool = False,
    early_stop_layers=None,
):
    """Export a copy of the matcher (the export modifies it in place)."""
    matcher = copy.deepcopy(matcher)
    output_names = MATCHER_OUTPUTS + (DEBUG_OUTPUTS if debug_outputs else [])
    if early_stop_layers:
        assert not debug_outputs
        check_layers = sorted(early_stop_layers)
        assert 1 <= check_layers[0] and check_layers[-1] < matcher.conf.n_layers
        model = LayerScoresMatcherONNX(matcher, check_layers, fp16=fp16).eval()
        output_names += [f"covisibility_score_layer{k}" for k in check_layers]
    else:
        model = SparseMatcherONNX(matcher, fp16, debug_outputs).eval()
    inputs = dummy_matcher_inputs(size)
    # torch.export specializes dimensions of size 0 and 1, hence min=2. The
    # exported graph has no such constraint, and N = 1 is checked in the CI.
    n0 = torch.export.Dim("num_keypoints0", min=2, max=65536)
    n1 = torch.export.Dim("num_keypoints1", min=2, max=65536)
    dynamic_shapes = {name: None for name in MATCHER_INPUTS}
    dynamic_shapes["kpts0"], dynamic_shapes["kpts1"] = {1: n0}, {1: n1}
    with exportable(matcher, fused):
        torch.onnx.export(
            model,
            inputs,
            str(path),
            input_names=MATCHER_INPUTS,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=opset,
            dynamo=True,
            external_data=False,
            custom_translation_table={
                torch.ops.flexglue.multi_head_attention.default: (
                    multi_head_attention_onnx
                ),
                torch.ops.flexglue.gelu.default: gelu_onnx,
            },
        )
    make_keypoint_dims_dynamic(path)
    if early_stop_layers:
        add_early_stopping(path, check_layers, matcher.conf.n_layers)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS_URL, help="Path or URL.")
    parser.add_argument(
        "--output_dir", type=Path, default=Path(__file__).resolve().parent / "models"
    )
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Cast the transformer layers of the matcher to float16 (for CUDA).",
    )
    parser.add_argument(
        "--no_fused_attention",
        action="store_true",
        help="Use standard ONNX operators instead of the fused attention and "
        "GELU operators of ONNX Runtime.",
    )
    parser.add_argument(
        "--early_stop_layers",
        default="",
        help="Comma-separated layers after which the matcher may stop early on "
        "non-covisible pairs, e.g. 2,4,6,8 (adds the early_stop_decay input).",
    )
    parser.add_argument(
        "--matcher_only",
        action="store_true",
        help="Only export the matcher.",
    )
    args = parser.parse_args()
    args.early_stop_layers = [int(k) for k in args.early_stop_layers.split(",") if k]
    args.matcher_name = (
        "flexglue_matcher"
        + ("_early_stop" if args.early_stop_layers else "")
        + ("_fp16" if args.fp16 else "")
        + ".onnx"
    )
    return args


def print_exported(path: Path, start: float):
    print(
        f"Exported {path} in {time.time() - start:.0f}s "
        f"({path.stat().st_size / 1e6:.1f}MB, sha256={sha256sum(path)})"
    )


def main():
    args = parse_args()
    size = tuple(args.size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.matcher_only:
        descriptor = DINOv3Descriptor(weights=args.weights, resize=size).eval()
        path, t = args.output_dir / "flexglue_descriptor.onnx", time.time()
        export_descriptor(descriptor, path, size, args.opset)
        print_exported(path, t)
    matcher = FlexGlue(weights=args.weights, describe=False, mp=None).eval()
    path, t = args.output_dir / args.matcher_name, time.time()
    export_matcher(
        matcher,
        path,
        size,
        args.opset,
        fp16=args.fp16,
        fused=not args.no_fused_attention,
        early_stop_layers=args.early_stop_layers,
    )
    print_exported(path, t)


if __name__ == "__main__":
    main()
