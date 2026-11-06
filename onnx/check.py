"""Check FlexGlue models exported to ONNX against the PyTorch implementation.

Runs ONNX Runtime and PyTorch (with the given checkpoint) on image pairs from
the assets, with SIFT keypoints, and compares:
- for the descriptor: the dense features;
- for the matcher: the matches, match scores and covisibility scores, and the
  predicted points and confidences of all keypoints. The latter are not
  outputs of the matcher: they are compared on a debug graph, exported from
  the checkpoint with the same options as the matcher (onnx/export.py).

The options of the matcher (float16 transformer layers, fused operators of
ONNX Runtime, early stopping) and the input resolution are read from the
models. A matcher with early stopping is compared with early_stop_decay=1,
which never stops early. If it checks the covisibility score after every layer
but the last, as FlexGlue does (export with --early_stop_layers 1,...,8), it is
also compared with early stopping, for early_stop_decay in EARLY_STOP_DECAYS.
Otherwise, the outputs with early stopping are only printed.

Requires onnx, onnxscript and onnxruntime (onnxruntime-gpu on GPU).

Usage (from the root of the repository):
    python onnx/check.py --descriptor onnx/models/flexglue_descriptor.onnx \\
        --matcher onnx/models/flexglue_matcher.onnx
    python onnx/check.py --matcher onnx/models/flexglue_matcher_fp16.onnx
"""

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

# Also run from a clone of the repository without installing it.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from export import DEBUG_OUTPUTS, MATCHER_OUTPUTS, export_matcher  # noqa: E402

from flexglue import DINOv3Descriptor, FlexGlue  # noqa: E402
from flexglue.flexglue import WEIGHTS_URL  # noqa: E402
from flexglue.utils import load_image  # noqa: E402
from flexglue.utils import preprocess as pp  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CHECK_PAIRS = [
    ("berlin1.jpg", "berlin2.jpg"),
    ("lamar_cab_1446.jpg", "lamar_cab_1450.jpg"),
]
CHECK_MIN_COVISIBILITY = (0.0, 0.8)
EARLY_STOP_DECAYS = (0.0, 0.2, 0.5)

# Tolerances of the check, for float32 on CPU / float32 on CUDA / float16 on
# CUDA. On CPU, ONNX Runtime and PyTorch agree to ~1e-5. On CUDA, ONNX Runtime
# runs float32 matrix multiplications in TF32 (median point error ~0.01px), so a
# few matches at the decision boundaries may flip, and a few points may jump
# between near-equal peaks of their correspondence logits (jumps: fraction of
# points moving by more than 1px, or of match scores changing by more than 0.01).
TOLERANCES = {
    "cpu": dict(
        features=1e-3, px=1e-3, jumps=0.005, confidence=1e-4, score=1e-4, flips=1
    ),
    "cuda": dict(
        features=2e-2, px=0.05, jumps=0.02, confidence=5e-3, score=5e-3, flips=2
    ),
    "cuda_fp16": dict(
        features=2e-2, px=0.05, jumps=0.03, confidence=2e-2, score=1e-2, flips=3
    ),
}


def subgraphs(graph):
    """The graph and, recursively, the graphs of its If nodes."""
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from subgraphs(attr.g)


def input_size(model, name: str, stride: int = 1):
    """Image size (h, w) from the static spatial dimensions of an input."""
    value = next(i for i in model.graph.input if i.name == name)
    h, w = (d.dim_value for d in value.type.tensor_type.shape.dim[2:])
    return h * stride, w * stride


def matcher_options(path: Path):
    """Options of an exported matcher, to export the equivalent debug graph."""
    model = onnx.load(str(path))
    graphs = list(subgraphs(model.graph))
    return SimpleNamespace(
        size=input_size(model, "features0", stride=8),
        opset=next(o.version for o in model.opset_import if o.domain == ""),
        fp16=any(
            init.data_type == onnx.TensorProto.FLOAT16
            for g in graphs
            for init in g.initializer
        ),
        fused=any(
            node.domain == "com.microsoft" and node.op_type == "MultiHeadAttention"
            for g in graphs
            for node in g.node
        ),
        early_stop="early_stop_decay" in [i.name for i in model.graph.input],
        # Layers after which the graph may stop early (see add_early_stopping).
        early_stop_layers=sorted(
            int(m.group(1))
            for g in graphs
            for node in g.node
            if (m := re.fullmatch(r"early_stop_layer(\d+)_if", node.name))
        ),
    )


def onnx_providers(cpu: bool):
    if not cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def debug_session(matcher, options, providers):
    """Session of the debug graph: the matcher with the DEBUG_OUTPUTS."""
    t = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "debug_matcher.onnx"
        export_matcher(
            matcher,
            path,
            options.size,
            options.opset,
            fp16=options.fp16,
            fused=options.fused,
            debug_outputs=True,
        )
        session = ort.InferenceSession(str(path), providers=providers)
    print(f"Exported the debug matcher in {time.time() - t:.0f}s")
    return session


def sift_keypoints(path: str, num_keypoints: int) -> torch.Tensor:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=num_keypoints)
    kpts = sorted(sift.detect(gray, None), key=lambda k: -k.response)
    # OpenCV returns the same location once per dominant orientation. Remove
    # duplicates, since ties in the mutual nearest neighbor search are broken
    # arbitrarily and would make the comparison flaky.
    pts = np.unique(np.array([k.pt for k in kpts], dtype=np.float32), axis=0)
    order = np.random.default_rng(0).permutation(len(pts))[:num_keypoints]
    # OpenCV puts the center of the top-left pixel at (0, 0), FlexGlue the
    # top-left corner of the image.
    return torch.from_numpy(pts[order] + 0.5)[None]


def median_and_fraction_above(err: np.ndarray, threshold: float):
    """Median of the errors and fraction of them above threshold (0 if empty)."""
    if not err.size:
        return 0.0, 0.0
    return np.median(err), (err > threshold).mean()


def describe_pytorch(descriptor, image: torch.Tensor, size):
    device = next(descriptor.parameters()).device
    with torch.no_grad():
        return descriptor({"image": image[None].to(device)}, resize=size)


def descriptor_error(session, image: torch.Tensor, features, size) -> float:
    """Max abs error between the features of ONNX Runtime and PyTorch."""
    # Identical resize in both runtimes.
    x = pp.resize_images(image[None], size)
    features_ort = session.run(None, {"image": x.numpy()})[0]
    return np.abs(features_ort - features["features"].cpu().numpy()).max()


def match_pytorch(
    matcher,
    features0,
    features1,
    kpts0,
    kpts1,
    min_covisibility,
    early_stop_decay=None,
):
    """Predictions of FlexGlue.match as in the exported graph, as numpy arrays."""
    device = features0["features"].device
    data0 = {**features0, "keypoints": kpts0.to(device)}
    data1 = {**features1, "keypoints": kpts1.to(device)}
    with torch.no_grad():
        pred = matcher.match(
            data0,
            data1,
            sparse=True,
            mp=None,
            exact_nn=True,
            min_covisibility=min_covisibility,
            early_stop_decay=early_stop_decay,
        )
    return {k: v.cpu().numpy() for k, v in pred.items() if isinstance(v, torch.Tensor)}


def matcher_feeds(features0, features1, kpts0, kpts1, min_covisibility):
    return {
        "features0": features0["features"].cpu().numpy(),
        "features1": features1["features"].cpu().numpy(),
        "kpts0": kpts0.numpy(),
        "kpts1": kpts1.numpy(),
        "image_size0": features0["image_size"].cpu().numpy(),
        "image_size1": features1["image_size"].cpu().numpy(),
        "filter_threshold": np.array(0.1, np.float32),
        "max_distance_px": np.array(3.0, np.float32),
        "min_covisibility": np.array(min_covisibility, np.float32),
    }


def compare_matches(pred, outputs, tol):
    """Compare the matches, match scores and covisibility scores."""
    m0_ort, s0_ort, covis_ort = (x[0] for x in outputs[:3])
    m0_pt, s0_pt = pred["matches0"][0], pred["matching_scores0"][0]
    covis_pt = pred["covisibility_score"][0]
    valid_pt, valid_ort = m0_pt > -1, m0_ort > -1
    same = valid_pt & valid_ort & (m0_pt == m0_ort)
    flips = (valid_pt | valid_ort).sum() - same.sum()
    # As for the points (see compare_points), the score of a match may change
    # where its predicted point jumps: count them.
    score_median, score_jumps = median_and_fraction_above(
        np.abs(s0_ort[same] - s0_pt[same]), 0.01
    )
    ok = (
        flips <= tol["flips"]
        and score_median <= tol["score"]
        and score_jumps <= tol["jumps"]
        and abs(covis_ort - covis_pt) <= tol["score"]
    )
    message = (
        f"matches pt={valid_pt.sum():5d} ort={valid_ort.sum():5d} flips={flips} "
        f"scores: median={score_median:.1e} >0.01={score_jumps:.1%} "
        f"covis pt={covis_pt:.4f} ort={covis_ort:.4f}"
    )
    return bool(ok), message


def compare_points(pred, debug, tol):
    """Compare the per-keypoint predictions of the debug graph.

    Predicted points (pixels) of the keypoints confident enough to be matched,
    and confidences of all keypoints. A few points may jump between two
    near-equal peaks of their correspondence logits: count them instead of
    taking a maximum.
    """
    pairs = (("p2d0_i1", "confidence0"), ("p2d1_i0", "confidence1"))
    px = np.concatenate(
        [np.linalg.norm(debug[p] - pred[p], axis=-1)[pred[c] >= 0.1] for p, c in pairs]
    )
    px_median, px_jumps = median_and_fraction_above(px, 1.0)
    conf_err = np.concatenate([np.abs(debug[c] - pred[c]) for _, c in pairs], -1)
    conf_p99 = np.quantile(conf_err, 0.99)
    ok = (
        px_median <= tol["px"]
        and px_jumps <= tol["jumps"]
        and conf_p99 <= tol["confidence"]
    )
    message = (
        f"points: median={px_median:.1e}px >1px={px_jumps:.1%} (of {px.size}) "
        f"conf p99={conf_p99:.1e}"
    )
    return bool(ok), message


def time_session(session, feeds, warmup: int = 2, runs: int = 10) -> float:
    """Average run time in milliseconds."""
    for _ in range(warmup):
        session.run(None, feeds)
    t = time.time()
    for _ in range(runs):
        session.run(None, feeds)
    return (time.time() - t) * 1000 / runs


def check_matches(sessions, matcher, features, kpts, min_covisibility, tol, timing):
    """Compare the matchers on one pair of images and keypoints."""
    pred = match_pytorch(matcher, *features, *kpts, min_covisibility)
    feeds = matcher_feeds(*features, *kpts, min_covisibility)
    match_feeds = feeds
    if sessions.early_stop:  # early_stop_decay=1 never stops early
        match_feeds = {**feeds, "early_stop_decay": np.array(1.0, np.float32)}
    outputs = sessions.matcher.run(None, match_feeds)
    ok, message = compare_matches(pred, outputs, tol)
    if min_covisibility == 0:  # otherwise, the points may not be computed
        outputs = sessions.debug.run(None, feeds)[len(MATCHER_OUTPUTS) :]
        points_ok, points_message = compare_points(
            pred, dict(zip(DEBUG_OUTPUTS, outputs)), tol
        )
        ok &= points_ok
        message += " " + points_message
    if timing:
        message += f" ort_time={time_session(sessions.matcher, match_feeds):.0f}ms"
    num_keypoints = "/".join(str(k.shape[1]) for k in kpts)
    print(
        f"  N={num_keypoints} min_covis={min_covisibility:.1f}: {message} "
        f"{'OK' if ok else 'FAIL'}"
    )
    if sessions.early_stop and min_covisibility > 0:
        if sessions.early_stop_comparable:
            for decay in EARLY_STOP_DECAYS:
                ok &= check_early_stop(
                    sessions.matcher, matcher, features, kpts, feeds, decay, tol
                )
        else:
            print_early_stop(sessions.matcher, feeds, early_stop_decay=0.5)
    return bool(ok)


def check_early_stop(session, matcher, features, kpts, feeds, decay, tol):
    """Compare the matchers with early stopping, if both check the same layers."""
    min_covisibility = float(feeds["min_covisibility"])
    pred = match_pytorch(
        matcher, *features, *kpts, min_covisibility, early_stop_decay=decay
    )
    outputs = session.run(
        None, {**feeds, "early_stop_decay": np.array(decay, np.float32)}
    )
    ok, message = compare_matches(pred, outputs, tol)
    print(f"    early_stop_decay={decay}: {message} {'OK' if ok else 'FAIL'}")
    return ok


def print_early_stop(session, feeds, early_stop_decay: float):
    """Outputs with early stopping, not compared: the graph checks fewer layers
    than FlexGlue, which checks every layer but the last."""
    feeds = {**feeds, "early_stop_decay": np.array(early_stop_decay, np.float32)}
    matches0, _, covisibility_score = session.run(None, feeds)
    print(
        f"    early_stop_decay={early_stop_decay}: "
        f"matches={(matches0[0] > -1).sum()} covis={covisibility_score[0]:.4f}"
    )


def check_pair(sessions, models, names, size, num_keypoints, tol, timing):
    """Compare the descriptors and the matchers on a pair of asset images."""
    paths = [ASSETS / name for name in names]
    images = [load_image(p) for p in paths]
    features = [describe_pytorch(models.descriptor, im, size) for im in images]
    all_ok, line = True, f"[{names[0]} / {names[1]}]"
    if sessions.descriptor is not None:
        error = max(
            descriptor_error(sessions.descriptor, im, f, size)
            for im, f in zip(images, features)
        )
        all_ok = bool(error <= tol["features"])
        line += f" descriptor max abs error: {error:.2e} {'OK' if all_ok else 'FAIL'}"
    print(line)
    if sessions.matcher is None:
        return all_ok
    for n in num_keypoints:
        kpts = [sift_keypoints(str(p), n) for p in paths]
        for min_covisibility in CHECK_MIN_COVISIBILITY:
            all_ok &= check_matches(
                sessions, models.matcher, features, kpts, min_covisibility, tol, timing
            )
    return all_ok


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--weights", default=WEIGHTS_URL, help="PyTorch checkpoint (path or URL)."
    )
    parser.add_argument("--descriptor", type=Path, help="Descriptor ONNX model.")
    parser.add_argument("--matcher", type=Path, help="Matcher ONNX model.")
    parser.add_argument("--num_keypoints", default="512,2048")
    parser.add_argument("--timing", action="store_true", help="Time the matcher.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.descriptor is None and args.matcher is None:
        parser.error("At least one of --descriptor and --matcher is required.")
    return args


def main():
    args = parse_args()
    providers = onnx_providers(args.cpu)
    device = "cuda" if providers[0].startswith("CUDA") else "cpu"
    print(f"ONNX Runtime {ort.__version__}, providers: {providers}")
    sessions = SimpleNamespace(
        descriptor=None,
        matcher=None,
        debug=None,
        early_stop=False,
        early_stop_comparable=False,
    )
    models = SimpleNamespace(descriptor=None, matcher=None)
    sizes, precision = set(), ""
    if args.descriptor is not None:
        descriptor = onnx.load(str(args.descriptor), load_external_data=False)
        sizes.add(input_size(descriptor, "image"))
        sessions.descriptor = ort.InferenceSession(
            str(args.descriptor), providers=providers
        )
    if args.matcher is not None:
        options = matcher_options(args.matcher)
        print(f"Matcher options: {vars(options)}")
        if options.fp16 and device == "cpu":
            raise SystemExit("The float16 matcher requires CUDA.")
        sizes.add(options.size)
        precision = "_fp16" if options.fp16 else ""
        models.matcher = FlexGlue(weights=args.weights, describe=False, mp=None)
        models.matcher.eval()
        sessions.matcher = ort.InferenceSession(str(args.matcher), providers=providers)
        sessions.debug = debug_session(models.matcher, options, providers)
        sessions.early_stop = options.early_stop
        sessions.early_stop_comparable = options.early_stop_layers == list(
            range(1, models.matcher.conf.n_layers)
        )
        models.matcher = models.matcher.to(device)
    if len(sizes) > 1:
        raise SystemExit(f"The models have different input sizes: {sizes}")
    (size,) = sizes
    models.descriptor = DINOv3Descriptor(weights=args.weights, resize=size)
    models.descriptor = models.descriptor.eval().to(device)

    tol = TOLERANCES[device + precision]
    num_keypoints = [int(n) for n in args.num_keypoints.split(",")]
    all_ok = True
    for names in CHECK_PAIRS:
        all_ok &= check_pair(
            sessions, models, names, size, num_keypoints, tol, args.timing
        )
    print("CHECK", "PASSED" if all_ok else "FAILED")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
