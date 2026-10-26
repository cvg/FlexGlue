"""Match two images with the FlexGlue ONNX models, using only ONNX Runtime.

This example runs the sparse matcher exported by onnx/export.py, which is also
used by COLMAP. It doesn't need PyTorch, only numpy, opencv-python and
onnxruntime (or onnxruntime-gpu for CUDA).

FlexGlue matches keypoints through dense image features, in two steps:
1. Describe each image once: the image is resized (squashed) to the input
   resolution of the descriptor model (512 x 512) and encoded into a dense
   feature map of shape [1, 256, 64, 64].
2. Match a pair of feature maps: the matcher takes the keypoints of both images
   and returns, for each keypoint of image 0, the index of its match in image 1
   (or -1), a match confidence, and the covisibility score of the pair. All
   matches are rejected if the covisibility score is below min_covisibility,
   which filters out look-alike pairs (doppelgangers).

Keypoints can come from any detector. They are in pixels, with the origin at
the top-left corner of the image (the center of the top-left pixel is at
(0.5, 0.5)). This example uses SIFT keypoints from OpenCV.

The models are downloaded from the release page on the first run. On CUDA, the
matcher with float16 transformer layers is used by default.

Usage (from the root of the repository):
    python onnx/match_pair.py assets/berlin1.jpg assets/berlin2.jpg \\
        --output matches.png
"""

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

RELEASE_URL = "https://github.com/cvg/FlexGlue/releases/download/v0.1/"
MODELS = {
    "flexglue_descriptor.onnx": (
        "93fab8a4ccde1b7331640915357d96cff0929ea87134872f37f7b0150d9914ff"
    ),
    "flexglue_matcher.onnx": (
        "a0baf84f00b32fc613206511a464fc3e71f4321e9d34df3a83362bd7d2a73012"
    ),
    "flexglue_matcher_fp16.onnx": (
        "f81cb68f27367171728c2c51ea5b60acab348520bc12154d4a3befa79f2dfe11"
    ),
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_model(name: str, model_dir: Path) -> Path:
    """Return the path to a model, downloading it from the release if needed."""
    path = model_dir / name
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {RELEASE_URL + name} to {path}")
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(RELEASE_URL + name, tmp)
        if sha256sum(tmp) != MODELS[name]:
            tmp.unlink()
            raise RuntimeError(f"Checksum mismatch for {name}.")
        tmp.rename(path)
    return path


def read_image(path: str) -> np.ndarray:
    """Read an image as an RGB float32 array (H, W, 3) in [0, 1]."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def detect_keypoints(image: np.ndarray, num_keypoints: int) -> np.ndarray:
    """SIFT keypoints (N, 2) in pixels, with the origin at the image corner."""
    gray = cv2.cvtColor((image * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
    keypoints = cv2.SIFT_create(nfeatures=num_keypoints).detect(gray, None)
    keypoints = sorted(keypoints, key=lambda k: -k.response)
    # OpenCV returns a location once per dominant orientation: deduplicate.
    points = np.array([k.pt for k in keypoints], dtype=np.float32)
    _, first = np.unique(points, axis=0, return_index=True)
    points = points[np.sort(first)][:num_keypoints]
    # OpenCV puts the center of the top-left pixel at (0, 0), FlexGlue at (0.5, 0.5).
    return points + 0.5


def describe(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    """Dense features [1, 256, S/8, S/8] of an image.

    The image is squashed to the S x S input resolution of the model with a
    bilinear resize without antialiasing, as torch.nn.functional.interpolate
    (mode="bilinear", align_corners=False) does in the PyTorch implementation.
    """
    height, width = session.get_inputs()[0].shape[-2:]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    (features,) = session.run(None, {"image": resized.transpose(2, 0, 1)[None]})
    return features


def match(
    session: ort.InferenceSession,
    features0: np.ndarray,
    features1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    image_size0: tuple,
    image_size1: tuple,
    filter_threshold: float = 0.1,
    max_distance_px: float = 3.0,
    min_covisibility: float = 0.0,
):
    """Match the keypoints of two images given their dense features.

    Args:
        features0, features1: dense features of each image, from describe().
        kpts0, kpts1: keypoints (N0, 2) and (N1, 2) in pixels of the original
            images, with the origin at the top-left corner. The model requires
            N0, N1 >= 1: without keypoints, no matches are returned and the
            model is not run.
        image_size0, image_size1: (width, height) of the original images.
        filter_threshold: minimum match confidence.
        max_distance_px: maximum distance between a keypoint and the predicted
            location of its match, in pixels of the 512 x 512 resized images:
            in the original images, it is scaled by the long image side / 512,
            averaged over both images. inf disables the check, while 0 rejects
            all matches (unlike in PyTorch, where 0 disables it).
        min_covisibility: reject all matches if the covisibility score of the
            pair is lower than this value (0 disables the check).

    Returns:
        matches: (K, 2) indices into kpts0 and kpts1.
        scores: (K,) match confidences.
        covisibility_score: probability that the two images overlap (nan
            without keypoints).
    """
    if len(kpts0) == 0 or len(kpts1) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, np.float32), np.nan
    matches0, mscores0, covisibility_score = session.run(
        None,
        {
            "features0": features0,
            "features1": features1,
            "kpts0": kpts0[None].astype(np.float32),
            "kpts1": kpts1[None].astype(np.float32),
            "image_size0": np.array([image_size0], dtype=np.float32),
            "image_size1": np.array([image_size1], dtype=np.float32),
            "filter_threshold": np.array(filter_threshold, dtype=np.float32),
            "max_distance_px": np.array(max_distance_px, dtype=np.float32),
            "min_covisibility": np.array(min_covisibility, dtype=np.float32),
        },
    )
    matches0, mscores0 = matches0[0], mscores0[0]
    valid = matches0 > -1
    matches = np.stack([np.flatnonzero(valid), matches0[valid]], -1)
    return matches, mscores0[valid], float(covisibility_score[0])


def draw_matches(image0, image1, kpts0, kpts1, matches, scores, path):
    """Draw the matches side by side, colored by confidence (red to green)."""
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = (image0 * 255).round()
    canvas[:h1, w0:] = (image1 * 255).round()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for (i, j), score in zip(matches, scores):
        color = (0, int(255 * score), int(255 * (1 - score)))  # BGR
        p0 = tuple(np.round(kpts0[i] - 0.5).astype(int))
        p1 = tuple(np.round(kpts1[j] - 0.5 + [w0, 0]).astype(int))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    print(f"Saved the matches to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image0")
    parser.add_argument("image1")
    parser.add_argument("--num_keypoints", type=int, default=4096)
    parser.add_argument("--filter_threshold", type=float, default=0.1)
    parser.add_argument(
        "--max_distance_px",
        type=float,
        default=3.0,
        help="Maximum distance to the predicted location of a match, in pixels "
        "of the 512 x 512 resized images (inf disables the check).",
    )
    parser.add_argument("--min_covisibility", type=float, default=0.0)
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path.home() / ".cache/flexglue/onnx",
        help="Where to find or download the models.",
    )
    parser.add_argument("--cpu", action="store_true", help="Don't use CUDA.")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use the float32 matcher on CUDA (always used on CPU).",
    )
    parser.add_argument("--output", type=Path, help="Path of the visualization.")
    args = parser.parse_args()

    ort.set_default_logger_severity(3)  # errors only
    providers = ["CPUExecutionProvider"]
    use_cuda = not args.cpu and "CUDAExecutionProvider" in ort.get_available_providers()
    if use_cuda:
        providers.insert(0, "CUDAExecutionProvider")
    matcher_name = (
        "flexglue_matcher_fp16.onnx"
        if use_cuda and not args.fp32
        else "flexglue_matcher.onnx"
    )
    print(f"ONNX Runtime {ort.__version__}, providers: {providers}")
    descriptor = ort.InferenceSession(
        str(get_model("flexglue_descriptor.onnx", args.model_dir)), providers=providers
    )
    matcher = ort.InferenceSession(
        str(get_model(matcher_name, args.model_dir)), providers=providers
    )

    images, keypoints, features = [], [], []
    for path in (args.image0, args.image1):
        image = read_image(path)
        kpts = detect_keypoints(image, args.num_keypoints)
        t = time.time()
        feats = describe(descriptor, image)
        print(
            f"{path}: {image.shape[1]}x{image.shape[0]}, {len(kpts)} keypoints, "
            f"descriptor {1e3 * (time.time() - t):.0f}ms"
        )
        images.append(image)
        keypoints.append(kpts)
        features.append(feats)

    sizes = [(im.shape[1], im.shape[0]) for im in images]
    t = time.time()
    matches, scores, covisibility_score = match(
        matcher,
        *features,
        *keypoints,
        *sizes,
        filter_threshold=args.filter_threshold,
        max_distance_px=args.max_distance_px,
        min_covisibility=args.min_covisibility,
    )
    print(
        f"{matcher_name}: {len(matches)} matches, covisibility score "
        f"{covisibility_score:.3f}, {1e3 * (time.time() - t):.0f}ms "
        "(the first run includes the initialization)"
    )
    if covisibility_score < args.min_covisibility:
        print(f"The pair is rejected: covisibility score < {args.min_covisibility}.")

    if args.output:
        draw_matches(*images, *keypoints, matches, scores, args.output)


if __name__ == "__main__":
    main()
