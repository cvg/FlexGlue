<p align="center">
  <h1 align="center"><ins>FlexGlue</ins><br>Sparse-to-Dense Feature Matching and Visual Disambiguation</h1>
  <h2 align="center">
    <a href="#" align="center">Paper</a> | 
    <a href="https://colab.research.google.com/github/cvg/FlexGlue/blob/main/demo.ipynb" align="center">Colab</a>
  </h2>

</p>
<p align="center">
    <img src="assets/matching_modes.jpg" alt="matching modes" width=90%>
    <br>
    <em>One model, four matching modes: FlexGlue provides matches tailored to each use-case.</em>
</p>
<p align="center">
    <img src="assets/hard_pairs.jpg" alt="hard pairs" width=90%>
    <br>
    <em>FlexGlue is robust under extreme viewpoint changes, low-texture environments, rotations, and lighting conditions.</em>
</p>
<p align="center">
    <img src="assets/doppelgangers.jpg" alt="doppelgangers" width=90%>
    <br>
    <em>To match or not to match: A covisibility tells whether a pair is an unmatchable doppelganger and FlexGlue stops early on the latter.</em>
</p>

##

This repository hosts the inference code of FlexGlue, a feature matcher that works with any keypoints, with dense grids, or with no keypoints at all.
FlexGlue matches through dense features: we use a DINOv3-ConvNeXt to encode an image into a stride-8 feature map, a transformer refines both maps jointly, and each query point finds its correspondence in the other image by correlation.
On top, FlexGlue predicts whether the two images are covisible at all, which rejects look-alike pairs such as repeated facades.

<p align="center">
    <img src="assets/pipeline.png" alt="pipeline" width=90%>
    <br>
    <em>A pair of images is encoded with a CNN backbone. Dense tokens are forwarded through a series of transformers gathering context within and between images.<br>The aggregated tokens are interpolated at sparse keypoints, and their coarse correspondence is estimated with a windowed soft-argmax.<br>A simple refinement head yields subpixel accuracy. Unmatchable image pairs are detected in every layer.</em>
</p>

## Installation and demo

Install this repo using pip:

```bash
git clone https://github.com/cvg/FlexGlue.git && cd FlexGlue
python -m pip install -e .
# or, without cloning:
python -m pip install git+https://github.com/cvg/FlexGlue.git
```

FlexGlue works with any keypoints. The examples below detect them with the extractors of [LightGlue](https://github.com/cvg/LightGlue):

```bash
# requires the latest LightGlue commit
python -m pip install git+https://github.com/cvg/LightGlue.git
```

We provide a [demo notebook](demo.ipynb) which shows all matching modes on an image pair and the rejection of a look-alike pair.

> [!IMPORTANT]
> FlexGlue assumes keypoints in corner coordinates (COLMAP convention): the center of the top-left pixel is at (0.5, 0.5). LightGlue assumes center coordinates (a 0.5 px offset), so pass `origin="corner"` to its detectors, e.g. `ALIKED(origin="corner")`.

Here is a minimal script to match two images. Detection is optional:

```python
from lightglue import ALIKED
from flexglue import FlexGlue
from flexglue.utils import load_image, pack_matches, rbd

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image("assets/sacre_coeur1.jpg").cuda()
image1 = load_image("assets/sacre_coeur2.jpg").cuda()
inputs0, inputs1 = {"image": image0[None]}, {"image": image1[None]}

# optional: detect keypoints to be matched
detector = ALIKED(origin="corner", max_num_keypoints=2048).eval().cuda()
inputs0.update(detector.extract(image0))  # {keypoints, ...}
inputs1.update(detector.extract(image1))

# match: each keypoint (or sampled point) is paired with its predicted point in the other image
matcher = FlexGlue().eval().cuda()
matches01 = rbd(pack_matches(matcher.match(inputs0, inputs1)))
matches = matches01["matches"]  # indices with shape (K,2)
points0 = matches01["keypoints0"][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = matches01["keypoints1"][matches[..., 1]]  # coordinates in image #1, shape (K,2)
```

With keypoints in both images, `matcher.match(inputs0, inputs1, sparse=True)` instead matches the keypoints to each other, as LightGlue does: `matches` then indexes `inputs0["keypoints"]` and `inputs1["keypoints"]`.

We also provide a convenience method to match a pair of images directly:

```python
matches01 = rbd(pack_matches(matcher.match_pair(image0, image1, detector=detector)))
points0 = matches01["keypoints0"][matches01["matches"][..., 0]]
points1 = matches01["keypoints1"][matches01["matches"][..., 1]]
```

## Matching modes

FlexGlue supports multiple matching modes, depending on whether keypoints are provided for each image and the selected mode (`sparse`):

| neither has keypoints | one has keypoints (`i`) | both have keypoints |
|---|---|---|
| sample correspondences | each keypoint of `i` with its predicted point in `j` | both directions concatenated (`sparse=False`) or mutual nearest neighbors (`sparse=True`) |

`matches` always indexes `keypoints0` and `keypoints1`. Without `sparse=True`, these are row-aligned lists of corresponding points (keypoints or sampled points paired with their predicted points), so row *k* of one matches row *k* of the other. With `sparse=True`, they are the input keypoints, matched to each other.

The snippet below shows how to run each mode, one after the other. The [demo notebook](demo.ipynb) shows the matches of each mode on an example pair.

```python
inputs0 = matcher.descriptor.describe(image0)  # {features, image_size}
inputs1 = matcher.descriptor.describe(image1)

# neither has keypoints: correspondences are sampled
pred = matcher.match(inputs0, inputs1)
# >>>
# keypoints0/1: (1, S, 2), sampled points
# matches0: (1, S), k for row k, or -1 if filtered

# one has keypoints: each keypoint of image #0 with its predicted point in image #1
inputs0.update(detector.extract(image0))  # {keypoints, ...}
pred = matcher.match(inputs0, inputs1)
# >>>
# keypoints0: (1, M, 2), detected keypoints <-- inputs0["keypoints"]
# keypoints1: (1, M, 2), predictions
# matches0: (1, M), k for row k, or -1 if filtered
# p2d0_i1: (1, M, 2), predicted points of all keypoints, also filtered ones

# both have keypoints: concatenate bidirectional (default)
inputs1.update(detector.extract(image1))
pred = matcher.match(inputs0, inputs1)
# >>>
# keypoints0: (1, M+N, 2), [keypoints0, p2d1_i0]
# keypoints1: (1, M+N, 2), [p2d0_i1, keypoints1]
# matches0: (1, M+N), k for row k, or -1 if filtered

# both have keypoints: mutual nearest neighbors between the keypoints
pred = matcher.match(inputs0, inputs1, sparse=True)
# >>>
# keypoints0/1: (1, M, 2) / (1, N, 2)
# matches0: (1, M), index of the matched keypoint in image #1, or -1

# all modes return the same output format
matches01 = rbd(pack_matches(pred))
matches = matches01["matches"]  # indices with shape (K,2)
points0 = matches01["keypoints0"][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = matches01["keypoints1"][matches[..., 1]]  # coordinates in image #1, shape (K,2)
```

FlexGlue can also return dense correspondences directly:

```python
# dense grids only, without matching
pred = matcher.match_dense(inputs0, inputs1)
# dense_p2d0_i1: (B, h, w, 2), h x w = describe_resize / coarse_stride (64x64 by default)
```

For increased sub-pixel accuracy, we refine dense correspondences with the refiners of [RoMa v2](https://github.com/Parskatt/RoMaV2), whose weights are downloaded automatically:

```python
# refined with the RoMa v2 refiners, at refine_resize resolution
matcher = FlexGlue(refine=True).eval().cuda()
pred = matcher.match_dense(inputs0, inputs1)
# dense_p2d0_i1: (B, h_r, w_r, 2), h_r x w_r = refine_resize (512x512 by default)
# or
pred = matcher.match(inputs0, inputs1)  # refined matches
```

The coarse features of each image can be computed once and cached, useful when matching a large image set exhaustive:

```python
from flexglue import DINOv3Descriptor

descriptor = DINOv3Descriptor().eval().cuda()
inputs0.update(descriptor.describe(image0))  # {features, image_size}
inputs1.update(descriptor.describe(image1))
pred = matcher.match(inputs0, inputs1)  # reuses the cached features
```

<details>
<summary>[Detail of all outputs - click to expand]</summary>

Inputs per image (`inputs0`, `inputs1`), as returned by `DINOv3Descriptor.describe`, merged with a LightGlue detector:
- `features`: `(B, 256, h, w)` stride-8 feature map.
- `image_size`: `(B, 2)` image size `(w, h)` in pixels.
- `image`: `(B, 3, H, W)` the image, read by the refiner.
- `keypoints`: `(B, M, 2)` keypoints in pixels, optional.
- `descriptors`: `(B, M, 64)` descriptors of the keypoints, with `descriptor.describe(image, keypoints)`.

Outputs of `matcher(...)` and `matcher.match(...)`, for each direction `i -> j` whose image `i` has keypoints:
- `p2d{i}_i{j}`: `(B, M, 2)` predicted location of each keypoint of image `i` in image `j`, in pixels.
- `p2d{i}_n{j}`: `(B, M, 2)` the same, in normalized coordinates `[-1, 1]`.
- `confidence{i}`: `(B, M)` confidence that the keypoint has a correspondence. With the cycle check, the geometric mean of the forward confidence and the confidence of the prediction mapped back to `i`.
- `valid{i}`: `(B, M)` whether the keypoint passes the covisibility, border and cycle checks.
- `cycle_error{i}`: `(B, M)` cycle error in pixels, with `max_distance_px` and not `sparse=True`.
- `covisibility_score`: `(B,)` probability that the two images show the same place.
- `stop_layer`: the layer at which the matcher stopped.

Matches, as in LightGlue, also returned by `matcher(...)` and `matcher.match(...)` in every mode:
- `keypoints0`, `keypoints1`: the points that `matches0` and `matches1` index.
  - With `sparse=True`: the input keypoints, `(B, M, 2)` and `(B, N, 2)`.
  - Otherwise: row-aligned lists of corresponding points, `(B, K, 2)`. K is M+N with keypoints in both images, M with keypoints in one, and at most `max_num_samples` without keypoints.
- `matches0`, `matches1`: index of the match in the other image, or -1.
  - With `sparse=True`: `(B, M)` and `(B, N)`, from mutual nearest neighbors.
  - Otherwise: `(B, K)`. Both are the same tensor: row k matches row k, or -1 if filtered out.
- `matching_scores0`, `matching_scores1`: confidence of each match, 0 where unmatched. Same shapes as `matches0` and `matches1`.

Added by `pack_matches(pred)`, not returned by the matcher:
- `matches`: list of `(S, 2)` index pairs per batch element.
- `scores`: list of `(S,)` match confidences per batch element.

Dense matches, from `match_dense` and the sampled mode, per direction `i -> j`. The coarse grid has `h x w` = `describe_resize / coarse_stride` cells (64x64 by default), the refined grid `h_r x w_r` = `refine_resize` (512x512 by default):
- `dense_p2d{i}_i{j}`: `(B, h, w, 2)` correspondence of each cell of image `i` in image `j`, in pixels. At the refiner resolution `(B, h_r, w_r, 2)` with `refine=True`.
- `dense_p2d{i}_n{j}`: the same, in normalized coordinates.
- `dense_confidence{i}`: `(B, h, w)` confidence of each correspondence, zero where it fails a check.
- `dense_valid{i}`: `(B, h, w)` whether the correspondence passes the covisibility, filter, border and cycle checks.
- `dense_cov{i}_i{j}`: `(B, h_r, w_r, 2, 2)` covariance of the refined correspondences in pixels, with `refine=True`.

</details>

## Look-alike pairs and early stopping

`pred["covisibility_score"]` is the probability that the two images show the same place. With `min_covisibility > 0`, all matches of a pair below this score are rejected.
With `early_stop_decay`, FlexGlue also stops at an early layer once the pair is confidently rejected, which saves most of the compute on non-pairs (no batch support):

```python
pred = matcher.match(inputs0, inputs1, min_covisibility=0.8, early_stop_decay=0.2)
pred["is_covisible"]  # whether covisibility_score >= min_covisibility, only with min_covisibility > 0
pred["stop_layer"]  # the layer at which FlexGlue stopped
```

Note that all matches of a pair are invalidated if its `covisibility_score` is below `min_covisibility`, and FlexGlue returns empty matches in this case.

## Batch inference

FlexGlue matches a batch of image pairs in a single call. Images are passed as a batched tensor `(B, 3, H, W)` or as a list of `(3, H, W)` tensors, which may have different sizes:

```python
images0 = [load_image(path).cuda() for path in paths0]  # B images, any size
images1 = [load_image(path).cuda() for path in paths1]

# images only, or with a detector that returns a fixed number of keypoints
detector = ALIKED(origin="corner", max_num_keypoints=2048, detection_threshold=0.0).eval().cuda()
pred = matcher.match_pair(images0, images1, detector=detector)

# or describe first, e.g. to cache the descriptors
inputs0 = matcher.descriptor.describe(images0)  # {features, image_size, image}
inputs1 = matcher.descriptor.describe(images1)
pred = matcher.match(inputs0, inputs1)

pred = pack_matches(pred)  # lists of matches and scores, one entry per pair
for b in range(len(images0)):
    matches = pred["matches"][b]  # (K, 2)
    points0 = pred["keypoints0"][b][matches[..., 0]]  # pixels of images0[b]
    points1 = pred["keypoints1"][b][matches[..., 1]]  # pixels of images1[b]
```

Batching has a few requirements:
- Images of different sizes are squashed to the same `describe_resize` (the default `(512, 512)`). With an int `describe_resize` or `refine_resize`, all images must share the same aspect ratio.
- All images need the same number of keypoints, for example a detector with `max_num_keypoints` and `detection_threshold=0.0`.
- Early stopping only stops once all pairs of the batch are rejected.

## Advanced configuration

<details>
<summary>[Detail of all parameters - click to expand]</summary>

`DINOv3Descriptor`:
- ```resize```: Size of the image fed to the backbone. An int sets the long side and keeps the aspect ratio, both sides floored to multiples of 32. A tuple ```(h, w)``` squashes the image to that size, ignoring the aspect ratio. Default: (512, 512).
- ```describe```: Add the descriptors of the keypoints when ```describe``` gets keypoints. Default: True.
- ```weights```: Path or URL of the descriptor weights. Default: the release weights.

`FlexGlue`:
- ```n_layers```: Number of transformer layers. Reduce this value for faster inference at the cost of accuracy. Can be overridden per call. Default: 9 (all layers).
- ```filter_threshold```: Match confidence. Increase this value to obtain less, but stronger matches. In sparse mode, it applies to the product of the confidences of both keypoints. Default: 0.1.
- ```max_distance_px```: Maximum cycle error: a prediction is mapped back to its source image and must land within this distance. In sparse mode, it instead bounds the distance of mutual nearest neighbors, without the cycle check. In pixels of the resized image fed to the descriptor. None disables it. Default: 3.0.
- ```min_covisibility```: Reject all matches of a pair whose ```covisibility_score``` is below this value. Default: 0.0 (off). Recommended: 0.8 for noisy image sets.
- ```early_stop_decay```: Controls the early stopping on rejected pairs: FlexGlue stops after layer ```k``` if ```covisibility_score < min_covisibility * (1 - early_stop_decay) ** (9 - k)```. A lower value stops more often at earlier layers. Requires ```min_covisibility > 0```. Default: None (off).
- ```remove_border_px```: Invalidate matches whose points lie within this many pixels of the image border, in pixels of the resized image as ```max_distance_px```. Default: 0.0.
- ```mp```: Precision of the transformer: ```"float16"```, ```"bfloat16"``` or ```None``` (float32). The readout always runs in float32. Default: ```"float16"```.
- ```describe_resize```: Size of the images fed to the built-in descriptor, for inputs without ```features```, as ```resize``` of ```DINOv3Descriptor```. Default: (512, 512).
- ```refine```: Build the RoMa v2 refiners, and refine the predictions at ```refine_resize```. Needs ```image``` in the inputs, as returned by ```DINOv3Descriptor.describe```. Can be disabled per call. Default: False.
- ```refine_resize```: Size of the images fed to the refiners, as ```resize``` of ```DINOv3Descriptor```. Default: (512, 512).
- ```refine_scales```: Strides of the RoMa v2 refiners, coarse to fine. Fixed at construction. Default: (2, 1), RoMa v2 also has (4, 2, 1).
- ```coarse_stride```: Pitch of the coarse grid of ```match_dense``` and the sampled mode, in pixels of the resized image. Without refinement it is the output grid, with refinement the grid the refiners start from. 8 queries every feature cell, smaller values query between cells (4 gives 2x the resolution). Must divide the resized size. Default: 8.
- ```max_num_samples```: Maximum number of matches sampled from the dense grids when neither image has keypoints, before filtering. Default: 10000.
- ```query_chunk_size```: Maximum number of points the correspondence head processes at once, to bound memory for many keypoints or a fine ```coarse_stride```. Results do not depend on it. Fixed at construction. Default: 4096.
- ```kde_balance```: Sample the matches as RoMa v2, spread over the image by a kernel density estimate, and drop isolated matches. Otherwise they are drawn proportional to their confidence. Default: True.
- ```weights```: Path or URL of the matcher weights. Default: the release weights.

</details>

The default values give a good trade-off between speed and accuracy. For increased efficiency, use fewer layers to linearly reduce run time:
```python
matcher = FlexGlue(n_layers=6)
```
`n_layers=0` is also supported and falls back to matching the descriptors, without any transformer layer.

To obtain fewer but more reliable matches, raise the confidence threshold and bound the cycle error in pixels of the resized image. These parameters can be specified at initialization or overwritten per-call:
```python
matcher = FlexGlue(filter_threshold=0.2, max_distance_px=2.0)
pred = matcher.match(inputs0, inputs1, filter_threshold=0.2, max_distance_px=2.0)
```

For keypoint-less matching, you can specify a maximum number of samples (correspondences are still filtered):
```python
pred = matcher.match_pair(image0, image1, max_num_samples=1024)
```

FlexGlue can be compiled with `torch.compile`, as LightGlue:
```python
matcher = FlexGlue().eval().cuda().compile(mode="reduce-overhead")
```
The transformer layers and the correspondence heads are compiled into CUDA graphs. Keypoints are padded to the next of `static_lengths` (512, 1024, 2048, 4096 by default), larger inputs fall back to eager mode. The first calls of each new input size are slow while compiling.

## License
The code provided in this repository is released under the [MIT license](./LICENSE). The model weights (`flexglue.pth`) contain the [DINOv3](https://github.com/facebookresearch/dinov3) ConvNeXt-B backbone (`extractor.backbone.*`), which is distributed under the [DINOv3 License](./DINOv3_LICENSE.md) and restricts its use. All other weights, the matcher and the descriptor heads, are released under the MIT license. The refiners in [refiner.py](./flexglue/refiner.py) and their weights come from [RoMa v2](https://github.com/Parskatt/RoMaV2) and follow its MIT license. The keypoint detectors are not part of this repository, see [LightGlue](https://github.com/cvg/LightGlue#license) for their licenses.
