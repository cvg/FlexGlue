"""Independent full-materialization oracle for delayed sampled coordinates."""
import unittest
from types import SimpleNamespace

import torch

from flexglue import FlexGlue
from flexglue.utils import ops


def reference(pred, data0, data1, conf):
    data, endpoints, scores = (data0, data1), ([], []), []
    for i, j in ((0, 1), (1, 0)):
        target = pred[f"dense_p2d{i}_i{j}"]
        source = ops.denormalize_keypoints(
            ops.grid_sources(target), data[i]["image_size"]
        )
        pair = (source.flatten(1, 2), target.flatten(1, 2))
        endpoints[0].append(pair[i])
        endpoints[1].append(pair[j])
        scores.append(pred[f"dense_confidence{i}"].flatten(1))
    k0, k1 = (torch.cat(v, 1) for v in endpoints)
    scores = torch.cat(scores, 1)
    weights = scores
    if conf.kde_balance:
        weights = ops.kde_weights(
            pred["dense_p2d0_n1"], pred["dense_p2d1_n0"], scores, conf.max_num_samples
        )
    idx, xy = ops.sample_cells(weights, conf.max_num_samples)
    return ops.identity_matches(
        k0.gather(1, xy), k1.gather(1, xy), scores.gather(1, idx), conf.filter_threshold
    )


def inputs(
    shapes, dtype=torch.float32, size_dtype=torch.float32, batch=2, strided=False
):
    pred, data = {}, []
    for i, (h, w) in enumerate(shapes):
        target = torch.randn(batch, h, w * (2 if strided else 1), 2, dtype=dtype)
        if strided:
            target = target[:, :, ::2]
        pred[f"dense_p2d{i}_i{1-i}"] = target
        pred[f"dense_p2d{i}_n{1-i}"] = target.float().clamp(-1, 1)
        pred[f"dense_confidence{i}"] = torch.rand(batch, h, w, dtype=torch.float32)
        data.append(
            {
                "image_size": torch.tensor(
                    [[1537, 1023]] * batch, dtype=size_dtype
                ).reshape(batch, 2)
            }
        )
    return pred, data


class SampleEndpointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old)

    def compare(self, pred, data, conf):
        outputs, errors, rng = [], [], []
        for function in (reference, lambda *a: FlexGlue.sample_matches(None, *a)):
            torch.manual_seed(81)
            try:
                outputs.append(function(pred, *data, conf))
                errors.append(None)
            except Exception as exc:
                outputs.append(None)
                errors.append((type(exc).__name__, str(exc)))
            rng.append(torch.get_rng_state())
        self.assertEqual(errors[0], errors[1])
        self.assertTrue(torch.equal(*rng))
        if errors[0] is None:
            self.assertEqual(outputs[0].keys(), outputs[1].keys())
            for key in outputs[0]:
                a, b = outputs[0][key], outputs[1][key]
                self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(a.shape, b.shape)
                self.assertTrue(
                    torch.equal(
                        a.contiguous().view(torch.uint8),
                        b.contiguous().view(torch.uint8),
                    ),
                    key,
                )

    def test_sampling_dtypes_layouts_empty_and_counts(self):
        old = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, old)
        for default in (torch.float32, torch.float64):
            torch.set_default_dtype(default)
            for shapes in (
                ((3, 5), (4, 7)),
                ((1, 1), (1, 1)),
                ((0, 3), (2, 5)),
                ((2, 3), (4, 0)),
            ):
                for dtype in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ):
                    for size_dtype in (torch.int64, torch.float64):
                        pred, data = inputs(shapes, dtype, size_dtype, strided=True)
                        for count in (1, 7, 1000):
                            with self.subTest(
                                default=default,
                                shapes=shapes,
                                dtype=dtype,
                                size_dtype=size_dtype,
                                count=count,
                            ):
                                self.compare(
                                    pred,
                                    data,
                                    SimpleNamespace(
                                        kde_balance=False,
                                        max_num_samples=count,
                                        filter_threshold=0.1,
                                    ),
                                )

    def test_rejected_nonfinite_and_refusals(self):
        for shapes in (((3, 5), (4, 7)), ((0, 1), (0, 1))):
            for count in (0, -1, 7):
                pred, data = inputs(shapes)
                self.compare(
                    pred,
                    data,
                    SimpleNamespace(
                        kde_balance=False, max_num_samples=count, filter_threshold=0.1
                    ),
                )
        for bad in (float("nan"), float("inf"), -float("inf"), -0.0):
            pred, data = inputs(((3, 5), (4, 7)))
            for key in ("dense_p2d0_i1", "dense_p2d1_i0"):
                pred[key][..., 0] = bad
            pred["dense_confidence0"].zero_()
            pred["dense_confidence1"].zero_()
            self.compare(
                pred,
                data,
                SimpleNamespace(
                    kde_balance=False, max_num_samples=13, filter_threshold=0.1
                ),
            )

        for bad in (float("nan"), float("inf"), -float("inf"), -1.0):
            pred, data = inputs(((3, 5), (4, 7)))
            pred["dense_confidence0"][:, 0, 0] = bad
            self.compare(
                pred,
                data,
                SimpleNamespace(
                    kde_balance=False, max_num_samples=7, filter_threshold=0.1
                ),
            )

    def test_kde_preserves_seeded_outputs_and_rng(self):
        conf = SimpleNamespace(
            kde_balance=True, max_num_samples=13, filter_threshold=0.1
        )
        for seed in (19, 43):
            torch.manual_seed(seed)
            pred, data = inputs(((3, 5), (4, 7)), strided=True)
            expected = reference(pred, *data, conf)
            self.assertTrue((expected["matches0"] >= 0).any())
            self.compare(pred, data, conf)

    def test_mixed_direction_dtypes_and_empty_batch(self):
        for shapes in (((3, 5), (4, 7)), ((0, 3), (2, 5)), ((2, 3), (4, 0))):
            for first in (torch.float16, torch.float64):
                pred, data = inputs(shapes)
                pred["dense_p2d0_i1"] = pred["dense_p2d0_i1"].to(first)
                data[0]["image_size"] = data[0]["image_size"].to(torch.int64)
                data[1]["image_size"] = data[1]["image_size"].to(torch.float64)
                self.compare(
                    pred,
                    data,
                    SimpleNamespace(
                        kde_balance=False, max_num_samples=7, filter_threshold=0.1
                    ),
                )
        pred, data = inputs(((3, 5), (4, 7)), batch=0)
        self.compare(
            pred,
            data,
            SimpleNamespace(kde_balance=False, max_num_samples=7, filter_threshold=0.1),
        )

    def test_malformed_endpoint_metadata_refuses_before_draw(self):
        for kde in (False, True):
            for missing in (False, True):
                pred, data = inputs(((3, 5), (4, 7)))
                if missing:
                    del data[0]["image_size"]
                else:
                    data[0]["image_size"] = torch.ones(2, 3)
                self.compare(
                    pred,
                    data,
                    SimpleNamespace(
                        kde_balance=kde, max_num_samples=7, filter_threshold=0.1
                    ),
                )

    def test_gradients(self):
        pred, data = inputs(((3, 5), (4, 7)), torch.float64, torch.float64)
        conf = SimpleNamespace(
            kde_balance=False, max_num_samples=13, filter_threshold=-1
        )
        gradients = []
        for function in (reference, lambda *a: FlexGlue.sample_matches(None, *a)):
            p = {k: v.detach().clone().requires_grad_(True) for k, v in pred.items()}
            d = [
                {"image_size": x["image_size"].clone().requires_grad_(True)}
                for x in data
            ]
            torch.manual_seed(81)
            out = function(p, *d, conf)
            loss = sum(
                out[k].square().sum()
                for k in ("keypoints0", "keypoints1", "matching_scores0")
            )
            variables = list(p.values()) + [x["image_size"] for x in d]
            gradients.append(torch.autograd.grad(loss, variables, allow_unused=True))
        for a, b in zip(*gradients):
            if a is None:
                self.assertIsNone(b)
            else:
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-10)

    def test_endpoint_score_shape_mismatches_keep_full_pool_behavior(self):
        conf = SimpleNamespace(
            kde_balance=False, max_num_samples=7, filter_threshold=0.1
        )
        pred, data = inputs(((3, 5), (4, 7)))
        pred["dense_confidence0"] = torch.rand(2, 1, 58)
        self.compare(pred, data, conf)
        pred, data = inputs(((3, 5), (4, 7)))
        pred["dense_confidence0"] = torch.rand(2, 1, 13)
        pred["dense_confidence1"] = torch.rand(2, 1, 30)
        self.compare(pred, data, conf)
        pred, data = inputs(((3, 5), (4, 7)))
        for i in (0, 1):
            pred[f"dense_confidence{i}"] = pred[f"dense_confidence{i}"][:1]
        self.compare(pred, data, conf)


if __name__ == "__main__":
    unittest.main()
