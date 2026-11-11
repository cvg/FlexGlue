import copy
import importlib.util
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from flexglue.flexglue import FlexGlue
from flexglue.refiner import VGG19BN, ConvRefiner, RoMav2Refiner


def tiny():
    net = VGG19BN()
    net.layers = nn.ModuleList(
        [
            nn.Conv2d(3, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]
    )
    return net


class SelectedFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, old_threads)

    def setUp(self):
        torch.manual_seed(123)
        self.image = torch.rand(2, 3, 8, 8)

    def test_eval_selected_features_and_hooks(self):
        net = tiny().eval()
        calls = []
        handles = [
            layer.register_forward_hook(lambda m, a, o: calls.append(m))
            for layer in net.layers
        ]
        full = net(self.image)
        self.assertEqual(set(full), {1, 2, 4})
        self.assertEqual(len(calls), 11)
        calls.clear()
        for scales in ((1,), (2,), (2, 1), (4,), (4, 1)):
            selected = net(self.image, scales=scales)
            self.assertEqual(set(selected), set(scales))
            for s in scales:
                torch.testing.assert_close(selected[s], full[s], rtol=0, atol=0)
        calls.clear()
        net(self.image, scales=(2, 1))
        self.assertEqual(calls, list(net.layers[:7]))
        for handle in handles:
            handle.remove()
        for invalid in ((), (8,), (1, 3)):
            with self.assertRaises(ValueError):
                net(self.image, scales=invalid)

    def test_training_full_execution_and_state(self):
        full, selected = tiny().train(), None
        selected = copy.deepcopy(full)
        expected = full(self.image)
        actual = selected(self.image, scales=(1, 2))
        self.assertEqual(set(actual), {1, 2})
        for s in actual:
            torch.testing.assert_close(actual[s], expected[s], rtol=0, atol=0)
        for k, v in full.state_dict().items():
            torch.testing.assert_close(v, selected.state_dict()[k], rtol=0, atol=0)
        full.load_state_dict(selected.state_dict(), strict=True)

    def test_refiner_full_and_selected_scale_contract(self):
        ref = RoMav2Refiner.__new__(RoMav2Refiner)
        nn.Module.__init__(ref)
        ref.features = tiny().eval()
        ref.scales = (2,)
        with patch("flexglue.refiner.pp.resize_images", side_effect=lambda x, _: x):
            full = ref.extract(self.image)
            selected = ref.extract_selected(self.image)
        self.assertEqual(set(full), {1, 2, 4})
        self.assertEqual(set(selected), {1, 2})
        for s in selected:
            torch.testing.assert_close(selected[s], full[s], rtol=0, atol=0)

    def test_selected_preserves_training_child_under_eval_parent(self):
        full = tiny().eval()
        full.layers[9].train()
        selected = copy.deepcopy(full)
        self.assertFalse(selected.training)
        expected = full(self.image)
        actual = selected(self.image, scales=(1, 2))
        self.assertEqual(set(actual), {1, 2})
        for stride in actual:
            torch.testing.assert_close(actual[stride], expected[stride], rtol=0, atol=0)
        self.assertEqual(int(selected.layers[9].num_batches_tracked), 1)
        for key, value in full.state_dict().items():
            torch.testing.assert_close(
                value, selected.state_dict()[key], rtol=0, atol=0
            )

    def test_real_subclass_and_instance_extract_overrides(self):
        class CustomRefiner(RoMav2Refiner):
            def extract(self, img, resize=(512, 512)):
                self.seen.append((img, resize))
                return {s: t + 7 for s, t in super().extract(img, resize).items()}

        ref = CustomRefiner.__new__(CustomRefiner)
        nn.Module.__init__(ref)
        ref.features = tiny().eval()
        ref.scales = (2, 1)
        ref.seen = []
        resize = (8, 8)
        with patch("flexglue.refiner.pp.resize_images", side_effect=lambda x, _: x):
            full = ref.extract(self.image, resize=resize)
            selected = ref.extract_selected(self.image, resize=resize)
            self.assertEqual(len(ref.seen), 2)
            self.assertIs(ref.seen[-1][0], self.image)
            self.assertEqual(ref.seen[-1][1], resize)
            self.assertEqual(set(full), {1, 2, 4})
            self.assertEqual(set(selected), {1, 2})
            for s in selected:
                torch.testing.assert_close(selected[s], full[s], rtol=0, atol=0)
            original = RoMav2Refiner.extract(ref, self.image, resize=resize)
            self.assertTrue(torch.equal(selected[1], original[1] + 7))

        plain = RoMav2Refiner.__new__(RoMav2Refiner)
        nn.Module.__init__(plain)
        plain.features = tiny().eval()
        plain.scales = (2, 1)
        calls = []

        def custom(img, resize):
            calls.append((img, resize))
            return {s: torch.tensor(float(s + 10)) for s in (1, 2, 4)}

        plain.extract = custom
        selected = plain.extract_selected(self.image, resize=resize)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], self.image)
        self.assertEqual(calls[0][1], resize)
        self.assertEqual(set(selected), {1, 2})
        self.assertEqual(selected[2].item(), 12)
        plain.extract = lambda img, resize: {1: torch.ones(1)}
        with self.assertRaises(KeyError):
            plain.extract_selected(self.image, resize=resize)

    def test_class_monkeypatched_extract_is_honored(self):
        ref = RoMav2Refiner.__new__(RoMav2Refiner)
        nn.Module.__init__(ref)
        ref.features = tiny().eval()
        ref.scales = (2, 1)
        original = RoMav2Refiner.extract
        seen = []

        def replacement(self, img, resize=(512, 512)):
            seen.append((img, resize))
            return {s: t + 9 for s, t in original(self, img, resize).items()}

        with patch("flexglue.refiner.pp.resize_images", side_effect=lambda x, _: x):
            with patch.object(RoMav2Refiner, "extract", replacement):
                full = ref.extract(self.image, resize=(8, 8))
                selected = ref.extract_selected(self.image, resize=(8, 8))
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[-1][0], self.image)
        self.assertEqual(seen[-1][1], (8, 8))
        self.assertEqual(set(full), {1, 2, 4})
        self.assertEqual(set(selected), {1, 2})
        for stride in selected:
            torch.testing.assert_close(selected[stride], full[stride], rtol=0, atol=0)

    def test_decode_default_override_opt_in_and_cache(self):
        class Extractor:
            def __init__(self):
                self.calls = []

            def extract(self, image, resize):
                self.calls.append(("custom", image, resize))
                return {"custom": image}

            def extract_selected(self, image, resize):
                self.calls.append(("selected", image, resize))
                return {"selected": image}

        context = namedtuple("Context", "fine")
        ref = Extractor()
        fake = SimpleNamespace(
            static_lengths=None,
            transformers=[],
            refiner=ref,
            encode_pair=lambda a, b: (a, b, None, None),
            pair_context=lambda *args: context(None),
        )
        f = torch.zeros(1, 1, 2, 2)
        fake.default_conf = FlexGlue.default_conf
        fake.fixed_conf = FlexGlue.fixed_conf
        fake.conf = SimpleNamespace(**FlexGlue.default_conf)
        fake.conf.n_layers = 0
        fake.conf.refine = True
        fake.conf.describe = False
        fake.conf.mp = None
        fake.conf.refine_resize = (8, 8)
        fake.loaded_heads = {0}
        fake.descriptor = None
        conf = FlexGlue.resolve_conf(fake, {})
        self.assertFalse(conf.refine_selected_features)
        cached = ({1: torch.ones(1)}, {1: torch.zeros(1)})
        for selected in (False, True):
            conf = FlexGlue.resolve_conf(fake, {"refine_selected_features": selected})
            self.assertEqual(conf.refine_selected_features, selected)
            self.assertFalse(fake.conf.refine_selected_features)
            kind = "selected" if selected else "custom"
            for use_cache in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                with self.subTest(selected=selected, use_cache=use_cache):
                    ref.calls.clear()
                    inputs, expected_calls = [], []
                    for i, image in enumerate(("a", "b")):
                        data = {"features": f}
                        if use_cache[i]:
                            data["fine_features"] = cached[i]
                        else:
                            data["image"] = image
                            expected_calls.append((kind, image, (8, 8)))
                        inputs.append(data)
                    result = FlexGlue.decode_pair(fake, *inputs, conf)
                    self.assertEqual(ref.calls, expected_calls)
                    for i in range(2):
                        if use_cache[i]:
                            self.assertIs(result.fine[i], cached[i])
                        else:
                            self.assertEqual(result.fine[i], {kind: ("a", "b")[i]})

    @unittest.skipUnless(
        torch.cuda.is_available()
        and callable(
            getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
        )
        and importlib.util.find_spec("triton") is not None,
        "CUDA compilation support is required",
    )
    def test_owned_compiled_features_survive_pair_generations(self):
        ref = RoMav2Refiner.__new__(RoMav2Refiner)
        nn.Module.__init__(ref)
        ref.features = tiny().eval().cuda()
        ref.scales = (2, 1)
        ref.features.forward = torch.compile(
            ref.features.forward, mode="reduce-overhead", fullgraph=True
        )
        context = namedtuple("Context", "fine")
        fake = SimpleNamespace(
            static_lengths=(4096,),
            transformers=[],
            refiner=ref,
            encode_pair=lambda a, b: (a, b, None, None),
            pair_context=lambda *args: context(None),
        )
        conf = SimpleNamespace(
            n_layers=0,
            mp=None,
            refine=True,
            refine_resize=(32, 32),
            refine_selected_features=False,
        )
        images = [torch.rand(1, 3, 32, 32, device="cuda") for _ in range(2)]
        feature = torch.zeros(1, 1, 4, 4)
        with torch.inference_mode():
            for selected in (False, True):
                conf.refine_selected_features = selected
                extract = ref.extract_selected if selected else ref.extract
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    extract(images[0], resize=(32, 32))
                caches = []
                for image in images:
                    raw = extract(image, resize=(32, 32))
                    owned = {s: value.clone() for s, value in raw.items()}
                    for s in owned:
                        self.assertNotEqual(owned[s].data_ptr(), raw[s].data_ptr())
                    caches.append(owned)
                saved = [
                    {s: value.clone() for s, value in cache.items()} for cache in caches
                ]
                inputs = [
                    {"features": feature, "fine_features": cache} for cache in caches
                ]
                for _ in range(2):
                    with patch.object(
                        ref, "extract", side_effect=AssertionError("cache missed")
                    ), patch.object(
                        ref,
                        "extract_selected",
                        side_effect=AssertionError("cache missed"),
                    ):
                        pair = FlexGlue.decode_pair(fake, *inputs, conf)
                    # Reuse buffers after decode_pair starts another generation.
                    extract(images[0], resize=(32, 32))
                    for i, (cache, expected) in enumerate(zip(caches, saved)):
                        self.assertIs(pair.fine[i], cache)
                        for s, value in cache.items():
                            torch.testing.assert_close(
                                value, expected[s], rtol=0, atol=0
                            )
                            self.assertTrue(torch.isfinite(value).all())


class RefinerSpatialContractTest(unittest.TestCase):
    def test_unequal_grids_rejected_before_projection(self):
        model = ConvRefiner(2, 2, 2, radius=1, hidden_blocks=0).eval()
        source = torch.ones(1, 3, 3, 2)
        for target_shape in ((1, 3, 9, 2), (1, 9, 3, 2), (1, 2, 2, 2)):
            with self.subTest(target_shape=target_shape):
                calls = []
                hook = model.proj.register_forward_pre_hook(
                    lambda module, args: calls.append(1)
                )
                try:
                    with self.assertRaisesRegex(ValueError, "matching feature sizes"):
                        model(
                            source,
                            torch.ones(target_shape),
                            torch.zeros(1, 3, 3, 2),
                            torch.zeros(1, 3, 3, 1),
                            (1.0, 1.0),
                        )
                    self.assertEqual(calls, [])
                finally:
                    hook.remove()

    def test_equal_rectangular_grids_are_supported(self):
        model = ConvRefiner(2, 2, 2, radius=1, hidden_blocks=0).eval()
        for shape in ((1, 3, 9, 2), (2, 9, 3, 2)):
            with self.subTest(shape=shape), torch.inference_mode():
                warp, logits = model(
                    torch.ones(shape),
                    torch.zeros(shape),
                    torch.zeros(shape),
                    torch.zeros(*shape[:3], 1),
                    (1.0, 1.0),
                )
                self.assertEqual(warp.shape, shape)
                self.assertEqual(logits.shape, (*shape[:3], 4))
                self.assertTrue(torch.isfinite(warp).all())
                self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
