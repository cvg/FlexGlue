import importlib.util
import math
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from onnx_test_utils import load_export


class EarlyStopExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        cls.export = load_export()

    def test_early_stop_export_checks_only_required_heads_before_copy(self):
        from flexglue import FlexGlue

        export = self.export
        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=5,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
            early_stop_decay=0.5,
            min_covisibility=0.5,
        ).eval()
        original_conf = vars(model.conf).copy()
        model.loaded_heads = {2, 4, 5}  # unchecked intermediate heads can be absent
        original_heads = model.loaded_heads.copy()
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "matcher.onnx"

        with patch.object(
            export, "dummy_matcher_inputs", return_value=()
        ), patch.object(
            torch.onnx,
            "export",
            side_effect=lambda *args, **kwargs: Path(args[2]).write_bytes(b"graph"),
        ) as write, patch.object(
            export, "finalize_matcher_graph"
        ), patch.object(
            export, "add_early_stopping"
        ):
            export.export_matcher(model, path, (32, 32), 18, early_stop_layers=(2, 4))
            self.assertIsInstance(
                write.call_args.args[0], export.LayerScoresMatcherONNX
            )
            self.assertIsNot(write.call_args.args[0].matcher, model)
            self.assertEqual(write.call_args.args[0].check_layers, (2, 4))
            self.assertIsNone(write.call_args.args[0].conf.early_stop_decay)
        self.assertEqual(path.read_bytes(), b"graph")
        self.assertEqual(list(path.parent.iterdir()), [path])
        self.assertEqual(vars(model.conf), original_conf)
        self.assertEqual(model.loaded_heads, original_heads)

        for heads, schedule, debug in (
            ({4, 5}, (2, 4), False),  # missing a checked head
            ({2, 4}, (2, 4), False),  # missing the final head
            ({2, 4, 5}, (2, 2), False),
            ({2, 4, 5}, (True,), False),
            ({2, 4, 5}, (5,), False),
            ({2, 4, 5}, (0,), False),
            ({2, 4, 5}, (2, 4), True),
        ):
            with self.subTest(heads=heads, schedule=schedule, debug=debug):
                model.loaded_heads = heads
                with patch.object(
                    export.copy, "deepcopy", wraps=export.copy.deepcopy
                ) as copy_model:
                    with self.assertRaises(ValueError):
                        export.export_matcher(
                            model,
                            path,
                            (32, 32),
                            18,
                            early_stop_layers=schedule,
                            debug_outputs=debug,
                        )
                    copy_model.assert_not_called()
                self.assertEqual(vars(model.conf), original_conf)
                self.assertEqual(model.loaded_heads, heads)

    def test_failed_export_preserves_destination_and_removes_staging(self):
        from flexglue import FlexGlue

        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=5,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
        ).eval()
        export = self.export
        for existing in (False, True):
            for stage in ("write", "dynamic", "early_stop", "replace"):
                with self.subTest(existing=existing, stage=stage):
                    with tempfile.TemporaryDirectory() as temporary:
                        destination = Path(temporary) / "matcher.onnx"
                        if existing:
                            destination.write_bytes(b"original")

                        def write(*args, **kwargs):
                            Path(args[2]).write_bytes(b"partial")
                            if stage == "write":
                                raise RuntimeError("export failed")

                        def rewrite(*args):
                            if stage == "dynamic":
                                raise RuntimeError("rewrite failed")

                        def branch(*args):
                            if stage == "early_stop":
                                raise RuntimeError("branch failed")

                        original_replace = Path.replace

                        def replace(staged, target):
                            if stage == "replace":
                                raise RuntimeError("replace failed")
                            return original_replace(staged, target)

                        with patch.object(
                            export, "dummy_matcher_inputs", return_value=()
                        ), patch.object(
                            torch.onnx, "export", side_effect=write
                        ), patch.object(
                            export, "finalize_matcher_graph", side_effect=rewrite
                        ), patch.object(
                            export, "add_early_stopping", side_effect=branch
                        ), patch.object(
                            Path, "replace", new=replace
                        ):
                            with self.assertRaises(RuntimeError):
                                export.export_matcher(
                                    model,
                                    destination,
                                    (32, 32),
                                    18,
                                    early_stop_layers=(2, 4),
                                )
                        if existing:
                            self.assertEqual(destination.read_bytes(), b"original")
                        else:
                            self.assertFalse(destination.exists())
                        self.assertEqual(
                            list(Path(temporary).iterdir()),
                            [destination] if existing else [],
                        )

    def test_successful_export_replaces_complete_destination(self):
        from flexglue import FlexGlue

        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=5,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
        ).eval()
        export = self.export
        for existing in (False, True):
            with self.subTest(existing=existing):
                with tempfile.TemporaryDirectory() as temporary:
                    destination = Path(temporary) / "matcher.onnx"
                    if existing:
                        destination.write_bytes(b"original")

                    def write(*args, **kwargs):
                        Path(args[2]).write_bytes(b"graph")
                        if existing:
                            self.assertEqual(destination.read_bytes(), b"original")
                        else:
                            self.assertFalse(destination.exists())

                    def rewrite(staged, *_):
                        staged.write_bytes(staged.read_bytes() + b"-rewritten")

                    with patch.object(
                        export, "dummy_matcher_inputs", return_value=()
                    ), patch.object(
                        torch.onnx, "export", side_effect=write
                    ), patch.object(
                        export, "finalize_matcher_graph", side_effect=rewrite
                    ):
                        export.export_matcher(model, destination, (32, 32), 18)
                    self.assertEqual(destination.read_bytes(), b"graph-rewritten")
                    self.assertEqual(list(Path(temporary).iterdir()), [destination])

    def test_export_preserves_permissions_and_follows_destination_link(self):
        from flexglue import FlexGlue

        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=5,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
        ).eval()
        export = self.export
        for linked in (False, True):
            with self.subTest(linked=linked):
                with tempfile.TemporaryDirectory() as temporary:
                    target = Path(temporary) / "target.onnx"
                    target.write_bytes(b"original")
                    target.chmod(0o600)
                    mode = stat.S_IMODE(target.stat().st_mode)
                    destination = target
                    if linked:
                        destination = target.with_name("link.onnx")
                        try:
                            destination.symlink_to(target.name)
                        except OSError as error:
                            if getattr(error, "winerror", None) == 1314:
                                self.skipTest("Windows symlink privilege unavailable")
                            raise

                    def write(*args, **kwargs):
                        Path(args[2]).write_bytes(b"complete")

                    with patch.object(
                        export, "dummy_matcher_inputs", return_value=()
                    ), patch.object(
                        torch.onnx, "export", side_effect=write
                    ), patch.object(
                        export, "finalize_matcher_graph"
                    ):
                        export.export_matcher(model, destination, (32, 32), 18)
                    self.assertEqual(target.read_bytes(), b"complete")
                    self.assertEqual(stat.S_IMODE(target.stat().st_mode), mode)
                    self.assertEqual(destination.is_symlink(), linked)
                    self.assertEqual(
                        set(Path(temporary).iterdir()), {target, destination}
                    )

    def test_layer_scores_wrapper_requires_selected_heads(self):
        from flexglue import FlexGlue

        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=5,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
        ).eval()
        model.loaded_heads = {2, 4, 5}
        wrapper = self.export.LayerScoresMatcherONNX(model, (2, 4))
        self.assertEqual(wrapper.check_layers, (2, 4))
        model.loaded_heads = {4, 5}
        with self.assertRaises(ValueError):
            self.export.LayerScoresMatcherONNX(model, (2, 4))

    def test_graph_metadata_controls_checker_depth_and_stride(self):
        import onnx
        from onnx import TensorProto, helper

        spec = importlib.util.spec_from_file_location(
            "flexglue_metadata_check_test",
            Path(__file__).resolve().parents[1] / "onnx" / "check.py",
        )
        check = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"export": self.export}):
            with patch.object(
                sys, "path", [str(Path(check.__file__).parent), *sys.path]
            ):
                spec.loader.exec_module(check)
        graph = helper.make_graph(
            [helper.make_node("Identity", ["features0"], ["output"])],
            "metadata_fixture",
            [
                helper.make_tensor_value_info(
                    "features0", TensorProto.FLOAT, [1, 256, 4, 6]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [1, 256, 4, 6]
                )
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.onnx"
            onnx.save(model, path)
            self.export.finalize_matcher_graph(path, 5, 8)
            options = check.matcher_options(path)
            self.assertEqual(options.n_layers, 5)
            self.assertEqual(options.size, (32, 48))
            self.assertEqual(options.early_stop_layers, [])
            model = onnx.load(path)
            self.assertEqual(
                {e.key: e.value for e in model.metadata_props},
                {"flexglue.n_layers": "5", "flexglue.stride": "8"},
            )
            for metadata in (
                {"flexglue.n_layers": "-1"},
                {"flexglue.n_layers": "5.0"},
                {"flexglue.stride": "4"},
            ):
                helper.set_model_props(model, metadata)
                onnx.save(model, path)
                with self.assertRaises(ValueError):
                    check.matcher_options(path)
            helper.set_model_props(model, {})
            onnx.save(model, path)
            self.assertEqual(check.matcher_options(path).n_layers, 9)
            model.graph.input.append(
                helper.make_tensor_value_info("early_stop_decay", TensorProto.FLOAT, [])
            )
            helper.set_model_props(model, {"flexglue.n_layers": "5"})
            onnx.save(model, path)
            with patch.object(
                check, "read_early_stop_layers", return_value=[2, 4]
            ) as reader:
                self.assertEqual(check.matcher_options(path).early_stop_layers, [2, 4])
                self.assertEqual(reader.call_args.args[1], 5)

    def test_early_stop_checker_rejects_wrong_score(self):
        if importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("optional dependency onnxruntime is unavailable")
        path = Path(__file__).resolve().parents[1] / "onnx" / "check.py"
        spec = importlib.util.spec_from_file_location("flexglue_check_test", path)
        check = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"export": self.export}):
            with patch.object(
                sys, "path", [str(Path(check.__file__).parent), *sys.path]
            ):
                spec.loader.exec_module(check)

        class Matcher:
            def match(self, data0, data1, **kwargs):
                self.kwargs = kwargs
                return {
                    "matches0": torch.tensor([[-1, -1]]),
                    "matching_scores0": torch.zeros(1, 2),
                    "covisibility_score": torch.tensor([0.12]),
                }

        class Session:
            def __init__(self, score):
                self.score = score

            def run(self, names, feeds):
                self.decay = float(feeds["early_stop_decay"])
                return [
                    torch.tensor([[-1, -1]]).numpy(),
                    torch.zeros(1, 2).numpy(),
                    torch.tensor([self.score]).numpy(),
                ]

        features = [
            {
                "features": torch.zeros(1, 256, 4, 4),
                "image_size": torch.tensor([[32.0, 32.0]]),
            }
            for _ in range(2)
        ]
        keypoints = [torch.ones(1, 2, 2), torch.ones(1, 3, 2)]
        feeds = check.matcher_feeds(*features, *keypoints, 0.5)
        matcher = Matcher()
        for score, expected in ((0.12, True), (0.8, False)):
            with self.subTest(score=score):
                session = Session(score)
                result = check.check_early_stop(
                    session,
                    matcher,
                    features,
                    keypoints,
                    feeds,
                    0.3,
                    check.TOLERANCES["cpu"],
                )
                self.assertIs(result, expected)
                self.assertEqual(matcher.kwargs["early_stop_decay"], 0.3)
                self.assertAlmostEqual(session.decay, 0.3)

    def test_scheduled_early_stops_match_native_runtime(self):
        self.check_scheduled_early_stops(
            9,
            (2, 4, 6, 8),
            {2: 0.12, 4: 0.13, 6: 0.14, 8: 0.15, 9: 0.9},
            ((0.1, 2), (0.2, 4), (0.3, 6), (0.5, 8), (1.0, 9)),
        )

    def test_five_layer_early_stops_match_native_runtime(self):
        self.check_scheduled_early_stops(
            5,
            (2, 4),
            {2: 0.12, 4: 0.15, 5: 0.9},
            ((0.1, 2), (0.5, 4), (1.0, 5)),
        )

    def test_default_schedule_matches_native_runtime(self):
        self.check_scheduled_early_stops(
            3,
            (2,),
            {1: 0.001, 2: 0.12, 3: 0.9},
            ((0.1, 2), (1.0, 3)),
            native_default=True,
        )

    def test_explicit_first_layer_matches_native_runtime(self):
        self.check_scheduled_early_stops(
            3,
            (1,),
            {1: 0.001, 3: 0.9},
            ((0.1, 1), (1.0, 3)),
        )

    def check_scheduled_early_stops(
        self, n_layers, schedule, scores, decays, native_default=False
    ):
        if importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("optional dependency onnxruntime is unavailable")
        import onnx
        import onnxruntime as ort

        from flexglue import FlexGlue

        export = self.export
        torch.manual_seed(31)
        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=n_layers,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
            query_chunk_size=None,
        ).eval()
        with torch.no_grad():
            for head in model.heads:
                for parameter in head.offset_head.parameters():
                    parameter.zero_()
                if head.matchability is not None:
                    head.matchability.weight.zero_()
                    head.matchability.bias.fill_(2.0)
            for layer, score in scores.items():
                model.heads[layer].covisibility.weight.zero_()
                model.heads[layer].covisibility.bias.fill_(
                    math.log(score / (1 - score))
                )

        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            path = Path(tmp) / "scheduled.onnx"
            export.export_matcher(
                model, path, (32, 32), 18, fused=False, early_stop_layers=schedule
            )
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            check_path = Path(__file__).resolve().parents[1] / "onnx" / "check.py"
            spec = importlib.util.spec_from_file_location(
                "flexglue_check_test", check_path
            )
            check = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {"export": export}):
                with patch.object(
                    sys, "path", [str(Path(check.__file__).parent), *sys.path]
                ):
                    spec.loader.exec_module(check)
            graph_model = onnx.load(str(path))
            self.assertEqual(check.matcher_options(path).n_layers, n_layers)
            self.assertTrue(
                all(
                    not tensor.external_data for tensor in graph_model.graph.initializer
                )
            )
            graphs = list(check.subgraphs(graph_model.graph))
            if_nodes = [
                node for graph in graphs for node in graph.node if node.op_type == "If"
            ]
            self.assertEqual(len(if_nodes), len(schedule))
            for index, node in enumerate(if_nodes):
                node.name = f"renamed_if_{index}"
            onnx.checker.check_model(graph_model)
            renamed_path = Path(tmp) / "renamed.onnx"
            onnx.save(graph_model, str(renamed_path))
            self.assertEqual(
                check.matcher_options(renamed_path).early_stop_layers, list(schedule)
            )

            # Rename the score consistently throughout all graph scopes, so the
            # model remains valid but this stop predicate cannot be identified.
            old, new = f"covisibility_score_layer{schedule[-1]}", "anonymous_stop_score"
            replacements = 0
            for graph in graphs:
                for node in graph.node:
                    for names in (node.input, node.output):
                        for index, name in enumerate(names):
                            if name == old:
                                names[index] = new
                                replacements += 1
                for value in (
                    list(graph.input) + list(graph.output) + list(graph.value_info)
                ):
                    if value.name == old:
                        value.name = new
                        replacements += 1
            self.assertGreater(replacements, 0)
            onnx.checker.check_model(graph_model)
            unknown_path = Path(tmp) / "unknown.onnx"
            onnx.save(graph_model, str(unknown_path))
            with self.assertRaisesRegex(ValueError, "Cannot identify"):
                check.matcher_options(unknown_path)
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            for n0, n1 in ((1, 1), (3, 5), (512, 512)):
                inputs = list(export.dummy_matcher_inputs((32, 32), n0, n1))
                inputs[6], inputs[7], inputs[8] = (
                    torch.tensor(0.0),
                    torch.tensor(float("inf")),
                    torch.tensor(0.5),
                )
                data = [
                    {
                        "features": inputs[k],
                        "keypoints": inputs[2 + k],
                        "image_size": inputs[4 + k],
                    }
                    for k in (0, 1)
                ]
                for decay, layer in decays:
                    with self.subTest(decay=decay, layer=layer):
                        pred = model.match(
                            *data,
                            sparse=True,
                            exact_nn=True,
                            filter_threshold=0.0,
                            max_distance_px=float("inf"),
                            min_covisibility=0.5,
                            early_stop_layers=None if native_default else schedule,
                            early_stop_decay=None if decay == 1 else decay,
                        )
                        feed = {
                            key: value.numpy()
                            for key, value in zip(export.MATCHER_INPUTS, inputs)
                        }
                        feed["early_stop_decay"] = torch.tensor(decay).numpy()
                        matches, match_scores, covisibility = session.run(None, feed)
                        self.assertEqual(pred["stop_layer"], layer)
                        self.assertTrue(
                            torch.equal(torch.from_numpy(matches), pred["matches0"])
                        )
                        torch.testing.assert_close(
                            torch.from_numpy(match_scores),
                            pred["matching_scores0"],
                            rtol=1e-4,
                            atol=1e-5,
                        )
                        torch.testing.assert_close(
                            torch.from_numpy(covisibility),
                            pred["covisibility_score"],
                            rtol=1e-4,
                            atol=1e-5,
                        )
                        torch.testing.assert_close(
                            torch.from_numpy(covisibility),
                            torch.tensor([scores[layer]]),
                            rtol=1e-4,
                            atol=1e-5,
                        )
                        if layer == n_layers:
                            self.assertGreater(int((pred["matches0"] >= 0).sum()), 0)
                        else:
                            self.assertTrue(torch.all(torch.from_numpy(matches) == -1))
                            self.assertTrue(
                                torch.all(torch.from_numpy(match_scores) == 0)
                            )
