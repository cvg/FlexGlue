"""CPU scheduling/reader tests without model, checkpoint or Triton imports."""
import ast
import gc
import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
import weakref
from contextlib import contextmanager
from pathlib import Path

UTILS = Path(__file__).resolve().parents[1] / "flexglue/utils"
NAME = "_flex_image_io_tests"
package = types.ModuleType(NAME)
package.__path__ = [str(UTILS)]
sys.modules[NAME] = package
tree = ast.parse((UTILS / "preprocess.py").read_text())
wrapper = next(
    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "read_images"
)
scope = {
    "__name__": NAME + ".preprocess",
    "__package__": NAME,
    "contextmanager": contextmanager,
    "read_image": lambda p, grayscale=False: (p, grayscale),
}
exec(
    compile(
        ast.Module(body=[wrapper], type_ignores=[]), "<public read_images>", "exec"
    ),
    scope,
)
spec = importlib.util.spec_from_file_location(
    NAME + "._ordered_decode", UTILS / "_ordered_decode.py"
)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)


class Scheduling(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(
            any(t.name.startswith("flex-ordered-decode") for t in threading.enumerate())
        )

    def test_order_and_two_worker_progress(self):
        barrier = threading.Barrier(2)
        owner = threading.get_ident()

        def load(i):
            self.assertNotEqual(threading.get_ident(), owner)
            if i < 2:
                barrier.wait(timeout=5)
            return i * 7

        stats = {}
        with helper.ordered_decode(range(7), load, enabled=True, stats=stats) as rows:
            self.assertEqual(list(rows), [(i, i * 7) for i in range(7)])
        self.assertEqual(stats["maxPending"], 2)
        self.assertTrue(stats["workersJoined"])

    def test_consumed_payload_is_not_retained_until_next_read(self):
        class Payload:
            def __init__(self):
                self.pixels = bytearray(16 * 1024 * 1024)

        for enabled in (False, True):
            with self.subTest(prefetch=enabled):
                released = threading.Event()
                references = []

                def load(row):
                    payload = Payload()
                    references.append(weakref.ref(payload, lambda ref: released.set()))
                    return payload

                with helper.ordered_decode([0], load, enabled=enabled) as rows:
                    row, image = next(rows)
                    self.assertEqual(row, 0)
                    self.assertEqual(len(image.pixels), 16 * 1024 * 1024)
                    del image
                    gc.collect()
                    # A completed worker may still be returning from its task.
                    self.assertTrue(released.wait(timeout=2))
                    self.assertIsNone(references[0]())

    def test_partial_consumption(self):
        stats = {}
        with helper.ordered_decode(
            range(30), lambda x: x, enabled=True, stats=stats
        ) as rows:
            self.assertEqual(next(rows), (0, 0))
        self.assertEqual(stats["submitted"], 3)
        self.assertTrue(stats["workersJoined"])

    def test_ordered_failure(self):
        def load(i):
            if i == 2:
                raise ValueError("two")
            return i

        seen, stats = [], {}
        with self.assertRaisesRegex(ValueError, "two"):
            with helper.ordered_decode(
                range(9), load, enabled=True, stats=stats
            ) as rows:
                seen.extend(rows)
        self.assertEqual(seen, [(0, 0), (1, 1)])
        self.assertTrue(stats["workersJoined"])

    def test_empty(self):
        with scope["read_images"]([], prefetch=True) as rows:
            self.assertEqual(list(rows), [])

    def test_source_errors_follow_successful_prefix(self):
        for failure in (0, 1, 2, 5):
            for enabled in (False, True):
                with self.subTest(failure=failure, enabled=enabled):
                    calls = []

                    class Source:
                        def __iter__(self):
                            return self

                        def __next__(self):
                            i = len(calls)
                            calls.append(i)
                            if i == failure:
                                raise LookupError("source")
                            if i == 7:
                                raise StopIteration
                            return i

                    seen, stats = [], {}
                    with self.assertRaisesRegex(LookupError, "source"):
                        with helper.ordered_decode(
                            Source(), lambda x: x * 7, enabled=enabled, stats=stats
                        ) as rows:
                            seen.extend(rows)
                    self.assertEqual(seen, [(i, i * 7) for i in range(failure)])
                    self.assertEqual(calls, list(range(failure + 1)))
                    self.assertTrue(stats["workersJoined"])
                    if enabled:
                        self.assertLessEqual(stats["maxPending"], 2)

    def test_loader_error_precedes_later_source_error(self):
        def source():
            yield 0
            raise LookupError("source")

        def load(row):
            raise ValueError("loader")

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                with self.assertRaisesRegex(ValueError, "loader"):
                    with helper.ordered_decode(source(), load, enabled=enabled) as rows:
                        list(rows)

    def test_public_default_serial(self):
        owner = threading.get_ident()
        original = scope["read_image"]

        def read(path, grayscale=False):
            self.assertEqual(threading.get_ident(), owner)
            return (path, grayscale)

        scope["read_image"] = read
        try:
            with scope["read_images"](["a", "b"], grayscale=True) as rows:
                self.assertEqual(list(rows), [("a", ("a", True)), ("b", ("b", True))])
        finally:
            scope["read_image"] = original

    def test_public_consumer_exception(self):
        with self.assertRaisesRegex(RuntimeError, "consumer"):
            with scope["read_images"](["a", "b", "c"], prefetch=True) as rows:
                next(rows)
                raise RuntimeError("consumer")

    def test_actual_demo_loading_cells_keep_conversion_on_main(self):
        notebook = json.loads(
            (UTILS.parents[1] / "demo.ipynb").read_text(encoding="utf-8")
        )
        owner = threading.get_ident()
        calls = []

        class Tensor:
            def to(self, device):
                self_outer.assertEqual(threading.get_ident(), owner)
                calls.append(("to", device))
                return self

        self_outer = self

        def convert(rgb):
            self.assertEqual(threading.get_ident(), owner)
            calls.append(("convert", rgb[0].name))
            return Tensor()

        for index, count in ((5, 1), (17, 2)):
            nodes = ast.parse("".join(notebook["cells"][index]["source"])).body[:count]
            namespace = {
                "read_images": scope["read_images"],
                "images": Path("assets"),
                "numpy_image_to_torch": convert,
                "device": "cpu",
            }
            exec(
                compile(
                    ast.Module(body=nodes, type_ignores=[]), "<demo loading>", "exec"
                ),
                namespace,
            )
        self.assertEqual(
            calls,
            [
                ("convert", "berlin1.jpg"),
                ("to", "cpu"),
                ("convert", "berlin2.jpg"),
                ("to", "cpu"),
                ("convert", "lamar_cab_1446.jpg"),
                ("to", "cpu"),
                ("convert", "lamar_cab_1450.jpg"),
                ("to", "cpu"),
            ],
        )


class Reader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest("OpenCV/NumPy unavailable: " + str(error))
        cls.cv2, cls.np = cv2, np
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "read_image"
        )
        namespace = {"Path": Path, "cv2": cv2, "np": np}
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), "<public read_image>", "exec"
            ),
            namespace,
        )
        cls.read = staticmethod(namespace["read_image"])

    def test_pixels_layout_and_reader_errors(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / (str(i) + ".png") for i in range(4)]
            for i, p in enumerate(paths):
                bgr = (self.np.arange(7 * 11 * 3).reshape(7, 11, 3) + i).astype(
                    self.np.uint8
                )
                self.assertTrue(self.cv2.imwrite(str(p), bgr))
            original = scope["read_image"]
            scope["read_image"] = self.read
            try:
                for grayscale in (False, True):
                    expected = [self.read(p, grayscale=grayscale) for p in paths]
                    with scope["read_images"](
                        paths, grayscale=grayscale, prefetch=True
                    ) as rows:
                        for i, (p, actual) in enumerate(rows):
                            ref = expected[i]
                            self.assertEqual(p, paths[i])
                            self.assertEqual(
                                (actual.dtype, actual.shape, actual.strides),
                                (ref.dtype, ref.shape, ref.strides),
                            )
                            self.assertEqual(actual.tobytes(), ref.tobytes())
                corrupt = Path(root) / "broken.png"
                corrupt.write_bytes(b"broken")
                for path, error in (
                    (Path(root) / "absent.png", FileNotFoundError),
                    (corrupt, IOError),
                ):
                    with self.assertRaises(error):
                        with scope["read_images"]([path], prefetch=True) as rows:
                            list(rows)
            finally:
                scope["read_image"] = original


class PublicReaderIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("torch", "numpy", "cv2"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(name + " unavailable")
        import cv2
        import numpy as np

        from flexglue.utils import preprocess, read_images

        cls.cv2, cls.np = cv2, np
        cls.read_images = staticmethod(read_images)
        cls.preprocess = preprocess

    def test_exported_reader_matches_serial_for_both_colour_modes(self):
        self.assertIs(self.read_images, self.preprocess.read_images)
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / (str(i) + ".png") for i in range(3)]
            for i, path in enumerate(paths):
                pixels = (self.np.arange(7 * 11 * 3).reshape(7, 11, 3) + i).astype(
                    self.np.uint8
                )
                self.assertTrue(self.cv2.imwrite(str(path), pixels))
            for grayscale in (False, True):
                expected = [self.preprocess.read_image(p, grayscale) for p in paths]
                with self.read_images(
                    iter(paths), grayscale=grayscale, prefetch=True
                ) as rows:
                    for index, (path, actual) in enumerate(rows):
                        self.assertEqual(path, paths[index])
                        reference = expected[index]
                        self.assertEqual(actual.shape, reference.shape)
                        self.assertEqual(actual.strides, reference.strides)
                        self.assertEqual(actual.dtype, reference.dtype)
                        self.assertEqual(actual.tobytes(), reference.tobytes())


if __name__ == "__main__":
    unittest.main()
