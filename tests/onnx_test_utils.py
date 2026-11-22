"""Share one exporter module so ONNX operator registrations occur once."""
import importlib
import sys
from pathlib import Path
from unittest.mock import patch


def load_export():
    path = Path(__file__).resolve().parents[1] / "onnx" / "export.py"
    with patch.object(sys, "path", [str(path.parent), *sys.path]):
        module = importlib.import_module("export")
    if Path(module.__file__).resolve() != path:
        raise ImportError("export module came from another checkout")
    return module
