import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "materialize_model_source.py"
REGISTRY_PATH = ROOT / "bioreason2" / "utils" / "research_registry.py"


def load_research_registry_module():
    module_name = "materialize_model_source_research_registry_test_module"
    spec = importlib.util.spec_from_file_location(module_name, REGISTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_materialize_model_source_module():
    module_name = "materialize_model_source_contracts_test_module"
    registry_module = load_research_registry_module()

    bioreason2_module = sys.modules.get("bioreason2", types.ModuleType("bioreason2"))
    utils_module = types.ModuleType("bioreason2.utils")
    utils_module.research_registry = registry_module

    previous_bioreason2 = sys.modules.get("bioreason2")
    previous_utils = sys.modules.get("bioreason2.utils")
    previous_registry = sys.modules.get("bioreason2.utils.research_registry")

    sys.modules["bioreason2"] = bioreason2_module
    sys.modules["bioreason2.utils"] = utils_module
    sys.modules["bioreason2.utils.research_registry"] = registry_module

    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_bioreason2 is None:
            sys.modules.pop("bioreason2", None)
        else:
            sys.modules["bioreason2"] = previous_bioreason2

        if previous_utils is None:
            sys.modules.pop("bioreason2.utils", None)
        else:
            sys.modules["bioreason2.utils"] = previous_utils

        if previous_registry is None:
            sys.modules.pop("bioreason2.utils.research_registry", None)
        else:
            sys.modules["bioreason2.utils.research_registry"] = previous_registry
    return module


MATERIALIZE_MODEL_SOURCE = load_materialize_model_source_module()


class MaterializeModelSourceContractsTest(unittest.TestCase):
    def test_source_local_dir_and_download_dir_are_kept_distinct(self):
        captured = {}

        def fake_materialize_first_available_source(sources):
            captured["sources"] = sources
            return {
                "local_path": "/tmp/resolved-model",
                "source_ref": "demo/project/bioreason-pro-base:production",
                "source_type": "wandb_artifact",
            }

        argv = [
            "materialize_model_source.py",
            "--wandb-registry-path",
            "demo/project/bioreason-pro-base:production",
            "--source-local-dir",
            "checkpoints/bioreason_pro_base",
            "--local-dir",
            "data/artifacts/models/bioreason_pro_base",
            "--required-path",
            "config.json",
        ]

        with mock.patch.object(
            MATERIALIZE_MODEL_SOURCE,
            "materialize_first_available_source",
            side_effect=fake_materialize_first_available_source,
        ), mock.patch.object(sys, "argv", argv):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = MATERIALIZE_MODEL_SOURCE.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "/tmp/resolved-model")
        self.assertEqual(captured["sources"][0]["type"], "wandb_artifact")
        self.assertTrue(
            captured["sources"][0]["local_dir"].endswith("data/artifacts/models/bioreason_pro_base")
        )
        self.assertEqual(captured["sources"][1]["type"], "local_dir")
        self.assertTrue(
            captured["sources"][1]["local_dir"].endswith("checkpoints/bioreason_pro_base")
        )


if __name__ == "__main__":
    unittest.main()
