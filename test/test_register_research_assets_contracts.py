import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "register_research_assets.py"
REGISTRY_PATH = ROOT / "bioreason2" / "utils" / "research_registry.py"


def load_research_registry_module():
    module_name = "register_assets_research_registry_test_module"
    spec = importlib.util.spec_from_file_location(module_name, REGISTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_register_assets_module():
    module_name = "register_research_assets_contracts_test_module"
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


REGISTER_ASSETS = load_register_assets_module()


class RegisterResearchAssetsContractsTest(unittest.TestCase):
    def test_render_asset_definition_formats_aliases_and_paths(self):
        registry = {
            "assets": {
                "bioreason-pro-sft": {
                    "artifact_aliases": ["production", "{benchmark_alias}"],
                    "sources": [
                        {"type": "local_dir", "local_dir": "models/{benchmark_alias_dir}"},
                    ],
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            rendered = REGISTER_ASSETS.render_asset_definition(
                "bioreason-pro-sft",
                registry,
                repo_root,
                "213.221.225.228",
            )

        self.assertEqual(rendered["artifact_aliases"], ["production", "213.221.225.228"])
        self.assertTrue(
            rendered["sources"][0]["local_dir"].endswith("models/213_221_225_228")
        )

    def test_publish_asset_updates_registry_env_file(self):
        asset = {
            "asset_name": "bioreason-pro-sft",
            "artifact_name": "bioreason-pro-sft",
            "artifact_type": "model",
            "registry_env_var": "BIOREASON_SFT_MODEL_REGISTRY_PATH",
            "artifact_aliases": ["production", "213.221.225.228"],
            "sources": [{"type": "huggingface", "repo_id": "wanglab/bioreason-pro-sft"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "wandb_registry_paths.env"
            local_model_dir = Path(tmpdir) / "bioreason_pro_sft"
            local_model_dir.mkdir(parents=True, exist_ok=True)
            (local_model_dir / "config.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(
                REGISTER_ASSETS,
                "materialize_first_available_source",
                return_value={
                    "local_path": str(local_model_dir),
                    "source_ref": "hf://wanglab/bioreason-pro-sft",
                    "source_type": "huggingface",
                },
            ), mock.patch.object(
                REGISTER_ASSETS,
                "upload_local_asset",
                return_value={
                    "artifact_name": "bioreason-pro-sft",
                    "artifact_type": "model",
                    "aliases": ["production", "213.221.225.228"],
                    "local_path": str(local_model_dir),
                },
            ):
                status = REGISTER_ASSETS.publish_asset(
                    asset=asset,
                    entity="demo-entity",
                    project="demo-project",
                    registry_env_path=env_path,
                    benchmark_alias="213.221.225.228",
                )

            env_text = env_path.read_text(encoding="utf-8")

        self.assertEqual(
            status["registry_ref"],
            "demo-entity/demo-project/bioreason-pro-sft:production",
        )
        self.assertIn(
            'export BIOREASON_SFT_MODEL_REGISTRY_PATH="demo-entity/demo-project/bioreason-pro-sft:production"',
            env_text,
        )


if __name__ == "__main__":
    unittest.main()
