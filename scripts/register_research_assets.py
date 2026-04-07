#!/usr/bin/env python3
"""
Download research assets from their configured sources and publish them to W&B artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreason2.utils.research_registry import (
    RegistryError,
    apply_template_context,
    expand_placeholders,
    load_exported_env_file,
    load_json,
    materialize_first_available_source,
    normalize_text,
    resolve_repo_path,
)

DEFAULT_MANIFEST_PATH = "configs/disease_benchmark/artifact_publish_registry.json"
DEFAULT_REGISTRY_ENV_PATH = "configs/disease_benchmark/wandb_registry_paths.env"
DEFAULT_SOURCE_ENV_PATH = "configs/disease_benchmark/wandb_asset_sources.env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=str, default=None, help="Single asset name to publish.")
    parser.add_argument("--asset-group", type=str, default=None, help="Named asset group to publish.")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the asset publish manifest JSON.",
    )
    parser.add_argument(
        "--benchmark-alias",
        type=str,
        default="213.221.225.228",
        help="Benchmark alias used for templated aliases and baseline artifact refs.",
    )
    parser.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY") or None)
    parser.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT") or None)
    parser.add_argument(
        "--registry-env-file",
        type=str,
        default=DEFAULT_REGISTRY_ENV_PATH,
        help="Local env file to update with the published W&B Registry refs.",
    )
    parser.add_argument(
        "--source-env-file",
        type=str,
        default=DEFAULT_SOURCE_ENV_PATH,
        help="Optional env file with local source directories or HF repo IDs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue publishing other assets when one asset fails.",
    )
    return parser.parse_args()


def load_publish_registry(path_value: str, repo_root: Path) -> Dict[str, Any]:
    registry_path = resolve_repo_path(path_value, repo_root)
    payload = expand_placeholders(load_json(registry_path))
    payload["_registry_path"] = registry_path
    return payload


def resolve_asset_names(registry: Mapping[str, Any], asset_name: Optional[str], asset_group: Optional[str]) -> List[str]:
    if bool(asset_name) == bool(asset_group):
        default_group = normalize_text(registry.get("default_asset_group")).strip()
        if asset_name is None and asset_group is None and default_group:
            asset_group = default_group
        else:
            raise RegistryError("Specify exactly one of --asset or --asset-group.")

    if asset_name:
        return [asset_name]

    groups = registry.get("asset_groups", {})
    if asset_group not in groups:
        raise RegistryError(f"Unknown asset group: {asset_group}")
    return list(groups[asset_group])


def normalize_aliases(values: Iterable[Any]) -> List[str]:
    aliases: List[str] = []
    for value in values:
        alias = normalize_text(value).strip()
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def render_asset_definition(asset_name: str, registry: Mapping[str, Any], repo_root: Path, benchmark_alias: str) -> Dict[str, Any]:
    assets = registry.get("assets", {})
    if asset_name not in assets:
        raise RegistryError(f"Unknown asset: {asset_name}")

    context = {
        "benchmark_alias": benchmark_alias,
        "benchmark_alias_dir": benchmark_alias.replace(".", "_"),
    }
    rendered = apply_template_context(dict(assets[asset_name]), context)
    rendered["asset_name"] = asset_name

    source_list = []
    for source in rendered.get("sources", []):
        source_copy = dict(source)
        local_dir = normalize_text(source_copy.get("local_dir")).strip()
        if local_dir:
            source_copy["local_dir"] = resolve_repo_path(local_dir, repo_root)
        local_path = normalize_text(source_copy.get("local_path")).strip()
        if local_path:
            source_copy["local_path"] = resolve_repo_path(local_path, repo_root)
        source_list.append(source_copy)
    rendered["sources"] = source_list
    rendered["artifact_aliases"] = normalize_aliases(rendered.get("artifact_aliases", []))
    return rendered


def upload_local_asset(
    *,
    entity: str,
    project: str,
    artifact_name: str,
    artifact_type: str,
    local_path: str,
    aliases: Sequence[str],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RegistryError("wandb is required to publish research assets.") from exc

    local = Path(local_path)
    if not local.exists():
        raise RegistryError(f"Resolved asset path does not exist: {local_path}")

    run = wandb.init(
        entity=entity,
        project=project,
        job_type="asset_publish",
        name=f"publish-{artifact_name}-{aliases[0] if aliases else 'latest'}",
        config=dict(metadata),
        reinit=True,
    )
    artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata=dict(metadata))
    if local.is_dir():
        artifact.add_dir(str(local))
    else:
        artifact.add_file(str(local))
    run.log_artifact(artifact, aliases=list(aliases) or None)
    run.finish()
    return {
        "artifact_name": artifact_name,
        "artifact_type": artifact_type,
        "aliases": list(aliases),
        "local_path": str(local),
    }


def update_registry_env_file(env_path: Path, env_var: str, ref_value: str) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: List[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    pattern = re.compile(rf"^\s*export\s+{re.escape(env_var)}=")
    replacement = f'export {env_var}="{ref_value}"'
    updated_lines: List[str] = []
    replaced = False
    for line in existing_lines:
        if pattern.match(line):
            updated_lines.append(replacement)
            replaced = True
        else:
            updated_lines.append(line)

    if not replaced:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append(replacement)

    env_path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")


def publish_asset(
    *,
    asset: Mapping[str, Any],
    entity: str,
    project: str,
    registry_env_path: Path,
    benchmark_alias: str,
) -> Dict[str, Any]:
    resolved_source = materialize_first_available_source(asset.get("sources", []), allow_missing=False)
    if not resolved_source:
        raise RegistryError(f"Could not resolve any source for asset: {asset['asset_name']}")

    aliases = normalize_aliases(asset.get("artifact_aliases", []))
    if not aliases:
        aliases = ["latest"]

    metadata = {
        "asset_name": asset["asset_name"],
        "display_name": asset.get("display_name"),
        "benchmark_alias": benchmark_alias,
        "source_type": resolved_source.get("source_type"),
        "source_ref": resolved_source.get("source_ref"),
        "registry_env_var": asset.get("registry_env_var"),
    }

    upload_status = upload_local_asset(
        entity=entity,
        project=project,
        artifact_name=normalize_text(asset.get("artifact_name")).strip(),
        artifact_type=normalize_text(asset.get("artifact_type")).strip() or "dataset",
        local_path=normalize_text(resolved_source.get("local_path")).strip(),
        aliases=aliases,
        metadata=metadata,
    )

    primary_ref = f"{entity}/{project}/{upload_status['artifact_name']}:{aliases[0]}"
    env_var = normalize_text(asset.get("registry_env_var")).strip()
    if env_var:
        update_registry_env_file(registry_env_path, env_var, primary_ref)

    return {
        "asset_name": asset["asset_name"],
        "artifact_name": upload_status["artifact_name"],
        "artifact_type": upload_status["artifact_type"],
        "aliases": aliases,
        "source_ref": resolved_source.get("source_ref"),
        "local_path": resolved_source.get("local_path"),
        "registry_ref": primary_ref,
        "registry_env_var": env_var,
    }


def main() -> int:
    args = parse_args()
    entity = normalize_text(args.wandb_entity).strip()
    project = normalize_text(args.wandb_project).strip()
    if not entity or not project:
        raise RegistryError("--wandb-entity and --wandb-project are required.")

    if normalize_text(args.source_env_file).strip():
        load_exported_env_file(resolve_repo_path(args.source_env_file, ROOT))
    registry = load_publish_registry(args.manifest_path, ROOT)
    asset_names = resolve_asset_names(registry, args.asset, args.asset_group)
    registry_env_path = Path(resolve_repo_path(args.registry_env_file, ROOT))

    statuses: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for asset_name in asset_names:
        asset = render_asset_definition(asset_name, registry, ROOT, args.benchmark_alias)
        try:
            status = publish_asset(
                asset=asset,
                entity=entity,
                project=project,
                registry_env_path=registry_env_path,
                benchmark_alias=args.benchmark_alias,
            )
        except Exception as exc:  # pragma: no cover - integration-like branch
            status = {"asset_name": asset_name, "status": "failed", "error": str(exc)}
            failures.append(status)
            statuses.append(status)
            if not args.continue_on_error:
                print(json.dumps({"statuses": statuses, "failures": failures}, indent=2, sort_keys=True))
                return 1
            continue

        status["status"] = "published"
        statuses.append(status)

    print(json.dumps({"statuses": statuses, "failures": failures}, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
