#!/usr/bin/env python3
"""
Resolve a model source from W&B Registry, Hugging Face, or a local directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreason2.utils.research_registry import materialize_first_available_source, normalize_text, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-registry-path", type=str, default=None)
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-repo-id-env", type=str, default=None)
    parser.add_argument("--local-dir", type=str, default=None)
    parser.add_argument("--source-local-dir", type=str, default=None)
    parser.add_argument("--local-path", type=str, default=None)
    parser.add_argument("--required-path", action="append", default=[])
    parser.add_argument(
        "--print-field",
        type=str,
        choices=["json", "local_path", "source_ref", "source_type"],
        default="local_path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = normalize_text(args.local_dir).strip()
    source_local_dir = normalize_text(args.source_local_dir).strip()
    local_path = normalize_text(args.local_path).strip()
    required_paths = list(args.required_path or [])

    sources = []
    if normalize_text(args.wandb_registry_path).strip():
        sources.append(
            {
                "type": "wandb_artifact",
                "wandb_registry_path": normalize_text(args.wandb_registry_path).strip(),
                "local_dir": resolve_repo_path(local_dir, ROOT) if local_dir else "",
                "required_paths": required_paths,
            }
        )
    if normalize_text(args.hf_repo_id).strip() or normalize_text(args.hf_repo_id_env).strip():
        sources.append(
            {
                "type": "huggingface",
                "repo_id": normalize_text(args.hf_repo_id).strip(),
                "repo_id_env": normalize_text(args.hf_repo_id_env).strip(),
                "local_dir": resolve_repo_path(local_dir, ROOT) if local_dir else "",
                "required_paths": required_paths,
            }
        )
    if source_local_dir:
        sources.append(
            {
                "type": "local_dir",
                "local_dir": resolve_repo_path(source_local_dir, ROOT),
                "required_paths": required_paths,
            }
        )
    if local_path:
        sources.append(
            {
                "type": "local_path",
                "local_path": resolve_repo_path(local_path, ROOT),
            }
        )
    elif local_dir:
        sources.append(
            {
                "type": "local_dir",
                "local_dir": resolve_repo_path(local_dir, ROOT),
                "required_paths": required_paths,
            }
        )

    resolved = materialize_first_available_source(sources)
    if args.print_field == "json":
        print(json.dumps(resolved, indent=2, sort_keys=True))
    else:
        print(normalize_text(resolved.get(args.print_field)).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
