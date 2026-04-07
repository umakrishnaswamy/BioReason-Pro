#!/usr/bin/env python3
"""Build the reasoning dataset from a temporal split artifact."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd
import requests
from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bioreason2.dataset.cafa5.processor import _build_response


SPLIT_TO_FILE_PREFIX = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}

ASPECT_TO_COLUMN = {
    "P": "go_bp",
    "F": "go_mf",
    "C": "go_cc",
}

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FIELDS = "accession,organism_name,protein_name,cc_function,sequence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-split-dir", required=True, help="Path to the temporal split artifact directory.")
    parser.add_argument("--reasoning-output-dir", required=True, help="Directory for the reasoning DatasetDict.")
    parser.add_argument(
        "--metadata-cache-path",
        default=None,
        help="Optional path for cached UniProt metadata TSV. Defaults under the temporal split artifact.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def ensure_dir(path_value: Path) -> Path:
    path_value.mkdir(parents=True, exist_ok=True)
    return path_value


def load_split_labels(temporal_split_dir: Path, split: str) -> pd.DataFrame:
    prefix = SPLIT_TO_FILE_PREFIX[split]
    path = temporal_split_dir / f"{prefix}_assigned_labels.tsv"
    df = pd.read_csv(path, sep="\t")
    expected_columns = {"DB_ID", "GO_ID", "Aspect"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[list(expected_columns)].copy()


def aggregate_labels_by_protein(label_df: pd.DataFrame, split: str) -> pd.DataFrame:
    protein_rows: List[Dict[str, Any]] = []
    for protein_id, group in label_df.groupby("DB_ID"):
        row = {
            "protein_id": str(protein_id),
            "split": split,
            "go_bp": [],
            "go_mf": [],
            "go_cc": [],
        }
        for aspect, column_name in ASPECT_TO_COLUMN.items():
            terms = sorted(group.loc[group["Aspect"] == aspect, "GO_ID"].astype(str).drop_duplicates().tolist())
            row[column_name] = terms
        protein_rows.append(row)

    protein_rows.sort(key=lambda item: item["protein_id"])
    return pd.DataFrame(protein_rows)


def extract_protein_name(entry: Mapping[str, Any]) -> str:
    description = entry.get("proteinDescription") or {}
    recommended = description.get("recommendedName") or {}
    full_name = (recommended.get("fullName") or {}).get("value")
    if full_name:
        return str(full_name)

    for alt in description.get("alternativeNames") or []:
        value = (alt.get("fullName") or {}).get("value")
        if value:
            return str(value)
    return ""


def extract_function_text(entry: Mapping[str, Any]) -> str:
    comments = entry.get("comments") or []
    texts: List[str] = []
    for comment in comments:
        for text in comment.get("texts") or []:
            value = text.get("value")
            if value:
                normalized = " ".join(str(value).split())
                if normalized and normalized not in texts:
                    texts.append(normalized)
    return "\n".join(texts)


def fetch_uniprot_metadata(accessions: Sequence[str], cache_path: Path, *, batch_size: int, sleep_seconds: float) -> pd.DataFrame:
    existing = pd.DataFrame(columns=["protein_id", "sequence", "organism", "protein_name", "protein_function"])
    if cache_path.exists():
        existing = pd.read_csv(cache_path, sep="\t")
        existing = existing.fillna("")

    cached_ids = set(existing["protein_id"].astype(str).tolist()) if not existing.empty else set()
    missing_ids = [protein_id for protein_id in accessions if protein_id not in cached_ids]
    if not missing_ids:
        return existing

    session = requests.Session()
    fetched_rows: List[Dict[str, str]] = []
    total_batches = math.ceil(len(missing_ids) / batch_size)

    for index in range(total_batches):
        batch = missing_ids[index * batch_size : (index + 1) * batch_size]
        query = " OR ".join(f"accession:{protein_id}" for protein_id in batch)
        params = {
            "query": query,
            "format": "json",
            "fields": UNIPROT_FIELDS,
            "size": len(batch),
        }
        response = session.get(UNIPROT_SEARCH_URL, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        seen: set[str] = set()
        for entry in payload.get("results") or []:
            accession = str(entry.get("primaryAccession") or "").strip()
            if not accession:
                continue
            seen.add(accession)
            fetched_rows.append(
                {
                    "protein_id": accession,
                    "sequence": str((entry.get("sequence") or {}).get("value") or ""),
                    "organism": str((entry.get("organism") or {}).get("scientificName") or ""),
                    "protein_name": extract_protein_name(entry),
                    "protein_function": extract_function_text(entry),
                }
            )

        missing_after_batch = [protein_id for protein_id in batch if protein_id not in seen]
        for protein_id in missing_after_batch:
            fetched_rows.append(
                {
                    "protein_id": protein_id,
                    "sequence": "",
                    "organism": "",
                    "protein_name": "",
                    "protein_function": "",
                }
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    fetched_df = pd.DataFrame(fetched_rows).drop_duplicates(subset=["protein_id"], keep="last")
    combined = pd.concat([existing, fetched_df], ignore_index=True).drop_duplicates(subset=["protein_id"], keep="last")
    combined = combined.fillna("")
    ensure_dir(cache_path.parent)
    combined.to_csv(cache_path, sep="\t", index=False)
    return combined


def build_split_tables(temporal_split_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        split: aggregate_labels_by_protein(load_split_labels(temporal_split_dir, split), split)
        for split in ("train", "validation", "test")
    }


def attach_metadata(split_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata_df.copy()
    metadata["protein_id"] = metadata["protein_id"].astype(str)
    merged = split_df.merge(metadata, on="protein_id", how="left")
    for column in ("sequence", "organism", "protein_name", "protein_function"):
        if column not in merged.columns:
            merged[column] = ""
        merged[column] = merged[column].fillna("")

    merged["go_pred"] = ""
    merged["interpro_formatted"] = ""
    merged["ppi_formatted"] = ""
    return merged


def build_reasoning_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[MutableMapping[str, Any]] = []
    for record in df.to_dict(orient="records"):
        response_row = pd.Series(record)
        reasoning, final_answer = _build_response(
            response_row,
            interpro_metadata=None,
            include_go_defs=True,
            interpro_in_prompt=False,
            predict_interpro=False,
        )
        record["reasoning"] = reasoning
        record["final_answer"] = final_answer
        rows.append(record)
    return pd.DataFrame(rows)


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    normalized = df.copy()
    for list_column in ("go_bp", "go_mf", "go_cc"):
        normalized[list_column] = normalized[list_column].apply(lambda value: value if isinstance(value, list) else [])
    for column in normalized.columns:
        if column not in ("go_bp", "go_mf", "go_cc"):
            normalized[column] = normalized[column].fillna("").astype(str)
    return Dataset.from_pandas(normalized, preserve_index=False)


def write_dataset_dict(split_tables: Mapping[str, pd.DataFrame], output_dir: Path) -> Dict[str, int]:
    dataset_dict = DatasetDict(
        {
            split: dataframe_to_dataset(df)
            for split, df in split_tables.items()
        }
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_dict.save_to_disk(str(output_dir))
    summary = {split: len(dataset_dict[split]) for split in dataset_dict.keys()}
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    temporal_split_dir = Path(args.temporal_split_dir).resolve()
    reasoning_output_dir = Path(args.reasoning_output_dir).resolve()
    metadata_cache_path = (
        Path(args.metadata_cache_path).resolve()
        if args.metadata_cache_path
        else (temporal_split_dir / "uniprot_protein_metadata.tsv")
    )

    split_tables = build_split_tables(temporal_split_dir)
    all_accessions = sorted(
        {
            protein_id
            for split_df in split_tables.values()
            for protein_id in split_df["protein_id"].astype(str).tolist()
        }
    )

    if args.force_refresh and metadata_cache_path.exists():
        metadata_cache_path.unlink()

    metadata_df = fetch_uniprot_metadata(
        all_accessions,
        metadata_cache_path,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
    )

    reasoning_tables: Dict[str, pd.DataFrame] = {}
    for split, split_df in split_tables.items():
        merged = attach_metadata(split_df, metadata_df)
        reasoning_tables[split] = build_reasoning_columns(merged)

    reasoning_summary = write_dataset_dict(reasoning_tables, reasoning_output_dir)

    build_summary = {
        "temporal_split_dir": str(temporal_split_dir),
        "metadata_cache_path": str(metadata_cache_path),
        "reasoning_output_dir": str(reasoning_output_dir),
        "reasoning_counts": reasoning_summary,
        "proteins": len(all_accessions),
    }
    (reasoning_output_dir / "build_metadata.json").write_text(
        json.dumps(build_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(build_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
