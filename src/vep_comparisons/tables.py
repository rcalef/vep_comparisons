"""Shared score-table and shard file operations."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def shard_path(output_dir: Path, num_shards: int, shard_index: int) -> Path:
    stem = f"shard-{shard_index:05d}-of-{num_shards:05d}"
    return output_dir / f"{stem}.tsv.gz"


def write_score_table(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_score_table(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"output schema mismatch: {path}")
        return list(reader)
