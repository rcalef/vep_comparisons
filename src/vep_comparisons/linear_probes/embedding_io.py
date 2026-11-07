"""Small TSV/NumPy interface shared by the separate extraction environments."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_sites(path: Path, num_shards: int, shard_index: int) -> list[dict[str, str]]:
    assert 0 <= shard_index < num_shards
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    # Keep a transcript in one shard so identical windows can be reused.
    transcripts = sorted({row['feature'] for row in rows})
    selected = set(transcripts[shard_index::num_shards])
    return [row for row in rows if row['feature'] in selected]


def save_embeddings(
    output_dir: Path, rows: list[dict[str, str]], embeddings: np.ndarray,
    *, num_shards: int, shard_index: int,
) -> Path:
    assert embeddings.ndim == 2 and len(embeddings) == len(rows)
    assert np.isfinite(embeddings).all()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'shard-{shard_index:05d}-of-{num_shards:05d}.npz'
    np.savez(
        output, site_id=np.array([row['site_id'] for row in rows]),
        context_id=np.array([row['context_id'] for row in rows]),
        embeddings=embeddings.astype(np.float32),
    )
    print(f'Saved {embeddings.shape} vectors to {output}')
    return output
