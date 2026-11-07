"""Check full manifest coverage and collate deterministic embedding shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .constants import DEFAULT_MODELS, DIMENSIONS, PROTEIN_MODELS


def expected_sites(manifest: pl.DataFrame, model: str) -> pl.DataFrame:
    data = manifest.filter(pl.col('cohort') == 'shared_missense') if model in PROTEIN_MODELS else manifest
    return (
        data
        .select('site_id', 'context_id')
        .unique()
        .sort('site_id', 'context_id')
    )


def collate_model(
    manifest: pl.DataFrame, model: str, input_dir: Path, output: Path, num_shards: int,
) -> dict[str, str | int]:
    expected = expected_sites(manifest, model)
    identifiers, arrays = [], []
    for shard in range(num_shards):
        path = input_dir / f'shard-{shard:05d}-of-{num_shards:05d}.npz'
        with np.load(path, allow_pickle=False) as data:
            vectors = data['embeddings']
            assert vectors.shape == (len(data['site_id']), DIMENSIONS[model])
            assert vectors.dtype == np.float32 and np.isfinite(vectors).all()
            identifiers.append(pl.DataFrame({'site_id': data['site_id'], 'context_id': data['context_id']}))
            arrays.append(vectors)
    observed = pl.concat(identifiers).with_row_index('embedding_row')
    assert observed.height == observed.unique(['site_id', 'context_id']).height == expected.height
    aligned = expected.join(observed, on=['site_id', 'context_id'], how='left', validate='1:1', maintain_order='left')
    assert aligned['embedding_row'].null_count() == 0
    vectors = np.concatenate(arrays)[aligned['embedding_row'].to_numpy()]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, site_id=np.array(aligned['site_id'].to_list()),
             context_id=np.array(aligned['context_id'].to_list()), embeddings=vectors)
    return {'model': model, 'sites': len(vectors), 'dimension': vectors.shape[1],
            'shards': num_shards, 'bytes': output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-shards', type=int, default=4)
    parser.add_argument('--evo2-num-shards', type=int, help='override --num-shards for Evo2')
    parser.add_argument('--models', nargs='+', choices=list(DIMENSIONS), default=list(DEFAULT_MODELS))
    args = parser.parse_args()
    manifest = pl.read_parquet(args.manifest)
    rows = []
    for model in args.models:
        shards = args.evo2_num_shards if model == 'evo2' and args.evo2_num_shards is not None else args.num_shards
        row = collate_model(manifest, model, args.input_dir / model, args.output_dir / f'{model}.npz', shards)
        rows.append(row)
        print(f'{model}: {row["sites"]} sites, {row["dimension"]} dimensions, coverage matched', flush=True)
    pl.DataFrame(rows).write_csv(args.output_dir / 'summary.tsv', separator='\t')


if __name__ == '__main__':
    main()
