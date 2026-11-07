"""Save reproducible random and long-input subsets for GPU extraction pilots."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..protein.inputs import normalize_transcript, read_transcript_fasta


def random_sites(sites: pl.DataFrame, count: int, seed: int) -> pl.DataFrame:
    return (
        sites
        .sort('context_id')
        .sample(n=min(count, sites.height), seed=seed, shuffle=True)
        .sort('context_id')
    )


def longest_sites(sites: pl.DataFrame, length_column: str, count: int) -> pl.DataFrame:
    """Use distinct transcripts so stress batches contain multiple long inputs."""
    return (
        sites
        .sort([length_column, 'context_id'], descending=[True, False])
        .unique('feature', keep='first', maintain_order=True)
        .head(count)
        .sort('context_id')
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inputs-dir', type=Path, required=True)
    parser.add_argument('--sequences', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--random-sites', type=int, default=32)
    parser.add_argument('--long-sites', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261005)
    args = parser.parse_args()
    assert args.random_sites > 0 and args.long_sites > 0
    dna_rna = pl.read_csv(args.inputs_dir / 'dna_rna.tsv', separator='\t')
    protein = pl.read_csv(args.inputs_dir / 'protein.tsv', separator='\t')
    sequences = read_transcript_fasta(args.sequences)
    protein = protein.with_columns(protein_length=pl.Series([
        len(sequences[normalize_transcript(feature)]) for feature in protein['feature']
    ]))
    cases = {
        'random/dna_rna': random_sites(dna_rna, args.random_sites, args.seed),
        'random/protein': random_sites(protein, args.random_sites, args.seed),
        'long/dna_rna': longest_sites(dna_rna, 'cdna_position', args.long_sites),
        'long/protein': longest_sites(protein, 'protein_length', args.long_sites),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, sites in cases.items():
        path = args.output_dir / f'{name}.tsv'
        path.parent.mkdir(parents=True, exist_ok=True)
        sites.write_csv(path, separator='\t')
        print(f'{name}: {sites.height} contexts')
    # NTv3 uses fixed 8,192-base windows: only the random case is needed.
    manifest = pl.read_parquet(args.manifest)
    (
        manifest
        .filter(pl.col('context_id').is_in(cases['random/dna_rna']['context_id'].to_list()))
        .write_parquet(args.output_dir / 'random/ntv3.parquet')
    )
    pl.DataFrame({
        'seed': [args.seed], 'random_sites': [args.random_sites], 'long_sites': [args.long_sites],
        'manifest': [str(args.manifest.resolve())], 'inputs_dir': [str(args.inputs_dir.resolve())],
        'sequences': [str(args.sequences.resolve())],
    }).write_csv(args.output_dir / 'settings.tsv', separator='\t')


if __name__ == '__main__':
    main()
