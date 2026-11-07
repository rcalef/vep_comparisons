"""Export the selected context union for model-specific extraction environments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


def selected_sites(manifest: pl.DataFrame, num_shards: int, shard_index: int) -> pl.DataFrame:
    """One genomic input per site, shared across both tasks and cohorts."""
    assert 0 <= shard_index < num_shards
    return (
        manifest
        .sort("site_id", "context_id")
        .unique("site_id", keep="first", maintain_order=True)
        .gather_every(num_shards, offset=shard_index)
    )


def main() -> None:
    import polars as pl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = pl.read_parquet(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ('dna_rna', manifest),
        ('protein', manifest.filter(pl.col('cohort') == 'shared_missense')),
    ):
        sites = (
            data
            .select('site_id', 'context_id', 'variant', 'gene', 'feature', 'chromosome',
                    'start', 'end', 'ref', 'alt', 'cdna_position', 'protein_position',
                    'amino_acids', 'consequence', 'biotype')
            .unique('context_id')
            .sort('feature', 'start', 'context_id')
        )
        sites.write_csv(args.output_dir / f'{name}.tsv', separator='\t')
        print(f'{name}: {sites.height} contexts')


if __name__ == '__main__':
    main()
