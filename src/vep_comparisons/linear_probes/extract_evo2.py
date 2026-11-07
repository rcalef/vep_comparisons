"""Extract four 128-nt pooled Evo2 allele vectors and matched zero-shot scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..dna.evo2 import Evo2Model
from ..dna.inputs import DnaCandidate, IndexedReference
from ..dna.windows import context_for, plan_candidates
from ..dna.workflow import OUTPUT_COLUMNS, assemble_rows
from ..tables import shard_path, write_score_table
from .embedding_io import save_embeddings
from .inputs import selected_sites
from .profiling import profile_extraction


def extract_embeddings(
    sites: pl.DataFrame, reference_path: Path, model: Evo2Model,
    output_dir: Path, *, batch_size: int, num_shards: int, shard_index: int,
    profile: bool = False, device: str = "cuda",
) -> tuple[Path, Path]:
    rows = list(sites.iter_rows(named=True))
    candidates = [DnaCandidate(
        row["variant"], row["gene"], row["feature"], row["chromosome"],
        row["start"], row["end"], row["ref"], row["alt"], index,
    ) for index, row in enumerate(rows)]
    with IndexedReference(reference_path) as reference:
        plans = plan_candidates(
            candidates, {}, reference, window_length=8192, min_variant_margin=0,
            include_gene_context=False,
        )
        contexts = [context_for(plan.candidate, plan.centered) for plan in plans]
        with profile_extraction(profile, device, len(contexts)):
            scores, embeddings = model.score_and_embed_contexts(
                contexts, reference=reference, batch_size=batch_size,
            )
    embedding_path = save_embeddings(
        output_dir, rows, embeddings, num_shards=num_shards, shard_index=shard_index,
    )
    score_path = shard_path(output_dir, num_shards, shard_index)
    write_score_table(
        score_path, OUTPUT_COLUMNS,
        assemble_rows(plans, scores, window_length=8192, model_name="evo2-7b"),
    )
    print(f"Saved matched zero-shot scores to {score_path}", flush=True)
    return embedding_path, score_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="maximum sequences per forward pass (four sequences per site)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert args.batch_size > 0
    sites = selected_sites(pl.read_parquet(args.manifest), args.num_shards, args.shard_index)
    if sites.is_empty():
        raise ValueError("empty shard; use fewer shards")
    print(f"Loading Evo2 checkpoint: {args.checkpoint}", flush=True)
    model = Evo2Model(args.checkpoint, device=args.device, dtype=args.dtype)
    extract_embeddings(
        sites, args.reference, model, args.output_dir,
        batch_size=args.batch_size, num_shards=args.num_shards, shard_index=args.shard_index,
        profile=args.profile, device=args.device,
    )


if __name__ == "__main__":
    main()
