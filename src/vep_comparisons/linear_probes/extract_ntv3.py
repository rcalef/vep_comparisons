"""Extract selected NTv3 site vectors using the existing centered DNA windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..dna.inputs import DnaCandidate, IndexedReference
from ..dna.ntv3 import NTv3Model
from ..dna.windows import centered_window, context_for
from .inputs import selected_sites
from .profiling import profile_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-code-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert args.batch_size > 0
    sites = selected_sites(pl.read_parquet(args.manifest), args.num_shards, args.shard_index)
    if sites.is_empty():
        raise ValueError("empty shard; use fewer shards")
    print(f'Loading NTv3 checkpoint: {args.model_dir}', flush=True)
    model = NTv3Model(
        args.model_dir, args.model_code_dir, device=args.device, dtype=args.dtype,
    )
    with IndexedReference(args.reference) as reference:
        contexts = []
        for i, row in enumerate(sites.iter_rows(named=True)):
            candidate = DnaCandidate(
                row["variant"], row["gene"], row["feature"], row["chromosome"],
                row["start"], row["end"], row["ref"], row["alt"], i,
            )
            window = centered_window(
                candidate, reference.get_reference_length(candidate.chromosome), 8192,
            )
            contexts.append(context_for(candidate, window))
        print(f'Extracting {len(contexts)} site vectors', flush=True)
        with profile_extraction(args.profile, args.device, len(contexts)):
            embeddings = model.embed_contexts(contexts, reference=reference, batch_size=args.batch_size)
    assert np.isfinite(embeddings).all()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}.npz"
    np.savez(
        output, site_id=np.array(sites["site_id"].to_list()),
        context_id=np.array(sites["context_id"].to_list()), embeddings=embeddings,
    )
    print(f"Saved {embeddings.shape} vectors to {output}")


if __name__ == "__main__":
    main()
