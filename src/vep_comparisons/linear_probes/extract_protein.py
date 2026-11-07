"""Extract ESM-C or SaProt vectors with the existing weighted protein windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..protein.inputs import Candidate, normalize_transcript, read_transcript_fasta, validate_candidates
from ..protein.models import MODEL_REGISTRY, load_variant_model
from ..protein.windowing import aggregate_position_embeddings, build_position_requests
from .embedding_io import read_sites, save_embeddings
from .profiling import profile_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sites', type=Path, required=True)
    parser.add_argument('--model', choices=['esmc-600m', 'saprot-650m'], required=True)
    parser.add_argument('--sequences', type=Path, required=True)
    parser.add_argument('--structure-tokens', type=Path)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    spec = MODEL_REGISTRY[args.model]
    if spec.family == 'saprot' and args.structure_tokens is None:
        parser.error('SaProt requires --structure-tokens')
    rows = read_sites(args.sites, args.num_shards, args.shard_index)
    if not rows:
        raise ValueError('empty shard; use fewer shards')
    candidates = [
        Candidate(
            row['variant'], row['gene'], row['feature'], normalize_transcript(row['feature']),
            int(row['protein_position']), row['amino_acids'], *row['amino_acids'].split('/'),
        )
        for row in rows
    ]
    print(f'Preparing {len(rows)} {args.model} sites', flush=True)
    sequences = read_transcript_fasta(args.sequences)
    structures = read_transcript_fasta(args.structure_tokens) if spec.family == 'saprot' else None
    validate_candidates(candidates, sequences, structure_tokens=structures, require_structure=spec.family == 'saprot')
    requests = build_position_requests(
        candidates, sequences, structure_tokens=structures,
        max_sequence_length=spec.capacity, long_sequence_mode='window',
    )
    print(f'Loading {args.model} checkpoint from {args.model_dir}', flush=True)
    model = load_variant_model(spec, args.model_dir, device=args.device, dtype=args.dtype)
    print(f'Extracting {len(requests)} residue/window vectors', flush=True)
    with profile_extraction(args.profile, args.device, len(candidates)):
        vectors = aggregate_position_embeddings(
            requests, model.embed_positions(requests, batch_size=args.batch_size),
            sequences, window_length=spec.capacity,
        )
    save_embeddings(
        args.output_dir, rows,
        np.stack([vectors[(candidate.transcript, candidate.position)] for candidate in candidates]),
        num_shards=args.num_shards, shard_index=args.shard_index,
    )


if __name__ == '__main__':
    main()
