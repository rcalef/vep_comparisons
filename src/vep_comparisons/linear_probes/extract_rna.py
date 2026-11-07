"""Extract RiNALMo or Orthrus context vectors in its existing RNA environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..rna.common import RnaCandidate, normalize_stable_id, prepare_requests
from .embedding_io import read_sites, save_embeddings
from .profiling import profile_extraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sites', type=Path, required=True)
    parser.add_argument('--model', choices=['rinalmo', 'orthrus'], required=True)
    parser.add_argument('--transcripts', type=Path, required=True)
    parser.add_argument('--gff3', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    rows = read_sites(args.sites, args.num_shards, args.shard_index)
    if not rows:
        raise ValueError('empty shard; use fewer shards')
    candidates = [
        RnaCandidate(
            row['variant'], normalize_stable_id(row['gene']), normalize_stable_id(row['feature']),
            row['chromosome'], int(row['start']), int(row['end']), int(row['cdna_position']),
            row['ref'], row['alt'], index,
        )
        for index, row in enumerate(rows)
    ]
    transcripts = {candidate.feature for candidate in candidates}
    print(f'Preparing {len(rows)} {args.model} sites from {len(transcripts)} transcripts', flush=True)
    if args.model == 'rinalmo':
        from ..rna.rinalmo.annotation import read_rinalmo_annotations
        from ..rna.rinalmo.model import RiNALMoModel

        annotations = read_rinalmo_annotations(args.gff3, args.transcripts, transcripts)
        requests = prepare_requests(candidates, annotations)
        print(f'Loading RiNALMo checkpoint: {args.checkpoint}', flush=True)
        model = RiNALMoModel(args.checkpoint, args.device, args.dtype or 'bfloat16')
    else:
        from ..rna.orthrus.annotation import read_transcript_annotations
        from ..rna.orthrus.model import OrthrusModel

        if args.dtype not in (None, 'float32'):
            parser.error('Orthrus uses float32 in its existing scoring environment')
        annotations = read_transcript_annotations(args.gff3, args.transcripts, transcripts)
        requests = prepare_requests(candidates, annotations)
        print(f'Loading Orthrus checkpoint: {args.checkpoint}', flush=True)
        model = OrthrusModel(args.checkpoint, args.device)
    print(f'Extracting {len(requests)} site vectors', flush=True)
    with profile_extraction(args.profile, args.device, len(requests)):
        vectors = model.embed_requests(requests, batch_size=args.batch_size)
    save_embeddings(
        args.output_dir, rows, np.stack([vectors[request.key] for request in requests]),
        num_shards=args.num_shards, shard_index=args.shard_index,
    )


if __name__ == '__main__':
    main()
