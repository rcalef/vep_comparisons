"""Command-line entry point for one RiNALMo-giga scoring shard."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .dna_variant_scoring import InputValidationError, ShardCompatibilityError
from .rinalmo_scoring import ModelFactory, SUPPORTED_DTYPES, score_rinalmo_variants

DEFAULT_MANIFEST = Path(__file__).parent / "model_manifests" / "rinalmo_giga.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="score-rinalmo-variants", description="Score one deterministic transcript-SNV shard with RiNALMo-giga.")
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--transcripts", "--transcript-fasta", dest="transcripts", required=True, type=Path)
    parser.add_argument("--gff3", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= I < N")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def run_cli(argv: list[str] | None = None, *, model_factory: ModelFactory | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    kwargs = {} if model_factory is None else {"model_factory": model_factory}
    try:
        summary = score_rinalmo_variants(
            variants_path=args.variants, transcript_fasta_path=args.transcripts,
            gff3_path=args.gff3, weights=args.weights,
            manifest_path=args.checkpoint_manifest, output_dir=args.output_dir,
            num_shards=args.num_shards, shard_index=args.shard_index,
            device=args.device, dtype=args.dtype, batch_size=args.batch_size, **kwargs,
        )
    except (InputValidationError, ShardCompatibilityError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    logging.info("%s shard %d/%d: rows=%d output=%s", "Reused" if summary.reused else "Completed", args.shard_index, args.num_shards, summary.rows, summary.output_path)
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_cli(argv))


if __name__ == "__main__":
    main()
