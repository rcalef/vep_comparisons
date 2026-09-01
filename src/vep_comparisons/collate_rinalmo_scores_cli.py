"""Command-line entry point for verified RiNALMo-giga shard collation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dna_variant_scoring import InputValidationError, ShardCompatibilityError
from .rinalmo_scoring import SUPPORTED_DTYPES, collate_rinalmo_scores
from .score_rinalmo_variants_cli import DEFAULT_MANIFEST


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="collate-rinalmo-scores", description="Validate complete RiNALMo shards and publish one genomic-order table.")
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--transcripts", "--transcript-fasta", dest="transcripts", required=True, type=Path)
    parser.add_argument("--gff3", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-dir", "--output-dir", dest="input_dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="bfloat16")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    return args


def run_cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output, metadata = collate_rinalmo_scores(
            variants_path=args.variants, transcript_fasta_path=args.transcripts,
            gff3_path=args.gff3, weights=args.weights,
            manifest_path=args.checkpoint_manifest, output_dir=args.input_dir,
            output_path=args.output, num_shards=args.num_shards, dtype=args.dtype,
        )
    except (InputValidationError, ShardCompatibilityError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(f"output={output}")
    print(f"manifest={metadata}")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_cli(argv))


if __name__ == "__main__":
    main()
