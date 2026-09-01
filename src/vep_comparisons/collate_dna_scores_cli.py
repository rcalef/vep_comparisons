"""Command line entry point for deterministic NTv3 shard collation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dna_variant_scoring import InputValidationError, ShardCompatibilityError, collate_dna_scores


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collate-dna-variant-scores",
        description="Validate and collate a complete set of NTv3 score shards.",
    )
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--input-dir", "--output-dir", dest="input_dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    return args


def run_cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output, metadata = collate_dna_scores(
            variants_path=args.variants,
            output_dir=args.input_dir,
            output_path=args.output,
            num_shards=args.num_shards,
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
