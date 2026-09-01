"""Command-line interface for conservative SaProt structure recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from .saprot_recovery import recover_saprot_structures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover transcript-keyed SaProt tokens only from exact, full-length "
            "AlphaFoldDB models."
        )
    )
    parser.add_argument("--mismatches", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--existing-tokens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--variants", type=Path)
    parser.add_argument("--foldseek", default="foldseek")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-requests", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = recover_saprot_structures(
        mismatches_path=args.mismatches,
        translations_path=args.translations,
        existing_tokens_path=args.existing_tokens,
        output_dir=args.output_dir,
        foldseek=args.foldseek,
        threads=args.threads,
        merged_output=args.merged_output,
        variants_path=args.variants,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
        max_requests=args.max_requests,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        f"candidates={summary.candidates} recovered={summary.recovered} "
        f"unresolved={summary.unresolved}"
    )
    print(f"manifest={summary.manifest}")
    print(f"recovered_fasta={summary.recovered_fasta}")
    print(f"merged_fasta={summary.merged_fasta}")


if __name__ == "__main__":
    main()
