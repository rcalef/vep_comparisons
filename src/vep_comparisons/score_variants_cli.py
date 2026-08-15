"""CLI for inference-only masked-marginal protein variant scoring."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .variant_models import MODEL_REGISTRY
from .variant_scoring import InputValidationError, ModelFactory, score_protein_variants


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="score-protein-variants",
        description=(
            "Score protein-coding missense variants with one local ESM-C or "
            "SaProt checkpoint."
        ),
    )
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=(
            Path(value) if (value := os.environ.get("MAGNETON_MODEL_DIR")) else None
        ),
        help="checkpoint root (default: MAGNETON_MODEL_DIR)",
    )
    parser.add_argument(
        "--structure-tokens",
        type=Path,
        help="ENST-keyed 3Di FASTA; required for SaProt",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        help=(
            "residues per inference window (default: model capacity; cannot "
            "exceed model capacity)"
        ),
    )
    parser.add_argument(
        "--long-sequence-mode",
        choices=("window", "null"),
        default="window",
        help="tile long proteins or retain the legacy null-score behavior",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of masked protein positions per model forward pass",
    )
    args = parser.parse_args(argv)
    if args.model_dir is None:
        parser.error("--model-dir is required when MAGNETON_MODEL_DIR is not set")
    if MODEL_REGISTRY[args.model].family == "saprot" and args.structure_tokens is None:
        parser.error(f"--structure-tokens is required for {args.model}")
    return args


def run_cli(
    argv: list[str] | None = None,
    *,
    model_factory: ModelFactory | None = None,
) -> int:
    args = parse_args(argv)
    kwargs = {}
    if model_factory is not None:
        kwargs["model_factory"] = model_factory
    try:
        score_protein_variants(
            variants_path=args.variants,
            sequences_path=args.sequences,
            model_name=args.model,
            model_root=args.model_dir,
            structure_tokens_path=args.structure_tokens,
            output=args.output,
            device=args.device,
            dtype=args.dtype,
            max_sequence_length=args.max_sequence_length,
            long_sequence_mode=args.long_sequence_mode,
            batch_size=args.batch_size,
            **kwargs,
        )
    except InputValidationError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_cli(argv))


if __name__ == "__main__":
    main()
