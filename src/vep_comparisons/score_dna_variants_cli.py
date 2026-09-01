"""Command line entry point for one NTv3 scoring shard."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .dna_variant_scoring import (
    InputValidationError,
    ModelFactory,
    SUPPORTED_MODELS,
    ShardCompatibilityError,
    score_dna_variants,
)
from .ntv3_model import ModelPackageError


MANIFESTS = {
    "ntv3-100m-pre": Path(__file__).parent / "model_manifests" / "ntv3_100m_pre.json",
    "ntv3-650m-pre": Path(__file__).parent / "model_manifests" / "ntv3_650m_pre.json",
}
MODEL_DIRECTORIES = {
    "ntv3-100m-pre": "NTv3_100M_pre",
    "ntv3-650m-pre": "NTv3_650M_pre",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    model_root = os.environ.get("NTV3_MODEL_ROOT")
    root = Path(model_root) if model_root else None
    parser = argparse.ArgumentParser(
        prog="score-dna-variants",
        description="Score one deterministic genomic-SNV shard with local NTv3 files.",
    )
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default="ntv3-100m-pre")
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--genes",
        type=Path,
        help="GTF for gene-aware contexts; omit to score centered windows only",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-code-dir", type=Path, default=None if root is None else root / "ntv3_base_model")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-length", type=int, default=8192)
    parser.add_argument("--min-variant-margin", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.model_dir is None and root is not None:
        args.model_dir = root / MODEL_DIRECTORIES[args.model]
    if args.model_manifest is None:
        args.model_manifest = MANIFESTS[args.model]
    if args.model_dir is None or args.model_code_dir is None:
        parser.error("--model-dir and --model-code-dir are required when NTV3_MODEL_ROOT is not set")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= I < N")
    if args.window_length <= 0 or args.window_length % 128:
        parser.error("--window-length must be positive and divisible by 128")
    if args.genes is not None and not 0 <= args.min_variant_margin < args.window_length / 2:
        parser.error("--min-variant-margin must satisfy 0 <= M < W / 2")
    return args


def run_cli(
    argv: list[str] | None = None, *, model_factory: ModelFactory | None = None
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    logging.info("Resolved model weights: %s", args.model_dir.resolve())
    logging.info("Resolved model code: %s", args.model_code_dir.resolve())
    kwargs = {} if model_factory is None else {"model_factory": model_factory}
    try:
        summary = score_dna_variants(
            variants_path=args.variants,
            reference_path=args.reference,
            genes_path=args.genes,
            model_dir=args.model_dir,
            model_code_dir=args.model_code_dir,
            output_dir=args.output_dir,
            manifest_path=args.model_manifest,
            model_name=args.model,
            window_length=args.window_length,
            min_variant_margin=args.min_variant_margin,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            command_line=["score-dna-variants", *(argv if argv is not None else sys.argv[1:])],
            **kwargs,
        )
    except (InputValidationError, ModelPackageError, ShardCompatibilityError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    logging.info(
        "%s shard: candidates=%d variants=%d contexts=%d output=%s",
        "Reused" if summary.reused else "Completed",
        summary.candidates,
        summary.unique_variants,
        summary.contexts,
        summary.output_path,
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_cli(argv))


if __name__ == "__main__":
    main()
