"""Unified command-line interface for all workflows."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

Handler = Callable[[argparse.Namespace], None]


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _shards(parser: argparse.ArgumentParser, *, collate: bool = False) -> None:
    parser.add_argument("--num-shards", type=_positive, required=collate, default=None if collate else 1)
    if not collate:
        parser.add_argument("--shard-index", type=int, default=0)


def _transcript_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--transcripts", "--transcript-fasta", dest="transcripts", required=True, type=Path)
    parser.add_argument("--gff3", required=True, type=Path)


def _run_curate(args: argparse.Namespace) -> None:
    from ..curation import curate_variants

    curate_variants(
        vep_output=args.vep_output,
        selected_variants=args.selected_variants,
        output_prefix=args.output_prefix,
        has_target_genes=args.has_target_genes,
        neg_labels=args.neg_labels,
        neg_fraction=args.neg_fraction,
        seed=args.seed,
    )


def _run_protein(args: argparse.Namespace) -> None:
    from ..protein import score_protein_variants

    score_protein_variants(
        variants_path=args.variants,
        sequences_path=args.sequences,
        model_name=args.model,
        model_root=args.model_dir,
        output=args.output,
        structure_tokens_path=args.structure_tokens,
        ignore_missing_structure_tokens=args.ignore_missing_structure_tokens,
        device=args.device,
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        long_sequence_mode=args.long_sequence_mode,
        batch_size=args.batch_size,
    )


def _run_recovery(args: argparse.Namespace) -> None:
    from ..protein.recovery import recover_saprot_structures

    summary = recover_saprot_structures(
        mismatches_path=args.mismatches,
        translations_path=args.translations,
        existing_tokens_path=args.existing_tokens,
        output_dir=args.output_dir,
        merged_output=args.merged_output,
        variants_path=args.variants,
        foldseek=args.foldseek,
        threads=args.threads,
        max_requests=args.max_requests,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"candidates={summary.candidates} recovered={summary.recovered} unresolved={summary.unresolved}")
    print(f"recovered_fasta={summary.recovered_fasta}")
    print(f"merged_fasta={summary.merged_fasta}")


def _run_dna_score(args: argparse.Namespace) -> None:
    from ..dna import score_dna_variants

    summary = score_dna_variants(
        variants_path=args.variants,
        reference_path=args.reference,
        genes_path=args.genes,
        model_dir=args.model_dir,
        model_code_dir=args.model_code_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        window_length=args.window_length,
        min_variant_margin=args.min_variant_margin,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print(f"candidates={summary.candidates} variants={summary.unique_variants} contexts={summary.contexts} output={summary.output_path}")


def _run_dna_collate(args: argparse.Namespace) -> None:
    from ..dna import collate_dna_scores

    output = collate_dna_scores(
        variants_path=args.variants,
        output_dir=args.input_dir,
        output_path=args.output,
        num_shards=args.num_shards,
    )
    print(f"output={output}")


def _run_orthrus_score(args: argparse.Namespace) -> None:
    from ..rna.orthrus import score_orthrus_variants

    summary = score_orthrus_variants(
        variants_path=args.variants,
        transcript_fasta_path=args.transcripts,
        gff3_path=args.gff3,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"rows={summary.rows} output={summary.output_path}")


def _run_orthrus_collate(args: argparse.Namespace) -> None:
    from ..rna.orthrus import collate_orthrus_scores

    output = collate_orthrus_scores(
        variants_path=args.variants,
        transcript_fasta_path=args.transcripts,
        gff3_path=args.gff3,
        checkpoint=args.checkpoint,
        output_dir=args.input_dir,
        output_path=args.output,
        num_shards=args.num_shards,
    )
    print(f"output={output}")


def _run_rinalmo_score(args: argparse.Namespace) -> None:
    from ..rna.rinalmo import score_rinalmo_variants

    summary = score_rinalmo_variants(
        variants_path=args.variants,
        transcript_fasta_path=args.transcripts,
        gff3_path=args.gff3,
        weights=args.weights,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    print(f"rows={summary.rows} output={summary.output_path}")


def _run_rinalmo_collate(args: argparse.Namespace) -> None:
    from ..rna.rinalmo import collate_rinalmo_scores

    output = collate_rinalmo_scores(
        variants_path=args.variants,
        transcript_fasta_path=args.transcripts,
        gff3_path=args.gff3,
        weights=args.weights,
        output_dir=args.input_dir,
        output_path=args.output,
        num_shards=args.num_shards,
        dtype=args.dtype,
    )
    print(f"output={output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vep-comparisons")
    domains = parser.add_subparsers(dest="domain", required=True)

    curate = domains.add_parser("curate")
    curate.add_argument("--vep-output", required=True, type=Path)
    curate.add_argument("--selected-variants", required=True, type=Path)
    curate.add_argument("--has-target-genes", action="store_true")
    curate.add_argument("--output-prefix", required=True, type=Path)
    curate.add_argument("--neg-labels", action="append")
    curate.add_argument("--neg-fraction", type=float, default=0.1)
    curate.add_argument("--seed", type=int, default=42)
    curate.set_defaults(handler=_run_curate)

    protein = domains.add_parser("protein")
    protein_commands = protein.add_subparsers(dest="protein_command", required=True)
    protein_score = protein_commands.add_parser("score")
    protein_score.add_argument("--variants", required=True, type=Path)
    protein_score.add_argument("--sequences", required=True, type=Path)
    protein_score.add_argument("--model", required=True, choices=("esmc-300m", "esmc-600m", "saprot-35m", "saprot-650m"))
    default_model_root = os.environ.get("MAGNETON_MODEL_DIR")
    protein_score.add_argument("--model-dir", required=default_model_root is None, type=Path, default=None if default_model_root is None else Path(default_model_root))
    protein_score.add_argument("--structure-tokens", type=Path)
    protein_score.add_argument("--ignore-missing-structure-tokens", "--skip-missing-structure-tokens", action="store_true")
    protein_score.add_argument("--output", required=True, type=Path)
    protein_score.add_argument("--device", default="cuda")
    protein_score.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    protein_score.add_argument("--max-sequence-length", type=int)
    protein_score.add_argument("--long-sequence-mode", choices=("window", "null"), default="window")
    protein_score.add_argument("--batch-size", type=_positive, default=1)
    protein_score.set_defaults(handler=_run_protein)

    recovery = protein_commands.add_parser("recover-saprot")
    for name in ("mismatches", "translations", "existing-tokens", "output-dir"):
        recovery.add_argument(f"--{name}", required=True, type=Path)
    recovery.add_argument("--merged-output", type=Path)
    recovery.add_argument("--variants", type=Path)
    recovery.add_argument("--foldseek", default="foldseek")
    recovery.add_argument("--threads", type=_positive, default=1)
    recovery.add_argument("--max-requests", type=_positive, default=4)
    recovery.add_argument("--retries", type=int, default=5)
    recovery.add_argument("--backoff-seconds", type=float, default=1.0)
    recovery.add_argument("--timeout-seconds", type=float, default=60.0)
    recovery.set_defaults(handler=_run_recovery)

    dna = domains.add_parser("dna")
    dna_commands = dna.add_subparsers(dest="dna_command", required=True)
    dna_score = dna_commands.add_parser("score")
    dna_score.add_argument("--model", choices=("ntv3-100m-pre", "ntv3-650m-pre"), default="ntv3-100m-pre")
    dna_score.add_argument("--variants", required=True, type=Path)
    dna_score.add_argument("--reference", required=True, type=Path)
    dna_score.add_argument("--genes", type=Path)
    dna_score.add_argument("--model-dir", required=True, type=Path)
    dna_score.add_argument("--model-code-dir", required=True, type=Path)
    dna_score.add_argument("--output-dir", required=True, type=Path)
    dna_score.add_argument("--window-length", type=int, default=8192)
    dna_score.add_argument("--min-variant-margin", type=int, default=1024)
    dna_score.add_argument("--batch-size", type=_positive, default=1)
    dna_score.add_argument("--device", default="cuda")
    dna_score.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    _shards(dna_score)
    dna_score.set_defaults(handler=_run_dna_score)
    dna_collate = dna_commands.add_parser("collate")
    dna_collate.add_argument("--variants", required=True, type=Path)
    dna_collate.add_argument("--input-dir", "--output-dir", dest="input_dir", required=True, type=Path)
    dna_collate.add_argument("--output", required=True, type=Path)
    _shards(dna_collate, collate=True)
    dna_collate.set_defaults(handler=_run_dna_collate)

    rna = domains.add_parser("rna")
    rna_models = rna.add_subparsers(dest="rna_model", required=True)
    for name in ("orthrus", "rinalmo"):
        model = rna_models.add_parser(name)
        commands = model.add_subparsers(dest="rna_command", required=True)
        score = commands.add_parser("score")
        _transcript_inputs(score)
        score.add_argument("--checkpoint" if name == "orthrus" else "--weights", required=True, type=Path)
        score.add_argument("--output-dir", required=True, type=Path)
        score.add_argument("--device", default="cuda")
        score.add_argument("--batch-size", type=_positive, default=32 if name == "orthrus" else 8)
        if name == "rinalmo":
            score.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
        _shards(score)
        score.set_defaults(handler=_run_orthrus_score if name == "orthrus" else _run_rinalmo_score)
        collate = commands.add_parser("collate")
        _transcript_inputs(collate)
        collate.add_argument("--checkpoint" if name == "orthrus" else "--weights", required=True, type=Path)
        collate.add_argument("--input-dir", "--output-dir", dest="input_dir", required=True, type=Path)
        collate.add_argument("--output", required=True, type=Path)
        if name == "rinalmo":
            collate.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
        _shards(collate, collate=True)
        collate.set_defaults(handler=_run_orthrus_collate if name == "orthrus" else _run_rinalmo_collate)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "shard_index") and not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= I < N")
    if args.domain == "protein" and args.protein_command == "score":
        if args.model.startswith("saprot") and args.structure_tokens is None:
            parser.error(f"--structure-tokens is required for {args.model}")
    if args.domain == "protein" and args.protein_command == "recover-saprot":
        if args.retries < 0:
            parser.error("--retries must be nonnegative")
    if args.domain == "dna" and args.dna_command == "score":
        if args.window_length <= 0 or args.window_length % 128:
            parser.error("--window-length must be positive and divisible by 128")
        if args.genes is not None and not 0 <= args.min_variant_margin < args.window_length / 2:
            parser.error("--min-variant-margin must satisfy 0 <= M < W / 2")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    handler: Handler = args.handler
    handler(args)


__all__ = ["build_parser", "main", "parse_args"]
