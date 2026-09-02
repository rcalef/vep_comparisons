"""Genomic SNV scoring and shard collation."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..tables import read_score_table, shard_path, write_score_table
from .inputs import (
    DnaCandidate,
    GeneSpan,
    IndexedReference,
    ReferenceGenome,
    contig_aliases,
    filter_candidates_by_alt,
    normalize_gene,
    read_dna_candidates,
    read_gene_spans,
    resolve_contig,
    validate_and_resolve_candidates,
)
from .windows import (
    CandidatePlan,
    OrientationScores,
    ScoringContext,
    WindowPlan,
    centered_window,
    complemented,
    context_for,
    deduplicate_contexts,
    extract_window,
    plan_candidates,
    plan_gene_window,
    reverse_complement,
    validate_window_policy,
)

MODEL_NAME = "ntv3-100m-pre"
SUPPORTED_MODELS = frozenset(("ntv3-100m-pre", "ntv3-650m-pre"))
OUTPUT_COLUMNS = (
    "variant", "gene", "feature", "chromosome", "start", "end", "ref", "alt",
    "model", "window_length", "centered_window_start", "centered_window_end",
    "centered_left_pad", "centered_right_pad", "score_forward", "score_reverse",
    "score", "gene_context_status", "gene_window_start", "gene_window_end",
    "gene_score_forward", "gene_score_reverse", "gene_score",
)

InputValidationError = ValueError
ShardCompatibilityError = ValueError


@dataclass(frozen=True)
class DnaScoringSummary:
    output_path: Path
    candidates: int
    unique_variants: int
    contexts: int


class DnaModel(Protocol):
    def score_contexts(
        self,
        contexts: Sequence[ScoringContext],
        *,
        reference: ReferenceGenome,
        batch_size: int,
    ) -> Mapping[tuple[str, int, int, int, str, str], OrientationScores]: ...


ModelFactory = Callable[[Path, Path, str, str], DnaModel]


def shard_variant_ids(
    candidates: Sequence[DnaCandidate],
    references: Sequence[str],
    *,
    num_shards: int,
    shard_index: int,
) -> list[str]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    contig_order = {name: index for index, name in enumerate(references)}
    representatives: dict[str, DnaCandidate] = {}
    for candidate in candidates:
        representatives.setdefault(candidate.variant, candidate)
    ordered = sorted(
        representatives.values(),
        key=lambda item: (
            contig_order[item.chromosome], item.start, item.end,
            item.ref, item.alt, item.variant,
        ),
    )
    start = len(ordered) * shard_index // num_shards
    end = len(ordered) * (shard_index + 1) // num_shards
    return [candidate.variant for candidate in ordered[start:end]]


def _score_fields(scores: OrientationScores) -> tuple[float, float, float]:
    values = (scores.forward, scores.reverse, scores.mean)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"model returned non-finite scores: {values!r}")
    return values


def assemble_rows(
    plans: Sequence[CandidatePlan],
    scores: Mapping[tuple[str, int, int, int, str, str], OrientationScores],
    *,
    window_length: int,
    model_name: str = MODEL_NAME,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in sorted(plans, key=lambda item: item.candidate.input_index):
        candidate = plan.candidate
        forward, reverse, score = _score_fields(
            scores[context_for(candidate, plan.centered).key]
        )
        gene_window = plan.gene_window
        if gene_window is None:
            gene_scores: tuple[float | None, float | None, float | None] = (
                None, None, None,
            )
        elif plan.gene_context_status == "centered_full_gene":
            gene_scores = (forward, reverse, score)
        else:
            gene_scores = _score_fields(scores[context_for(candidate, gene_window).key])
        rows.append(
            {
                "variant": candidate.variant,
                "gene": candidate.gene,
                "feature": candidate.feature,
                "chromosome": candidate.chromosome,
                "start": candidate.start,
                "end": candidate.end,
                "ref": candidate.ref,
                "alt": candidate.alt,
                "model": model_name,
                "window_length": window_length,
                "centered_window_start": plan.centered.start,
                "centered_window_end": plan.centered.end,
                "centered_left_pad": plan.centered.left_pad,
                "centered_right_pad": plan.centered.right_pad,
                "score_forward": forward,
                "score_reverse": reverse,
                "score": score,
                "gene_context_status": plan.gene_context_status,
                "gene_window_start": None if gene_window is None else gene_window.start,
                "gene_window_end": None if gene_window is None else gene_window.end,
                "gene_score_forward": gene_scores[0],
                "gene_score_reverse": gene_scores[1],
                "gene_score": gene_scores[2],
            }
        )
    return rows


def _default_model_factory(
    model_dir: Path, model_code_dir: Path, device: str, dtype: str
) -> DnaModel:
    from .ntv3 import NTv3Model

    return NTv3Model(model_dir, model_code_dir, device=device, dtype=dtype)


def score_dna_variants(
    *,
    variants_path: Path,
    reference_path: Path,
    genes_path: Path | None = None,
    model_dir: Path,
    model_code_dir: Path,
    output_dir: Path,
    model_name: str = MODEL_NAME,
    window_length: int = 8192,
    min_variant_margin: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: str = "float32",
    num_shards: int = 1,
    shard_index: int = 0,
    model_factory: ModelFactory = _default_model_factory,
) -> DnaScoringSummary:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported DNA model: {model_name}")
    validate_window_policy(
        window_length, min_variant_margin if genes_path is not None else 0
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("dtype must be float32 or bfloat16")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")

    with IndexedReference(reference_path) as reference:
        raw_candidates = read_dna_candidates(variants_path)
        retained, _ = filter_candidates_by_alt(raw_candidates)
        candidates = validate_and_resolve_candidates(retained, reference)
        gene_spans = (
            {}
            if genes_path is None
            else read_gene_spans(
                genes_path,
                reference.references,
                required_genes={candidate.gene for candidate in candidates},
            )
        )
        for candidate in candidates:
            gene = gene_spans.get(candidate.gene)
            if gene is not None and candidate.chromosome != gene.chromosome:
                raise ValueError(
                    f"gene_contig_mismatch: {candidate.variant}/{candidate.gene}: "
                    f"variant={candidate.chromosome}, gene={gene.chromosome}"
                )
        selected_ids = set(
            shard_variant_ids(
                candidates,
                reference.references,
                num_shards=num_shards,
                shard_index=shard_index,
            )
        )
        selected = [
            candidate for candidate in candidates if candidate.variant in selected_ids
        ]
        plans = plan_candidates(
            selected,
            gene_spans,
            reference,
            window_length=window_length,
            min_variant_margin=min_variant_margin,
            include_gene_context=genes_path is not None,
        )
        contexts = deduplicate_contexts(plans)
        scores = model_factory(
            model_dir, model_code_dir, device, dtype
        ).score_contexts(contexts, reference=reference, batch_size=batch_size)
        expected_contexts = {context.key for context in contexts}
        if set(scores) != expected_contexts:
            missing = list(expected_contexts - set(scores))
            extra = list(set(scores) - expected_contexts)
            raise RuntimeError(
                f"model score key mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}"
            )
        rows = assemble_rows(
            plans, scores, window_length=window_length, model_name=model_name
        )

    output_path = shard_path(output_dir, num_shards, shard_index)
    write_score_table(output_path, OUTPUT_COLUMNS, rows)
    return DnaScoringSummary(
        output_path, len(selected), len(selected_ids), len(contexts)
    )


def _natural_references(candidates: Sequence[DnaCandidate]) -> list[str]:
    return sorted(
        {candidate.chromosome for candidate in candidates},
        key=lambda chromosome: (
            int(match.group(1))
            if (match := re.fullmatch(r"(?:chr)?(\d+)", chromosome))
            else 10_000,
            chromosome,
        ),
    )


def collate_dna_scores(
    *,
    variants_path: Path,
    output_dir: Path,
    output_path: Path,
    num_shards: int,
) -> Path:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    raw_candidates = read_dna_candidates(variants_path)
    candidates, _ = filter_candidates_by_alt(raw_candidates)
    expected_keys = [candidate.key for candidate in candidates]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("duplicate_variant_gene: input keys are not unique")

    rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
    references = _natural_references(candidates)
    for shard_index in range(num_shards):
        path = shard_path(output_dir, num_shards, shard_index)
        if not path.is_file():
            raise ValueError(f"missing shard {shard_index}: {path}")
        rows = read_score_table(path, OUTPUT_COLUMNS)
        selected_ids = set(
            shard_variant_ids(
                candidates,
                references,
                num_shards=num_shards,
                shard_index=shard_index,
            )
        )
        expected_shard = {
            candidate.key
            for candidate in candidates
            if candidate.variant in selected_ids
        }
        observed = [(row["variant"], normalize_gene(row["gene"])) for row in rows]
        if len(observed) != len(set(observed)):
            raise ValueError(f"duplicate key within shard: {path}")
        if set(observed) != expected_shard:
            missing = list(expected_shard - set(observed))
            extra = list(set(observed) - expected_shard)
            raise ValueError(
                f"shard key coverage mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}"
            )
        for key, row in zip(observed, rows, strict=True):
            if key in rows_by_key:
                raise ValueError(f"duplicate key across shards: {key!r}")
            rows_by_key[key] = row

    missing = [key for key in expected_keys if key not in rows_by_key]
    extra = list(set(rows_by_key) - set(expected_keys))
    if missing or extra:
        raise ValueError(
            f"final key coverage mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}"
        )
    write_score_table(
        output_path, OUTPUT_COLUMNS, [rows_by_key[key] for key in expected_keys]
    )
    return output_path


__all__ = [
    "CandidatePlan", "DnaCandidate", "DnaScoringSummary", "GeneSpan",
    "IndexedReference", "InputValidationError", "OrientationScores",
    "OUTPUT_COLUMNS", "ReferenceGenome", "ScoringContext",
    "ShardCompatibilityError", "WindowPlan", "assemble_rows",
    "centered_window", "collate_dna_scores", "complemented", "contig_aliases",
    "deduplicate_contexts", "extract_window", "filter_candidates_by_alt",
    "plan_candidates", "plan_gene_window", "read_dna_candidates",
    "read_gene_spans", "resolve_contig", "reverse_complement",
    "score_dna_variants", "shard_variant_ids", "validate_and_resolve_candidates",
    "validate_window_policy",
]
