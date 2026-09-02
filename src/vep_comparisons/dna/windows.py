"""Genomic inference-window planning and sequence orientation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .inputs import DnaCandidate, GeneSpan, ReferenceGenome

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


@dataclass(frozen=True)
class WindowPlan:
    chromosome: str
    start: int
    end: int
    target_index: int
    left_pad: int
    right_pad: int


@dataclass(frozen=True)
class ScoringContext:
    chromosome: str
    window_start: int
    window_length: int
    target_index: int
    ref: str
    alt: str

    @property
    def key(self) -> tuple[str, int, int, int, str, str]:
        return (
            self.chromosome, self.window_start, self.window_length,
            self.target_index, self.ref, self.alt,
        )


@dataclass(frozen=True)
class OrientationScores:
    forward: float
    reverse: float

    @property
    def mean(self) -> float:
        return (self.forward + self.reverse) / 2.0


@dataclass(frozen=True)
class CandidatePlan:
    candidate: DnaCandidate
    centered: WindowPlan
    gene_context_status: str
    gene_window: WindowPlan | None


def validate_window_policy(window_length: int, min_variant_margin: int) -> None:
    if window_length <= 0 or window_length % 128:
        raise ValueError(
            f"invalid_window_length: {window_length}; must be positive and divisible by 128"
        )
    if not 0 <= min_variant_margin < window_length / 2:
        raise ValueError(
            "invalid_variant_margin: "
            f"{min_variant_margin}; require 0 <= margin < window_length / 2"
        )


def centered_window(
    candidate: DnaCandidate, contig_length: int, window_length: int
) -> WindowPlan:
    target_index = window_length // 2
    start = candidate.start - target_index
    end = start + window_length
    return WindowPlan(
        candidate.chromosome,
        start,
        end,
        target_index,
        max(0, -start),
        max(0, end - contig_length),
    )


def _contains_gene_and_margin(
    start: int,
    *,
    window_length: int,
    variant_start: int,
    gene: GeneSpan,
    margin: int,
) -> bool:
    return (
        start <= gene.start
        and gene.end <= start + window_length
        and start <= variant_start - margin
        and variant_start + 1 + margin <= start + window_length
    )


def plan_gene_window(
    candidate: DnaCandidate,
    centered: WindowPlan,
    gene: GeneSpan | None,
    *,
    contig_length: int,
    window_length: int,
    min_variant_margin: int,
) -> tuple[str, WindowPlan | None]:
    if gene is None:
        return "missing_gene_span", None
    if gene.chromosome != candidate.chromosome:
        raise ValueError(
            f"gene_contig_mismatch: {candidate.variant}/{candidate.gene}: "
            f"variant={candidate.chromosome}, gene={gene.chromosome}"
        )
    if _contains_gene_and_margin(
        centered.start,
        window_length=window_length,
        variant_start=candidate.start,
        gene=gene,
        margin=min_variant_margin,
    ):
        return "centered_full_gene", centered

    lower = max(
        gene.end - window_length,
        candidate.start + 1 + min_variant_margin - window_length,
        0,
    )
    upper = min(
        gene.start,
        candidate.start - min_variant_margin,
        contig_length - window_length,
    )
    if lower > upper:
        return "gene_too_long", None
    start = min(max(centered.start, lower), upper)
    return (
        "shifted_full_gene",
        WindowPlan(
            candidate.chromosome,
            start,
            start + window_length,
            candidate.start - start,
            0,
            0,
        ),
    )


def plan_candidates(
    candidates: Sequence[DnaCandidate],
    gene_spans: Mapping[str, GeneSpan],
    reference: ReferenceGenome,
    *,
    window_length: int,
    min_variant_margin: int,
    include_gene_context: bool = True,
) -> list[CandidatePlan]:
    plans: list[CandidatePlan] = []
    centered_by_variant: dict[str, WindowPlan] = {}
    for candidate in candidates:
        contig_length = reference.get_reference_length(candidate.chromosome)
        centered = centered_by_variant.setdefault(
            candidate.variant,
            centered_window(candidate, contig_length, window_length),
        )
        if include_gene_context:
            status, gene_window = plan_gene_window(
                candidate,
                centered,
                gene_spans.get(candidate.gene),
                contig_length=contig_length,
                window_length=window_length,
                min_variant_margin=min_variant_margin,
            )
        else:
            status, gene_window = "not_requested", None
        plans.append(CandidatePlan(candidate, centered, status, gene_window))
    return plans


def context_for(candidate: DnaCandidate, window: WindowPlan) -> ScoringContext:
    return ScoringContext(
        candidate.chromosome,
        window.start,
        window.end - window.start,
        window.target_index,
        candidate.ref,
        candidate.alt,
    )


def deduplicate_contexts(plans: Sequence[CandidatePlan]) -> list[ScoringContext]:
    contexts: dict[tuple[str, int, int, int, str, str], ScoringContext] = {}
    for plan in plans:
        centered = context_for(plan.candidate, plan.centered)
        contexts.setdefault(centered.key, centered)
        if plan.gene_context_status == "shifted_full_gene":
            assert plan.gene_window is not None
            gene_context = context_for(plan.candidate, plan.gene_window)
            contexts.setdefault(gene_context.key, gene_context)
    return list(contexts.values())


def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(COMPLEMENT)[::-1]


def complemented(allele: str) -> str:
    return allele.translate(COMPLEMENT)


def extract_window(reference: ReferenceGenome, context: ScoringContext) -> str:
    contig_length = reference.get_reference_length(context.chromosome)
    fetch_start = max(0, context.window_start)
    fetch_end = min(contig_length, context.window_start + context.window_length)
    sequence = (
        "N" * max(0, -context.window_start)
        + reference.fetch(context.chromosome, fetch_start, fetch_end).upper()
        + "N" * max(
            0, context.window_start + context.window_length - contig_length
        )
    )
    if len(sequence) != context.window_length:
        raise RuntimeError(
            f"window extraction returned {len(sequence)}, expected {context.window_length}"
        )
    if sequence[context.target_index] != context.ref:
        raise RuntimeError(
            f"window REF mismatch for {context.key}: sequence={sequence[context.target_index]!r}"
        )
    return sequence


__all__ = [
    "CandidatePlan", "OrientationScores", "ScoringContext", "WindowPlan",
    "centered_window", "complemented", "context_for", "deduplicate_contexts",
    "extract_window", "plan_candidates", "plan_gene_window",
    "reverse_complement", "validate_window_policy",
]
