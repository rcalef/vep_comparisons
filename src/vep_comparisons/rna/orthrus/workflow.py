"""Orthrus transcript-SNV scoring and shard collation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ...tables import shard_path, write_score_table
from ..common import (
    ComponentScores,
    InputInventory,
    OrthrusCandidate,
    PreparedRequest,
    RnaCandidate,
    TranscriptAnnotation,
    candidate_sort_key,
    collate_rna_scores,
    natural_chromosome_key,
    normalize_stable_id,
    prepare_requests,
    read_orthrus_candidates,
    read_transcript_fasta,
    shard_candidates,
)
from .annotation import read_transcript_annotations
from .model import (
    OrthrusModel,
    build_six_track_input,
    validate_orthrus_checkpoint,
)

MODEL_NAME = "orthrus-mlm-6-track"
OUTPUT_COLUMNS = (
    "variant", "gene", "feature", "chromosome", "start", "end",
    "cdna_position", "ref", "alt", "strand", "transcript_ref",
    "transcript_alt", "model", "ref_log_probability", "alt_log_probability",
    "score",
)

InputValidationError = ValueError
ShardCompatibilityError = ValueError


@dataclass(frozen=True)
class OrthrusScoringSummary:
    output_path: Path
    rows: int


class RequestScorer(Protocol):
    def score_requests(
        self, requests: Sequence[PreparedRequest], *, batch_size: int
    ) -> Mapping[tuple[Any, ...], ComponentScores]: ...


ModelFactory = Callable[[Path, str], RequestScorer]


def _default_model_factory(checkpoint: Path, device: str) -> RequestScorer:
    return OrthrusModel(checkpoint, device)


def _score_row(
    request: PreparedRequest, score: ComponentScores
) -> dict[str, Any]:
    candidate = request.candidate
    return {
        "variant": candidate.variant,
        "gene": candidate.gene,
        "feature": candidate.feature,
        "chromosome": candidate.chromosome,
        "start": candidate.start,
        "end": candidate.end,
        "cdna_position": candidate.cdna_position,
        "ref": candidate.ref,
        "alt": candidate.alt,
        "strand": request.annotation.strand,
        "transcript_ref": request.transcript_ref,
        "transcript_alt": request.transcript_alt,
        "model": MODEL_NAME,
        "ref_log_probability": score.ref_log_probability,
        "alt_log_probability": score.alt_log_probability,
        "score": score.score,
    }


def score_orthrus_variants(
    *,
    variants_path: Path,
    transcript_fasta_path: Path,
    gff3_path: Path,
    checkpoint: Path,
    output_dir: Path,
    num_shards: int = 1,
    shard_index: int = 0,
    device: str = "cuda",
    batch_size: int = 32,
    model_factory: ModelFactory = _default_model_factory,
) -> OrthrusScoringSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    validate_orthrus_checkpoint(checkpoint)
    candidates, _ = read_orthrus_candidates(variants_path)
    selected = shard_candidates(
        candidates, num_shards=num_shards, shard_index=shard_index
    )
    annotations = read_transcript_annotations(
        gff3_path,
        transcript_fasta_path,
        {candidate.feature for candidate in selected},
    )
    requests = prepare_requests(selected, annotations)
    scores = model_factory(checkpoint, device).score_requests(
        requests, batch_size=batch_size
    )
    expected = {request.key for request in requests}
    if set(scores) != expected:
        missing = list(expected - set(scores))
        extra = list(set(scores) - expected)
        raise RuntimeError(
            f"model score key mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}"
        )
    rows = [_score_row(request, scores[request.key]) for request in requests]
    output_path = shard_path(output_dir, num_shards, shard_index)
    write_score_table(output_path, OUTPUT_COLUMNS, rows)
    return OrthrusScoringSummary(output_path, len(rows))


def collate_orthrus_scores(
    *,
    variants_path: Path,
    transcript_fasta_path: Path,
    gff3_path: Path,
    checkpoint: Path,
    output_dir: Path,
    output_path: Path,
    num_shards: int,
) -> Path:
    candidates, _ = read_orthrus_candidates(variants_path)
    ordered = sorted(candidates, key=candidate_sort_key)
    annotations = read_transcript_annotations(
        gff3_path,
        transcript_fasta_path,
        {candidate.feature for candidate in ordered},
    )
    prepare_requests(ordered, annotations)
    return collate_rna_scores(
        ordered,
        output_dir=output_dir,
        output_path=output_path,
        num_shards=num_shards,
        columns=OUTPUT_COLUMNS,
    )


__all__ = [
    "ComponentScores", "InputInventory", "InputValidationError",
    "MODEL_NAME", "OUTPUT_COLUMNS", "OrthrusCandidate", "OrthrusModel",
    "OrthrusScoringSummary", "PreparedRequest", "RnaCandidate",
    "ShardCompatibilityError", "TranscriptAnnotation", "build_six_track_input",
    "candidate_sort_key", "collate_orthrus_scores", "natural_chromosome_key",
    "normalize_stable_id", "prepare_requests", "read_orthrus_candidates",
    "read_transcript_annotations", "read_transcript_fasta",
    "score_orthrus_variants", "shard_candidates", "shard_path",
    "validate_orthrus_checkpoint",
]
