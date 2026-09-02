"""RiNALMo transcript-SNV scoring and shard collation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ...tables import shard_path, write_score_table
from ..common import (
    OrthrusCandidate,
    PreparedRequest,
    RnaCandidate,
    TranscriptAnnotation,
    candidate_sort_key,
    collate_rna_scores,
    prepare_requests,
    read_rna_candidates,
    shard_candidates,
)
from .annotation import read_rinalmo_annotations
from .model import (
    MAX_MODEL_TOKENS,
    SUPPORTED_DTYPES,
    TOKEN_IDS,
    TOKENS,
    RiNALMoModel,
    RiNALMoScore,
    TranscriptWindow,
    build_masked_tokens,
    select_transcript_window,
    validate_rinalmo_checkpoint,
)

MODEL_NAME = "rinalmo-giga"
OUTPUT_COLUMNS = (
    "variant", "gene", "feature", "chromosome", "start", "end",
    "cdna_position", "ref", "alt", "strand", "transcript_ref",
    "transcript_alt", "model", "ref_log_probability", "alt_log_probability",
    "score", "transcript_window_start", "transcript_window_end",
    "model_token_count",
)

InputValidationError = ValueError
ShardCompatibilityError = ValueError


@dataclass(frozen=True)
class RiNALMoScoringSummary:
    output_path: Path
    rows: int


class RequestScorer(Protocol):
    def score_requests(
        self, requests: Sequence[PreparedRequest], *, batch_size: int
    ) -> Mapping[tuple[Any, ...], RiNALMoScore]: ...


ModelFactory = Callable[[Path, str, str], RequestScorer]


def _default_model_factory(
    weights: Path, device: str, dtype: str
) -> RequestScorer:
    return RiNALMoModel(weights, device, dtype)


def _score_row(request: PreparedRequest, score: RiNALMoScore) -> dict[str, Any]:
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
        "transcript_window_start": score.transcript_window_start,
        "transcript_window_end": score.transcript_window_end,
        "model_token_count": score.model_token_count,
    }


def score_rinalmo_variants(
    *,
    variants_path: Path,
    transcript_fasta_path: Path,
    gff3_path: Path,
    weights: Path,
    output_dir: Path,
    num_shards: int = 1,
    shard_index: int = 0,
    device: str = "cuda",
    dtype: str = "bfloat16",
    batch_size: int = 8,
    model_factory: ModelFactory = _default_model_factory,
) -> RiNALMoScoringSummary:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    validate_rinalmo_checkpoint(weights)
    candidates, _ = read_rna_candidates(variants_path)
    selected = shard_candidates(
        candidates, num_shards=num_shards, shard_index=shard_index
    )
    annotations = read_rinalmo_annotations(
        gff3_path,
        transcript_fasta_path,
        {candidate.feature for candidate in selected},
    )
    requests = prepare_requests(selected, annotations)
    scores = model_factory(weights, device, dtype).score_requests(
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
    return RiNALMoScoringSummary(output_path, len(rows))


def collate_rinalmo_scores(
    *,
    variants_path: Path,
    transcript_fasta_path: Path,
    gff3_path: Path,
    weights: Path,
    output_dir: Path,
    output_path: Path,
    num_shards: int,
    dtype: str = "bfloat16",
) -> Path:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    candidates, _ = read_rna_candidates(variants_path)
    ordered = sorted(candidates, key=candidate_sort_key)
    annotations = read_rinalmo_annotations(
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
    "InputValidationError", "MAX_MODEL_TOKENS", "OUTPUT_COLUMNS",
    "OrthrusCandidate", "PreparedRequest", "RiNALMoModel", "RiNALMoScore",
    "RiNALMoScoringSummary", "SUPPORTED_DTYPES", "ShardCompatibilityError",
    "TOKEN_IDS", "TOKENS", "TranscriptAnnotation", "TranscriptWindow",
    "build_masked_tokens", "collate_rinalmo_scores", "read_rinalmo_annotations",
    "score_rinalmo_variants", "select_transcript_window",
    "validate_rinalmo_checkpoint",
]
