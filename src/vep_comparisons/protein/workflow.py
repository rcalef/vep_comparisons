"""Zero-shot protein variant scoring workflow."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import polars as pl

from .inputs import (
    Candidate,
    normalize_transcript,
    read_transcript_fasta,
    read_variant_candidates,
    validate_candidates,
)
from .models import MODEL_REGISTRY, ModelSpec, VariantModel, load_variant_model
from .windowing import (
    aggregate_position_scores,
    build_position_requests,
    sigmoid_window_weight,
    tile_window_starts,
)

logger = getLogger(__name__)
InputValidationError = ValueError

OUTPUT_COLUMNS = (
    "variant", "gene", "feature", "protein_position", "amino_acids", "model", "score",
)


@dataclass(frozen=True)
class ScoringSummary:
    model: str
    candidates: int
    unique_positions: int
    window_requests: int
    scored: int
    null: int
    output_path: Path


ModelFactory = Callable[[ModelSpec, Path, str, str], VariantModel]


def _default_model_factory(
    spec: ModelSpec, model_root: Path, device: str, dtype: str
) -> VariantModel:
    return load_variant_model(spec, model_root, device=device, dtype=dtype)


def score_protein_variants(
    *,
    variants_path: Path,
    sequences_path: Path,
    model_name: str,
    model_root: Path,
    output: Path,
    structure_tokens_path: Path | None = None,
    ignore_missing_structure_tokens: bool = False,
    device: str = "cuda",
    dtype: str = "float32",
    max_sequence_length: int | None = None,
    long_sequence_mode: str = "window",
    batch_size: int = 1,
    model_factory: ModelFactory = _default_model_factory,
) -> ScoringSummary:
    spec = MODEL_REGISTRY[model_name]
    limit = spec.capacity if max_sequence_length is None else max_sequence_length
    if not 1 <= limit <= spec.capacity:
        raise ValueError(
            f"invalid_max_sequence_length: requested={limit}, "
            f"supported=1..{spec.capacity} for {model_name}"
        )
    if long_sequence_mode not in {"window", "null"}:
        raise ValueError(f"invalid_long_sequence_mode: {long_sequence_mode}")
    if batch_size < 1:
        raise ValueError(f"invalid_batch_size: {batch_size}")

    require_structure = spec.family == "saprot"
    if ignore_missing_structure_tokens and not require_structure:
        raise ValueError(
            "invalid_ignore_missing_structure_tokens: "
            f"{model_name} does not use structure tokens"
        )
    if require_structure and structure_tokens_path is None:
        raise ValueError(
            f"missing_structure_tokens_argument: --structure-tokens is required for {model_name}"
        )

    candidates = read_variant_candidates(variants_path)
    sequences = read_transcript_fasta(sequences_path)
    structures = (
        read_transcript_fasta(structure_tokens_path)
        if structure_tokens_path is not None and require_structure
        else None
    )
    if ignore_missing_structure_tokens:
        assert structures is not None
        ignored = {
            candidate.transcript
            for candidate in candidates
            if candidate.transcript in sequences
            and candidate.transcript not in structures
        }
        if ignored:
            before = len(candidates)
            candidates = [
                candidate
                for candidate in candidates
                if candidate.transcript not in ignored
            ]
            logger.warning(
                "Ignored %d candidate(s) across %d transcript(s) with no structure tokens",
                before - len(candidates),
                len(ignored),
            )

    validate_candidates(
        candidates,
        sequences,
        structure_tokens=structures,
        require_structure=require_structure,
    )
    requests = build_position_requests(
        candidates,
        sequences,
        structure_tokens=structures,
        max_sequence_length=limit,
        long_sequence_mode=long_sequence_mode,
    )
    unique_positions = len({(request.transcript, request.position) for request in requests})
    print(
        f"model={model_name} candidates={len(candidates)} "
        f"unique_positions={unique_positions} window_requests={len(requests)} "
        f"max_length={limit} long_sequence_mode={long_sequence_mode} "
        f"batch_size={batch_size}",
        file=sys.stderr,
    )
    print(
        f"validation=passed transcripts={len({candidate.transcript for candidate in candidates})}",
        file=sys.stderr,
    )

    window_scores = model_factory(spec, model_root, device, dtype).score_positions(
        requests, batch_size=batch_size
    )
    expected_request_keys = {request.key for request in requests}
    if set(window_scores) != expected_request_keys:
        missing = list(expected_request_keys - set(window_scores))
        extra = list(set(window_scores) - expected_request_keys)
        raise RuntimeError(
            f"model score key mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}"
        )
    position_scores = aggregate_position_scores(
        requests, window_scores, sequences, window_length=limit
    )

    scores = [
        None
        if (candidate.transcript, candidate.position) not in position_scores
        else position_scores[(candidate.transcript, candidate.position)][candidate.alt]
        - position_scores[(candidate.transcript, candidate.position)][candidate.ref]
        for candidate in candidates
    ]
    frame = pl.DataFrame(
        {
            "variant": [candidate.variant for candidate in candidates],
            "gene": [candidate.gene for candidate in candidates],
            "feature": [candidate.feature for candidate in candidates],
            "protein_position": [candidate.position for candidate in candidates],
            "amino_acids": [candidate.amino_acids for candidate in candidates],
            "model": [model_name] * len(candidates),
            "score": pl.Series(scores, dtype=pl.Float64),
        }
    ).select(OUTPUT_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(output, separator="\t", compression="gzip")

    scored = sum(score is not None for score in scores)
    summary = ScoringSummary(
        model_name,
        len(candidates),
        unique_positions,
        len(requests),
        scored,
        len(scores) - scored,
        output,
    )
    print(f"scored={summary.scored} null={summary.null} output={output}", file=sys.stderr)
    return summary


__all__ = [
    "Candidate",
    "InputValidationError",
    "ScoringSummary",
    "aggregate_position_scores",
    "build_position_requests",
    "normalize_transcript",
    "read_transcript_fasta",
    "read_variant_candidates",
    "score_protein_variants",
    "sigmoid_window_weight",
    "tile_window_starts",
]
