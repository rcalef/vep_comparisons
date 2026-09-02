"""Protein inference-window planning and score aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import exp

from .inputs import Candidate
from .models import PositionRequest


def tile_window_starts(sequence_length: int, window_length: int) -> list[int]:
    if sequence_length <= window_length:
        return [0]
    overlap = (window_length + 1) // 2
    step = max(1, window_length - overlap)
    starts: list[int] = []
    left = 0
    right = sequence_length
    while right - left > window_length:
        starts.extend((left, right - window_length))
        left += step
        right -= step
    if starts[-2] + window_length - starts[-1] < overlap:
        starts.append((sequence_length - window_length) // 2)
    return sorted(set(starts))


def build_position_requests(
    candidates: Sequence[Candidate],
    sequences: Mapping[str, str],
    *,
    structure_tokens: Mapping[str, str] | None,
    max_sequence_length: int,
    long_sequence_mode: str = "window",
) -> list[PositionRequest]:
    seen: set[tuple[str, int, int]] = set()
    starts_by_transcript: dict[str, list[int]] = {}
    requests: list[PositionRequest] = []
    for candidate in candidates:
        sequence = sequences[candidate.transcript]
        if candidate.transcript not in starts_by_transcript:
            if len(sequence) <= max_sequence_length:
                starts_by_transcript[candidate.transcript] = [0]
            elif long_sequence_mode == "window":
                starts_by_transcript[candidate.transcript] = tile_window_starts(
                    len(sequence), max_sequence_length
                )
            else:
                starts_by_transcript[candidate.transcript] = []

        for window_start in starts_by_transcript[candidate.transcript]:
            window_end = window_start + max_sequence_length
            if not window_start < candidate.position <= window_end:
                continue
            key = (candidate.transcript, candidate.position, window_start)
            if key in seen:
                continue
            seen.add(key)
            requests.append(
                PositionRequest(
                    transcript=candidate.transcript,
                    sequence=sequence[window_start:window_end],
                    position=candidate.position,
                    structure_tokens=(
                        structure_tokens[candidate.transcript][window_start:window_end]
                        if structure_tokens is not None
                        else None
                    ),
                    window_start=window_start,
                )
            )
    return requests


def sigmoid_window_weight(
    request: PositionRequest,
    *,
    protein_length: int,
    window_length: int,
) -> float:
    if len(request.sequence) < window_length:
        return 1.0
    taper_length = round(((window_length + 1) // 2) / 2)
    if taper_length == 0:
        return 1.0
    scale = 20.0 * window_length / 1022.0
    local_index = request.local_position - 1
    if request.window_start > 0 and local_index < taper_length:
        return 1.0 / (1.0 + exp(-(local_index - taper_length / 2) / scale))
    if (
        request.window_start + len(request.sequence) < protein_length
        and local_index >= window_length - taper_length
    ):
        taper_index = local_index - (window_length - taper_length)
        return 1.0 / (1.0 + exp((taper_index - taper_length / 2) / scale))
    return 1.0


def aggregate_position_scores(
    requests: Sequence[PositionRequest],
    window_scores: Mapping[tuple[str, int, int], Mapping[str, float]],
    sequences: Mapping[str, str],
    *,
    window_length: int,
) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[PositionRequest]] = {}
    for request in requests:
        grouped.setdefault((request.transcript, request.position), []).append(request)

    result: dict[tuple[str, int], dict[str, float]] = {}
    for key, position_requests in grouped.items():
        weights = [
            sigmoid_window_weight(
                request,
                protein_length=len(sequences[request.transcript]),
                window_length=window_length,
            )
            for request in position_requests
        ]
        total_weight = sum(weights)
        result[key] = {
            amino_acid: sum(
                weight / total_weight * window_scores[request.key][amino_acid]
                for request, weight in zip(position_requests, weights, strict=True)
            )
            for amino_acid in window_scores[position_requests[0].key]
        }
    return result


__all__ = [
    "aggregate_position_scores", "build_position_requests",
    "sigmoid_window_weight", "tile_window_starts",
]
