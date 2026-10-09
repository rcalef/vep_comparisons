"""Orthrus model input construction and inference adapter."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..common import ComponentScores, PreparedRequest, TranscriptAnnotation, candidate_sort_key

NUCLEOTIDES = "ACGT"
NUCLEOTIDE_INDEX = {base: index for index, base in enumerate(NUCLEOTIDES)}


def build_six_track_input(
    annotation: TranscriptAnnotation,
    target_index: int,
    *,
    prefix_only: bool = True,
    mask_target: bool = True,
) -> list[list[float]]:
    end = target_index + 1 if prefix_only else len(annotation.sequence)
    result: list[list[float]] = []
    for index, base in enumerate(annotation.sequence[:end]):
        row = [0.0] * 6
        if (not mask_target or index != target_index) and base in NUCLEOTIDE_INDEX:
            row[NUCLEOTIDE_INDEX[base]] = 1.0
        row[4] = float(annotation.cds_track[index])
        row[5] = float(annotation.splice_track[index])
        result.append(row)
    return result


def validate_orthrus_checkpoint(checkpoint: Path) -> None:
    config = json.loads((checkpoint / "config.json").read_text())
    if (
        config.get("n_tracks") != 6
        or config.get("has_mlm_head") is not True
        or config.get("mlm_head_dim") != 4
    ):
        raise ValueError(
            "Orthrus checkpoint must have six tracks and a four-class MLM head"
        )


class OrthrusModel:
    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        validate_orthrus_checkpoint(checkpoint)
        import torch
        from transformers import AutoModel

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.model = AutoModel.from_pretrained(
            str(checkpoint.resolve()),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(self.device, dtype=torch.float32).eval()
        config = self.model.config
        if config.n_tracks != 6 or not config.has_mlm_head or config.mlm_head_dim != 4:
            raise ValueError(
                "Orthrus checkpoint must have six tracks and a four-class MLM head"
            )

    def embed_requests(
        self, requests: Sequence[PreparedRequest], *, batch_size: int,
    ) -> Mapping[tuple[Any, ...], np.ndarray]:
        """Unmasked transcript prefixes; forward returns norm_f before the MLM head."""
        assert batch_size > 0
        torch = self._torch
        grouped: dict[tuple[str, int], list[PreparedRequest]] = {}
        for request in requests:
            key = (request.candidate.feature, request.candidate.target_index)
            grouped.setdefault(key, []).append(request)
        groups = sorted(grouped.values(), key=lambda group: group[0].candidate.target_index)
        result = {}
        for offset in range(0, len(groups), batch_size):
            batch = groups[offset : offset + batch_size]
            lengths = torch.tensor(
                [group[0].candidate.target_index + 1 for group in batch], device=self.device,
            )
            inputs = torch.zeros((len(batch), int(lengths.max()), 6), device=self.device)
            for row, group in enumerate(batch):
                request = group[0]
                array = build_six_track_input(
                    request.annotation, request.candidate.target_index, mask_target=False,
                )
                inputs[row, :len(array)] = torch.tensor(array, device=self.device)
            with torch.inference_mode():
                # The pinned causal Mamba forward ignores lengths; right padding
                # cannot affect earlier positions. Retain the prefix policy.
                hidden = self.model(inputs, lengths=lengths, channel_last=True)
                vectors = hidden[torch.arange(len(batch), device=self.device), lengths - 1].float().cpu().numpy()
            for group, vector in zip(batch, vectors, strict=True):
                for request in group:
                    result[request.key] = vector
        return result

    def score_requests(
        self,
        requests: Sequence[PreparedRequest],
        *,
        batch_size: int,
        prefix_only: bool = True,
    ) -> Mapping[tuple[Any, ...], ComponentScores]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        torch = self._torch
        ordered = sorted(
            requests,
            key=lambda request: (
                request.candidate.target_index + 1
                if prefix_only
                else len(request.annotation.sequence),
                candidate_sort_key(request.candidate),
            ),
        )
        results: dict[tuple[Any, ...], ComponentScores] = {}
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset : offset + batch_size]
            arrays = [
                build_six_track_input(
                    request.annotation,
                    request.candidate.target_index,
                    prefix_only=prefix_only,
                )
                for request in batch
            ]
            lengths = torch.tensor(
                [len(array) for array in arrays],
                dtype=torch.long,
                device=self.device,
            )
            max_length = int(lengths.max().item())
            inputs = torch.zeros(
                (len(batch), max_length, 6),
                dtype=torch.float32,
                device=self.device,
            )
            for index, array in enumerate(arrays):
                inputs[index, : len(array)] = torch.tensor(
                    array, dtype=torch.float32, device=self.device
                )
            with torch.inference_mode():
                logits = self.model.predict_tokens(
                    inputs, lengths, channel_last=True
                )
                positions = torch.tensor(
                    [request.candidate.target_index for request in batch],
                    dtype=torch.long,
                    device=self.device,
                )
                selected = logits[
                    torch.arange(len(batch), device=self.device), positions
                ]
                probabilities = torch.log_softmax(selected.float(), dim=-1).cpu()
            for index, request in enumerate(batch):
                values = probabilities[index]
                score = ComponentScores(
                    float(values[NUCLEOTIDE_INDEX[request.transcript_ref]].item()),
                    float(values[NUCLEOTIDE_INDEX[request.transcript_alt]].item()),
                )
                if not all(
                    math.isfinite(value)
                    for value in (
                        score.ref_log_probability,
                        score.alt_log_probability,
                        score.score,
                    )
                ):
                    raise RuntimeError(
                        f"model returned non-finite scores for {request.key!r}"
                    )
                results[request.key] = score
        return results


__all__ = [
    "OrthrusModel", "build_six_track_input", "validate_orthrus_checkpoint",
]
