"""RiNALMo token windowing and inference adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import ComponentScores, PreparedRequest, candidate_sort_key

WEIGHTS_FILENAME = "rinalmo_giga_pretrained.pt"
MAX_MODEL_TOKENS = 1024
SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")
TOKENS = (
    "<cls>", "<pad>", "<eos>", "<unk>", "<mask>",
    "A", "C", "G", "T", "I", "R", "Y", "K", "M", "S", "W",
    "B", "D", "H", "V", "N", "-",
)
TOKEN_IDS = {token: index for index, token in enumerate(TOKENS)}


@dataclass(frozen=True)
class TranscriptWindow:
    start: int
    end: int
    target_token_index: int
    token_count: int
    add_cls: bool
    add_eos: bool


@dataclass(frozen=True)
class RiNALMoScore(ComponentScores):
    transcript_window_start: int
    transcript_window_end: int
    model_token_count: int


def select_transcript_window(length: int, target_index: int) -> TranscriptWindow:
    if length <= 0 or not 0 <= target_index < length:
        raise ValueError(
            "target_index must identify a nucleotide in a non-empty transcript"
        )
    choices: list[tuple[tuple[int, int, int], TranscriptWindow]] = []
    for start in range(
        max(0, target_index - MAX_MODEL_TOKENS + 1), target_index + 1
    ):
        add_cls = start == 0
        end = min(length, start + MAX_MODEL_TOKENS - int(add_cls))
        if end == length and end - start + int(add_cls) + 1 > MAX_MODEL_TOKENS:
            end -= 1
        if end <= target_index:
            continue
        add_eos = end == length
        token_count = end - start + int(add_cls) + int(add_eos)
        window = TranscriptWindow(
            start,
            end,
            int(add_cls) + target_index - start,
            token_count,
            add_cls,
            add_eos,
        )
        left = target_index - start
        right = end - 1 - target_index
        choices.append(((-(end - start), abs(left - right), start), window))
    if not choices:
        raise RuntimeError("no valid RiNALMo transcript window")
    return min(choices, key=lambda item: item[0])[1]


def build_masked_tokens(
    request: PreparedRequest,
) -> tuple[list[int], TranscriptWindow]:
    window = select_transcript_window(
        len(request.annotation.sequence), request.candidate.target_index
    )
    tokens = [TOKEN_IDS["<cls>"]] if window.add_cls else []
    tokens.extend(
        TOKEN_IDS.get(base, TOKEN_IDS["<unk>"])
        for base in request.annotation.sequence[window.start : window.end]
    )
    if window.add_eos:
        tokens.append(TOKEN_IDS["<eos>"])
    if len(tokens) != window.token_count:
        raise RuntimeError("RiNALMo token count does not match crop provenance")
    expected = TOKEN_IDS.get(request.transcript_ref)
    if expected is None or tokens[window.target_token_index] != expected:
        raise RuntimeError(f"masked-token alignment mismatch for {request.key!r}")
    tokens[window.target_token_index] = TOKEN_IDS["<mask>"]
    return tokens, window


def _resolve_weights(weights: Path) -> Path:
    return weights / WEIGHTS_FILENAME if weights.is_dir() else weights


def validate_rinalmo_checkpoint(weights: Path) -> Path:
    path = _resolve_weights(weights)
    if path.name != WEIGHTS_FILENAME or not path.is_file():
        raise ValueError(f"RiNALMo weights are missing: {path}")
    return path


class RiNALMoModel:
    def __init__(
        self,
        weights: Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        weights_path = validate_rinalmo_checkpoint(weights)
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {dtype}")
        import torch
        from rinalmo.config import model_config
        from rinalmo.data.alphabet import Alphabet
        from rinalmo.model.model import RiNALMo

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.dtype = getattr(torch, dtype)
        config = model_config("giga")
        config.model.token_dropout.active = False
        config.model.transformer.use_flash_attn = (
            self.device.type == "cuda" and dtype in ("bfloat16", "float16")
        )
        alphabet = Alphabet(**config["alphabet"])
        if (
            tuple(alphabet.idx_to_tkn) != TOKENS
            or {token: alphabet.get_idx(token) for token in TOKENS} != TOKEN_IDS
        ):
            raise ValueError("installed RiNALMo alphabet/token IDs mismatch")
        dimensions = (
            config.globals.embed_dim,
            config.model.transformer.num_blocks,
            config.model.transformer.num_heads,
        )
        if dimensions != (1280, 33, 20):
            raise ValueError("installed RiNALMo giga configuration mismatch")
        self.model = RiNALMo(config)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device, dtype=self.dtype).eval()

    def score_requests(
        self,
        requests: Sequence[PreparedRequest],
        *,
        batch_size: int,
    ) -> Mapping[tuple[Any, ...], RiNALMoScore]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        torch = self._torch
        prepared = [(request, *build_masked_tokens(request)) for request in requests]
        prepared.sort(
            key=lambda item: (
                len(item[1]), candidate_sort_key(item[0].candidate)
            )
        )
        results: dict[tuple[Any, ...], RiNALMoScore] = {}
        for offset in range(0, len(prepared), batch_size):
            batch = prepared[offset : offset + batch_size]
            max_length = max(len(item[1]) for item in batch)
            token_tensor = torch.full(
                (len(batch), max_length),
                TOKEN_IDS["<pad>"],
                dtype=torch.long,
                device=self.device,
            )
            positions: list[int] = []
            for index, (_, tokens, window) in enumerate(batch):
                token_tensor[index, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.long, device=self.device
                )
                positions.append(window.target_token_index)
            with torch.inference_mode():
                output = self.model(token_tensor)
                logits = output.get("logits") if isinstance(output, Mapping) else None
                expected_shape = (len(batch), max_length, len(TOKENS))
                if logits is None or tuple(logits.shape) != expected_shape:
                    shape = None if logits is None else tuple(logits.shape)
                    raise RuntimeError(f"malformed RiNALMo logits shape: {shape}")
                selected = logits[
                    torch.arange(len(batch), device=self.device),
                    torch.tensor(positions, device=self.device),
                ]
                probabilities = torch.log_softmax(selected.float(), dim=-1).cpu()
            for index, (request, _, window) in enumerate(batch):
                values = probabilities[index]
                score = RiNALMoScore(
                    float(values[TOKEN_IDS[request.transcript_ref]].item()),
                    float(values[TOKEN_IDS[request.transcript_alt]].item()),
                    window.start,
                    window.end,
                    window.token_count,
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
                if request.key in results:
                    raise RuntimeError(f"duplicate model score for {request.key!r}")
                results[request.key] = score
        return results


__all__ = [
    "MAX_MODEL_TOKENS", "RiNALMoModel", "RiNALMoScore", "SUPPORTED_DTYPES",
    "TOKEN_IDS", "TOKENS", "TranscriptWindow", "build_masked_tokens",
    "select_transcript_window", "validate_rinalmo_checkpoint",
]
