"""Model adapters for masked-marginal protein variant scoring.

Heavy inference dependencies are imported lazily so input validation and the
CPU-only test suite do not initialize CUDA or load a checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
FOLDSEEK_ALPHABET = "pynwrqhgdlvtmfsaeikc#"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    checkpoint_directory: str
    capacity: int
    size: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "esmc-300m": ModelSpec(
        name="esmc-300m",
        family="esmc",
        checkpoint_directory="esmc-300m-2024-12",
        capacity=2046,
        size="300m",
    ),
    "esmc-600m": ModelSpec(
        name="esmc-600m",
        family="esmc",
        checkpoint_directory="esmc-600m-2024-12",
        capacity=2046,
        size="600m",
    ),
    "saprot-35m": ModelSpec(
        name="saprot-35m",
        family="saprot",
        checkpoint_directory="SaProt_35M_AF2",
        capacity=1024,
        size="35m",
    ),
    "saprot-650m": ModelSpec(
        name="saprot-650m",
        family="saprot",
        checkpoint_directory="SaProt_650M_AF2",
        capacity=1024,
        size="650m",
    ),
}


@dataclass(frozen=True)
class PositionRequest:
    """One full-protein masked position to score."""

    transcript: str
    sequence: str
    position: int  # one based
    structure_tokens: str | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.transcript, self.position)

    @property
    def token_count(self) -> int:
        # ESM-C and SaProt both add CLS and EOS.
        return len(self.sequence) + 2


class VariantModel(Protocol):
    """Minimal model-neutral inference interface."""

    def score_positions(
        self,
        requests: Sequence[PositionRequest],
        *,
        max_tokens_per_batch: int,
    ) -> Mapping[tuple[str, int], Mapping[str, float]]:
        """Return an amino-acid log score for each requested position."""


def pack_token_batches(
    requests: Sequence[PositionRequest], max_tokens: int
) -> list[list[PositionRequest]]:
    """Greedily preserve order while respecting a padded-token budget."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    batches: list[list[PositionRequest]] = []
    current: list[PositionRequest] = []
    current_max = 0
    for request in requests:
        if request.token_count > max_tokens:
            raise ValueError(
                f"{request.transcript} requires {request.token_count} tokens, "
                f"above the per-batch budget of {max_tokens}"
            )
        proposed_max = max(current_max, request.token_count)
        proposed_tokens = proposed_max * (len(current) + 1)
        if current and proposed_tokens > max_tokens:
            batches.append(current)
            current = []
            current_max = 0
        current.append(request)
        current_max = max(current_max, request.token_count)

    if current:
        batches.append(current)
    return batches


def log_odds_from_logits(
    logits: Sequence[float], *, ref_token: int, alt_token: int
) -> float:
    """Compute log P(ALT) - log P(REF); the normalizer cancels."""

    return float(logits[alt_token]) - float(logits[ref_token])


def marginalized_log_odds_from_logits(
    logits: Sequence[float],
    *,
    ref_tokens: Sequence[int],
    alt_tokens: Sequence[int],
) -> float:
    """Compute SaProt ALT-vs-REF odds after structural-token marginalization."""

    import torch

    values = torch.as_tensor(logits, dtype=torch.float64)
    ref = torch.logsumexp(values[list(ref_tokens)], dim=0)
    alt = torch.logsumexp(values[list(alt_tokens)], dim=0)
    return float((alt - ref).item())


def _torch_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


class ESMCVariantModel:
    """Local ESM-C checkpoint adapter."""

    def __init__(
        self, spec: ModelSpec, model_root: Path, *, device: str, dtype: str
    ) -> None:
        import torch
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers

        checkpoint_dir = model_root / spec.checkpoint_directory
        weights = (
            checkpoint_dir
            / "data"
            / "weights"
            / f"esmc_{spec.size}_2024_12_v0.pth"
        )
        if not weights.is_file():
            raise FileNotFoundError(f"ESM-C checkpoint not found: {weights}")

        tokenizer = get_esmc_model_tokenizers()
        dimensions = {
            "300m": (960, 15, 30),
            "600m": (1152, 18, 36),
        }
        d_model, n_heads, n_layers = dimensions[spec.size]
        model = ESMC(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tokenizer=tokenizer,
            use_flash_attn=False,
        )
        state_dict = torch.load(weights, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        del state_dict

        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=_torch_dtype(dtype)).eval()
        self.tokenizer = tokenizer
        vocabulary = tokenizer.get_vocab()
        self.aa_token_ids = {aa: vocabulary[aa] for aa in STANDARD_AMINO_ACIDS}

    def score_positions(
        self,
        requests: Sequence[PositionRequest],
        *,
        max_tokens_per_batch: int,
    ) -> Mapping[tuple[str, int], Mapping[str, float]]:
        import torch
        from tqdm import tqdm

        batches = pack_token_batches(requests, max_tokens_per_batch)
        result: dict[tuple[str, int], dict[str, float]] = {}
        for batch in tqdm(batches, desc="Scoring masked positions", unit="batch"):
            tokens = self.tokenizer(
                [request.sequence for request in batch],
                return_tensors="pt",
                padding=True,
            )["input_ids"]
            for row, request in enumerate(batch):
                tokens[row, request.position] = self.tokenizer.mask_token_id
            tokens = tokens.to(self.device)

            with torch.inference_mode():
                logits = self.model(tokens).sequence_logits
            for row, request in enumerate(batch):
                position_logits = logits[row, request.position].float()
                result[request.key] = {
                    aa: float(position_logits[token_id].item())
                    for aa, token_id in self.aa_token_ids.items()
                }
        return result


class SaProtVariantModel:
    """Local Hugging Face SaProt checkpoint adapter."""

    def __init__(
        self, spec: ModelSpec, model_root: Path, *, device: str, dtype: str
    ) -> None:
        import torch
        from transformers import EsmForMaskedLM, EsmTokenizer

        checkpoint_dir = model_root / spec.checkpoint_directory
        if not (checkpoint_dir / "pytorch_model.bin").is_file():
            raise FileNotFoundError(f"SaProt checkpoint not found: {checkpoint_dir}")

        self.tokenizer = EsmTokenizer.from_pretrained(
            checkpoint_dir, local_files_only=True
        )
        self.device = torch.device(device)
        self.model = EsmForMaskedLM.from_pretrained(
            checkpoint_dir, local_files_only=True
        ).to(device=self.device, dtype=_torch_dtype(dtype)).eval()
        vocabulary = self.tokenizer.get_vocab()
        self.aa_token_ids = {
            aa: torch.tensor(
                [vocabulary[f"{aa}{structure}"] for structure in FOLDSEEK_ALPHABET],
                device=self.device,
            )
            for aa in STANDARD_AMINO_ACIDS
        }

    def _combined_sequence(self, request: PositionRequest) -> str:
        if request.structure_tokens is None:
            raise ValueError("SaProt requires structure tokens")
        parts = []
        for index, (aa, structure) in enumerate(
            zip(request.sequence, request.structure_tokens, strict=True), start=1
        ):
            parts.append(f"#{structure}" if index == request.position else f"{aa}{structure}")
        return "".join(parts)

    def score_positions(
        self,
        requests: Sequence[PositionRequest],
        *,
        max_tokens_per_batch: int,
    ) -> Mapping[tuple[str, int], Mapping[str, float]]:
        import torch
        from tqdm import tqdm

        batches = pack_token_batches(requests, max_tokens_per_batch)
        result: dict[tuple[str, int], dict[str, float]] = {}
        for batch in tqdm(batches, desc="Scoring masked positions", unit="batch"):
            combined = [self._combined_sequence(request) for request in batch]
            tokenized = self.tokenizer(combined, return_tensors="pt", padding=True)
            inputs = {name: tensor.to(self.device) for name, tensor in tokenized.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits

            for row, request in enumerate(batch):
                position_logits = logits[row, request.position].float()
                result[request.key] = {
                    aa: float(
                        torch.logsumexp(
                            position_logits[token_ids], dim=0
                        ).item()
                    )
                    for aa, token_ids in self.aa_token_ids.items()
                }
        return result


def load_variant_model(
    spec: ModelSpec,
    model_root: Path,
    *,
    device: str,
    dtype: str,
) -> VariantModel:
    """Construct the selected local model after all inputs have validated."""

    if spec.family == "esmc":
        return ESMCVariantModel(spec, model_root, device=device, dtype=dtype)
    if spec.family == "saprot":
        return SaProtVariantModel(spec, model_root, device=device, dtype=dtype)
    raise ValueError(f"Unknown model family: {spec.family}")
