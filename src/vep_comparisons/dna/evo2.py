"""Local Evo2 7B sequence likelihoods and pooled SNV representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from .inputs import ReferenceGenome
from .windows import OrientationScores, ScoringContext, extract_window, reverse_complement

if TYPE_CHECKING:
    import torch

EMBEDDING_LAYER = "blocks.28.mlp.l3"
POOL_LENGTH = 128
EMBEDDING_DIMENSION = 4 * 4096


def allele_sequences(reference: ReferenceGenome, context: ScoringContext) -> list[str]:
    """Feature order: reference, RC reference, alternate, RC alternate."""
    ref = extract_window(reference, context)
    assert ref[context.target_index] == context.ref
    alt = ref[:context.target_index] + context.alt + ref[context.target_index + 1:]
    return [ref, reverse_complement(ref), alt, reverse_complement(alt)]


def pool_bounds(context: ScoringContext) -> list[tuple[int, int]]:
    """Reverse the same genomic pooling interval, including its even-length offset."""
    start = context.target_index - POOL_LENGTH // 2
    end = start + POOL_LENGTH
    assert 0 <= start < end <= context.window_length
    forward = (start, end)
    reverse = (context.window_length - end, context.window_length - start)
    return [forward, reverse, forward, reverse]


def sequence_log_likelihoods(logits: torch.Tensor, input_ids: torch.Tensor) -> np.ndarray:
    """Mean next-token log likelihood, without BOS, over the full vocabulary."""
    import torch

    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    observed = logprobs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return observed.mean(dim=-1, dtype=torch.float64).cpu().numpy()


class Evo2Model:
    """Use Vortex's native BF16/FP32 policy; batch_size counts sequences, not loci."""

    def __init__(self, checkpoint: Path, *, device: str = "cuda", dtype: str = "bfloat16") -> None:
        import torch
        from evo2 import Evo2

        # Vortex places its layers across visible GPUs and starts inputs on cuda:0.
        if torch.device(device).type != "cuda" or torch.device(device).index not in (None, 0):
            raise ValueError("Evo2 requires cuda:0; select physical GPUs with CUDA_VISIBLE_DEVICES")
        if dtype != "bfloat16":
            raise ValueError("Evo2 uses Vortex's native bfloat16/float32 precision policy")
        self.evo = Evo2("evo2_7b", local_path=str(checkpoint), use_kernels=False)
        self.evo.model.eval()
        self.device = torch.device("cuda:0")

    def _predict(
        self, sequences: Sequence[str], bounds: Sequence[tuple[int, int]] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        import torch

        inputs = torch.tensor(
            [self.evo.tokenizer.tokenize(seq) for seq in sequences],
            dtype=torch.long, device=self.device,
        )
        with torch.inference_mode():
            outputs, embeddings = self.evo(
                inputs, return_embeddings=bounds is not None,
                layer_names=[EMBEDDING_LAYER] if bounds is not None else None,
            )
            logits = outputs[0]
            likelihoods = sequence_log_likelihoods(logits, inputs)
            pooled = None
            if bounds is not None:
                hidden = embeddings[EMBEDDING_LAYER]
                pooled = torch.stack([
                    hidden[i, start:end].float().mean(dim=0)
                    for i, (start, end) in enumerate(bounds)
                ]).cpu().numpy()
        return likelihoods, pooled

    def _evaluate_contexts(
        self, contexts: Sequence[ScoringContext], *, reference: ReferenceGenome,
        batch_size: int, return_embeddings: bool,
    ) -> tuple[dict[tuple[str, int, int, int, str, str], OrientationScores], np.ndarray | None]:
        assert batch_size > 0
        scores = {}
        vectors = []
        for offset in tqdm(range(0, len(contexts), batch_size), desc="Evo2", unit="batch"):
            batch = contexts[offset:offset + batch_size]
            sequences = [seq for context in batch for seq in allele_sequences(reference, context)]
            bounds = [pair for context in batch for pair in pool_bounds(context)] if return_embeddings else None
            likelihoods, pooled = [], []
            for start in range(0, len(sequences), batch_size):
                values, features = self._predict(
                    sequences[start:start + batch_size],
                    None if bounds is None else bounds[start:start + batch_size],
                )
                likelihoods.append(values)
                if features is not None:
                    pooled.append(features)
            values = np.concatenate(likelihoods).reshape(len(batch), 4)
            for context, (ref, ref_rc, alt, alt_rc) in zip(batch, values, strict=True):
                scores[context.key] = OrientationScores(float(alt - ref), float(alt_rc - ref_rc))
            if return_embeddings:
                vectors.append(np.concatenate(pooled).reshape(len(batch), -1))
        features = np.concatenate(vectors) if vectors else None
        return scores, features

    def score_contexts(
        self, contexts: Sequence[ScoringContext], *, reference: ReferenceGenome, batch_size: int,
    ) -> Mapping[tuple[str, int, int, int, str, str], OrientationScores]:
        scores, _ = self._evaluate_contexts(
            contexts, reference=reference, batch_size=batch_size, return_embeddings=False,
        )
        return scores

    def score_and_embed_contexts(
        self, contexts: Sequence[ScoringContext], *, reference: ReferenceGenome, batch_size: int,
    ) -> tuple[Mapping[tuple[str, int, int, int, str, str], OrientationScores], np.ndarray]:
        scores, embeddings = self._evaluate_contexts(
            contexts, reference=reference, batch_size=batch_size, return_embeddings=True,
        )
        assert embeddings is not None
        return scores, embeddings

    def embed_contexts(
        self, contexts: Sequence[ScoringContext], *, reference: ReferenceGenome, batch_size: int,
    ) -> np.ndarray:
        _, embeddings = self.score_and_embed_contexts(contexts, reference=reference, batch_size=batch_size)
        return embeddings
