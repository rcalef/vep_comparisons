"""SaProt structure-token recovery workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..inputs import read_variant_candidates, validate_candidates
from .alphafold import AlphaFoldClient, AlphaFoldModel, select_exact_model
from .foldseek import recover_tokens
from .inputs import (
    MismatchCandidate,
    merge_tokens,
    read_fasta,
    read_mismatch_candidates,
    write_fasta,
)

RecoveryError = ValueError


@dataclass(frozen=True)
class RecoverySummary:
    candidates: int
    recovered: int
    unresolved: int
    recovered_fasta: Path
    merged_fasta: Path


def recover_saprot_structures(
    *,
    mismatches_path: Path,
    translations_path: Path,
    existing_tokens_path: Path,
    output_dir: Path,
    foldseek: str = "foldseek",
    threads: int = 1,
    merged_output: Path | None = None,
    variants_path: Path | None = None,
    retries: int = 5,
    backoff_seconds: float = 1.0,
    max_requests: int = 4,
    timeout_seconds: float = 60.0,
) -> RecoverySummary:
    if threads < 1 or max_requests < 1 or retries < 0:
        raise ValueError("threads/max_requests must be positive and retries nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_mismatch_candidates(mismatches_path, translations_path)
    client = AlphaFoldClient(
        pdb_dir=output_dir / "pdb",
        retries=retries,
        backoff_seconds=backoff_seconds,
        max_requests=max_requests,
        timeout_seconds=timeout_seconds,
    )
    selected = client.discover(candidates)
    downloaded = client.download(selected)
    recovered = recover_tokens(
        downloaded,
        pdb_dir=client.pdb_dir,
        descriptor_path=output_dir / "foldseek_descriptors.tsv",
        foldseek=foldseek,
        threads=threads,
    )

    if variants_path is not None and recovered:
        variants = [
            candidate
            for candidate in read_variant_candidates(variants_path)
            if candidate.transcript in recovered
        ]
        sequences = {
            transcript: translation.sequence
            for transcript, translation in read_fasta(translations_path).items()
        }
        validate_candidates(
            variants,
            sequences,
            structure_tokens=recovered,
            require_structure=True,
        )

    recovered_path = output_dir / "recovered_foldseek_tokens.fa.bz2"
    write_fasta(recovered_path, recovered)
    merged_path = merged_output or output_dir / "merged_foldseek_tokens.fa.bz2"
    merge_tokens(
        existing_path=existing_tokens_path,
        recovered=recovered,
        translations_path=translations_path,
        output_path=merged_path,
    )
    return RecoverySummary(
        candidates=len(candidates),
        recovered=len(recovered),
        unresolved=len(candidates) - len(recovered),
        recovered_fasta=recovered_path,
        merged_fasta=merged_path,
    )


__all__ = [
    "AlphaFoldModel",
    "MismatchCandidate",
    "RecoveryError",
    "RecoverySummary",
    "merge_tokens",
    "read_fasta",
    "read_mismatch_candidates",
    "recover_saprot_structures",
    "select_exact_model",
]
