"""Shared transcript-SNV parsing, validation, ordering, and shard helpers.

The implementation currently lives in :mod:`orthrus_scoring` for backwards
compatibility.  This module is the model-neutral import surface used by new RNA
scorers; Orthrus's public names remain unchanged.
"""

from .orthrus_scoring import (
    COMPLEMENT,
    INELIGIBLE_MISSING_CDNA,
    INELIGIBLE_UNSUPPORTED_ALT,
    INELIGIBLE_UNSUPPORTED_REF,
    NUCLEOTIDES,
    ComponentScores,
    InputInventory,
    OrthrusCandidate,
    PreparedRequest,
    TranscriptAnnotation,
    candidate_sort_key,
    expected_range,
    file_identity,
    natural_chromosome_key,
    normalize_stable_id,
    prepare_requests,
    read_orthrus_candidates,
    read_transcript_fasta,
    sha256_file,
    shard_candidates,
    shard_paths,
)

# Neutral aliases.  Legacy Orthrus names intentionally continue to work.
RnaCandidate = OrthrusCandidate
read_rna_candidates = read_orthrus_candidates

__all__ = [
    "COMPLEMENT",
    "NUCLEOTIDES",
    "ComponentScores",
    "InputInventory",
    "PreparedRequest",
    "RnaCandidate",
    "TranscriptAnnotation",
    "candidate_sort_key",
    "expected_range",
    "file_identity",
    "natural_chromosome_key",
    "normalize_stable_id",
    "prepare_requests",
    "read_rna_candidates",
    "read_transcript_fasta",
    "sha256_file",
    "shard_candidates",
    "shard_paths",
]
