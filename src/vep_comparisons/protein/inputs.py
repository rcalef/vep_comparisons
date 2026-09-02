"""Protein candidate and transcript-sequence inputs."""

from __future__ import annotations

import bz2
import gzip
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import polars as pl
from Bio import SeqIO

logger = getLogger(__name__)

REQUIRED_VARIANT_COLUMNS = (
    "variant", "gene", "feature", "consequence", "protein_position",
    "amino_acids", "biotype",
)


@dataclass(frozen=True)
class Candidate:
    variant: str
    gene: str
    feature: str
    transcript: str
    position: int
    amino_acids: str
    ref: str
    alt: str


def normalize_transcript(value: str) -> str:
    identifiers = value.split(maxsplit=1)[0].split("|")
    identifier = next(
        (item for item in identifiers if item.startswith("ENST")), identifiers[0]
    )
    return identifier.split(".", maxsplit=1)[0]


def read_transcript_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    open_fasta = gzip.open if path.suffix == ".gz" else bz2.open if path.suffix == ".bz2" else open
    with open_fasta(path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            transcript = normalize_transcript(record.name)
            if transcript in records:
                raise ValueError(f"duplicate_transcript: {transcript}")
            records[transcript] = str(record.seq)
    return records


def read_variant_candidates(path: Path) -> list[Candidate]:
    frame = pl.read_csv(
        path, separator="\t", has_header=True, infer_schema=False, null_values="-"
    )
    frame = frame.rename(
        {
            column: (
                "variant"
                if column.lstrip("#").lower() == "uploaded_variation"
                else column.lstrip("#").lower()
            )
            for column in frame.columns
        }
    )
    logger.info("Loaded variants: %d", len(frame))
    frame = (
        frame
        .filter(
            pl.col("biotype") == "protein_coding",
            pl.col("consequence").str.split(",").list.contains("missense_variant"),
        )
        .select(REQUIRED_VARIANT_COLUMNS)
        .with_columns(pl.col("protein_position").cast(pl.Int64))
    )
    logger.info("Protein missense variants: %d", len(frame))

    candidates: list[Candidate] = []
    for row in frame.iter_rows(named=True):
        ref, alt = row["amino_acids"].split("/")
        candidates.append(
            Candidate(
                variant=row["variant"],
                gene=row["gene"],
                feature=row["feature"],
                transcript=normalize_transcript(row["feature"]),
                position=row["protein_position"],
                amino_acids=row["amino_acids"],
                ref=ref,
                alt=alt,
            )
        )
    return candidates


def validate_candidates(
    candidates: Sequence[Candidate],
    sequences: Mapping[str, str],
    *,
    structure_tokens: Mapping[str, str] | None,
    require_structure: bool,
) -> None:
    for transcript in sorted({candidate.transcript for candidate in candidates}):
        sequence = sequences.get(transcript)
        if sequence is None:
            raise ValueError(f"missing_sequence: {transcript}")
        if require_structure and (
            structure_tokens is None or transcript not in structure_tokens
        ):
            raise ValueError(f"missing_structure_tokens: {transcript}")
        if require_structure:
            assert structure_tokens is not None
            tokens = structure_tokens[transcript]
            if len(tokens) != len(sequence):
                raise ValueError(
                    "sequence_structure_length_mismatch: "
                    f"{transcript}: sequence={len(sequence)}, 3Di={len(tokens)}"
                )

    for candidate in candidates:
        sequence = sequences[candidate.transcript]
        if not 1 <= candidate.position <= len(sequence):
            raise ValueError(
                "position_out_of_range: "
                f"variant={candidate.variant}, transcript={candidate.transcript}, "
                f"position={candidate.position}, length={len(sequence)}"
            )
        observed = sequence[candidate.position - 1].upper()
        if observed != candidate.ref:
            raise ValueError(
                "reference_mismatch: "
                f"variant={candidate.variant}, transcript={candidate.transcript}, "
                f"position={candidate.position}, VEP={candidate.ref}, FASTA={observed!r}"
            )


__all__ = [
    "Candidate", "normalize_transcript", "read_transcript_fasta",
    "read_variant_candidates", "validate_candidates",
]
