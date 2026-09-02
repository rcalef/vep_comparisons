"""Inputs and FASTA output for SaProt structure-token recovery."""

from __future__ import annotations

import bz2
import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from ..inputs import normalize_transcript


@dataclass(frozen=True)
class Translation:
    transcript: str
    transcript_version: str
    sequence: str


@dataclass(frozen=True)
class MismatchCandidate:
    transcript: str
    transcript_version: str
    uniprot_base: str
    sequence: str
    existing_foldseek_length: int | None
    variant_count: int | None
    variant_positions: str

    @property
    def uniprot_accessions(self) -> tuple[str, ...]:
        return tuple(value for value in self.uniprot_base.split(";") if value)


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    if path.suffix == ".bz2":
        return bz2.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _versioned_transcript(value: str) -> str:
    fields = value.split(maxsplit=1)[0].split("|")
    return next((field for field in fields if field.startswith("ENST")), fields[0])


def read_fasta(path: Path) -> dict[str, Translation]:
    records: dict[str, Translation] = {}
    header: str | None = None
    chunks: list[str] = []

    def commit() -> None:
        if header is None:
            return
        versioned = _versioned_transcript(header)
        transcript = normalize_transcript(versioned)
        if transcript in records:
            raise ValueError(f"Duplicate FASTA transcript: {transcript}")
        records[transcript] = Translation(
            transcript,
            versioned,
            "".join(chunks),
        )

    with _open_text(path, "rt") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                header = line[1:]
                chunks = []
            elif header is None:
                raise ValueError(f"Sequence before first FASTA header in {path}")
            else:
                chunks.append(line)
    commit()
    return records


def _field(row: dict[str, str], *names: str) -> str:
    return next(
        (row[name].strip() for name in names if row.get(name, "").strip()),
        "",
    )


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


def read_mismatch_candidates(
    path: Path,
    translations_path: Path,
) -> list[MismatchCandidate]:
    translations = read_fasta(translations_path)
    candidates: dict[str, MismatchCandidate] = {}
    with _open_text(path, "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            transcript = normalize_transcript(
                _field(row, "transcript", "feature", "transcript_version")
            )
            translation = translations[transcript]
            supplied_sequence = _field(row, "gencode_sequence", "sequence")
            if supplied_sequence and supplied_sequence != translation.sequence:
                raise ValueError(
                    f"Mismatch inventory sequence disagrees with FASTA for {transcript}"
                )
            supplied_length = _optional_int(
                _field(row, "gencode_length", "sequence_length")
            )
            if supplied_length is not None and supplied_length != len(translation.sequence):
                raise ValueError(
                    f"Mismatch inventory length disagrees with FASTA for {transcript}"
                )
            accessions: set[str] = set()
            for value in re.split(
                r"[;,]", _field(row, "uniprot_base", "uniprot_id")
            ):
                accession = value.strip()
                if not accession:
                    continue
                suffix = accession.rsplit("-", 1)[-1]
                accessions.add(
                    accession.rsplit("-", 1)[0] if suffix.isdigit() else accession
                )
            candidate = MismatchCandidate(
                transcript=transcript,
                transcript_version=(
                    _field(row, "transcript_version")
                    or translation.transcript_version
                ),
                uniprot_base=";".join(sorted(accessions)),
                sequence=translation.sequence,
                existing_foldseek_length=_optional_int(
                    _field(
                        row,
                        "existing_foldseek_length",
                        "structure_token_length",
                    )
                ),
                variant_count=_optional_int(
                    _field(row, "variant_count", "n_variants")
                ),
                variant_positions=_field(
                    row,
                    "variant_positions",
                    "protein_positions",
                ),
            )
            previous = candidates.get(transcript)
            if previous is not None:
                candidate = MismatchCandidate(
                    transcript=transcript,
                    transcript_version=candidate.transcript_version,
                    uniprot_base=";".join(
                        sorted(
                            set(previous.uniprot_accessions)
                            | set(candidate.uniprot_accessions)
                        )
                    ),
                    sequence=candidate.sequence,
                    existing_foldseek_length=(
                        candidate.existing_foldseek_length
                        if candidate.existing_foldseek_length
                        == previous.existing_foldseek_length
                        else None
                    ),
                    variant_count=candidate.variant_count,
                    variant_positions=candidate.variant_positions,
                )
            candidates[transcript] = candidate
    return sorted(candidates.values(), key=lambda candidate: candidate.transcript)


def write_fasta(path: Path, records: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "wt") as handle:
        for transcript in sorted(records):
            handle.write(f">{transcript}\n")
            sequence = records[transcript]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def merge_tokens(
    *,
    existing_path: Path,
    recovered: dict[str, str],
    translations_path: Path,
    output_path: Path,
) -> dict[str, str]:
    existing = {
        transcript: record.sequence
        for transcript, record in read_fasta(existing_path).items()
    }
    overlap = sorted(set(existing) & set(recovered))
    if overlap:
        raise ValueError(
            "Recovery would replace existing transcript(s): " + ", ".join(overlap)
        )
    translations = read_fasta(translations_path)
    merged = {**existing, **recovered}
    for transcript, tokens in merged.items():
        sequence = translations[transcript].sequence
        if len(tokens) != len(sequence):
            raise ValueError(
                f"{transcript}: translation={len(sequence)}, 3Di={len(tokens)}"
            )
    write_fasta(output_path, merged)
    return merged
