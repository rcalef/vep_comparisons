"""Shared transcript-SNV parsing, preparation, sharding, and collation."""

from __future__ import annotations

import csv
import gzip
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import unquote

from ..tables import read_score_table, shard_path, write_score_table

NUCLEOTIDES = "ACGT"
COMPLEMENT = str.maketrans("ACGT", "TGCA")
REQUIRED_COLUMNS = (
    "variant", "gene", "feature", "chromosome", "start", "end",
    "cdna_position", "ref", "alt",
)
_MISSING = frozenset(("", "-", ".", "NA", "NaN", "nan", "null", "None"))
_AUTOSOME_RE = re.compile(r"^(?:chr)?([1-9]|1[0-9]|2[0-2])$")


@dataclass(frozen=True)
class RnaCandidate:
    variant: str
    gene: str
    feature: str
    chromosome: str
    start: int
    end: int
    cdna_position: int
    ref: str
    alt: str
    input_index: int

    @property
    def target_index(self) -> int:
        return self.cdna_position - 1

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.variant, self.gene, self.feature, self.chromosome, self.start,
            self.end, self.cdna_position, self.ref, self.alt,
        )


@dataclass(frozen=True)
class InputInventory:
    input_rows: int
    input_unique_variants: int
    eligible_rows: int
    eligible_unique_variants: int
    ineligible_rows: int
    ineligible_unique_variants: int
    ineligible_counts: Mapping[str, int]


@dataclass(frozen=True)
class TranscriptAnnotation:
    transcript: str
    gene: str | None
    chromosome: str
    strand: str
    sequence: str
    exons: tuple[tuple[int, int], ...]
    cds_track: tuple[int, ...]
    splice_track: tuple[int, ...]

    def genomic_position(self, transcript_index: int) -> int:
        remaining = transcript_index
        for start, end in self.exons:
            length = end - start
            if remaining < length:
                return start + remaining if self.strand == "+" else end - 1 - remaining
            remaining -= length
        raise IndexError(transcript_index)


@dataclass(frozen=True)
class PreparedRequest:
    candidate: RnaCandidate
    annotation: TranscriptAnnotation
    transcript_ref: str
    transcript_alt: str

    @property
    def key(self) -> tuple[Any, ...]:
        return self.candidate.key


@dataclass(frozen=True)
class ComponentScores:
    ref_log_probability: float
    alt_log_probability: float

    @property
    def score(self) -> float:
        return self.alt_log_probability - self.ref_log_probability


def open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")


def normalize_header(name: str) -> str:
    value = name.lstrip("#").strip().lower()
    return "variant" if value == "uploaded_variation" else value


def normalize_stable_id(value: str) -> str:
    return value.strip().split(".", 1)[0]


def natural_chromosome_key(chromosome: str) -> tuple[int, int | str, str]:
    match = _AUTOSOME_RE.fullmatch(chromosome)
    if match:
        return (0, int(match.group(1)), chromosome)
    bare = chromosome[3:] if chromosome.lower().startswith("chr") else chromosome
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    upper = bare.upper()
    return (1, special[upper], chromosome) if upper in special else (2, upper, chromosome)


def candidate_sort_key(candidate: RnaCandidate) -> tuple[Any, ...]:
    return (
        natural_chromosome_key(candidate.chromosome), candidate.start, candidate.end,
        candidate.ref, candidate.alt, candidate.feature, candidate.gene, candidate.variant,
    )


def read_rna_candidates(path: Path) -> tuple[list[RnaCandidate], InputInventory]:
    candidates: list[RnaCandidate] = []
    reasons: Counter[str] = Counter()
    all_variants: set[str] = set()
    ineligible_variants: set[str] = set()
    input_rows = 0
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"missing_header: {path}")
        headers = [normalize_header(name) for name in reader.fieldnames]
        if len(headers) != len(set(headers)):
            raise ValueError("duplicate_column")
        missing = sorted(set(REQUIRED_COLUMNS) - set(headers))
        if missing:
            raise ValueError(f"missing_columns: {', '.join(missing)}")

        for input_index, raw in enumerate(reader):
            input_rows += 1
            row = {normalize_header(key): (value or "").strip() for key, value in raw.items()}
            label = row["variant"] or f"row {input_index + 2}"
            required = ("variant", "gene", "feature", "chromosome", "start", "end", "ref")
            empty = [name for name in required if row[name] in _MISSING]
            if empty:
                raise ValueError(f"missing_value: {label}: {', '.join(empty)}")
            all_variants.add(row["variant"])
            try:
                start, end = int(row["start"]), int(row["end"])
            except ValueError:
                raise ValueError(
                    f"invalid_coordinate: {label}: start={row['start']!r}, end={row['end']!r}"
                ) from None

            ref, alt = row["ref"].upper(), row["alt"].upper()
            cdna = row["cdna_position"]
            reason = (
                "missing_cdna_position"
                if cdna in _MISSING
                else "unsupported_ref"
                if len(ref) != 1 or ref not in NUCLEOTIDES
                else "unsupported_alt"
                if len(alt) != 1 or alt not in NUCLEOTIDES
                else None
            )
            if reason is not None:
                reasons[reason] += 1
                ineligible_variants.add(row["variant"])
                continue
            try:
                cdna_position = int(cdna)
            except ValueError:
                raise ValueError(f"invalid_cdna_position: {label}: {cdna!r}") from None
            candidates.append(
                RnaCandidate(
                    row["variant"], normalize_stable_id(row["gene"]),
                    normalize_stable_id(row["feature"]), row["chromosome"], start, end,
                    cdna_position, ref, alt, input_index,
                )
            )

    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_eligible_row")
    eligible_variants = {candidate.variant for candidate in candidates}
    return candidates, InputInventory(
        input_rows,
        len(all_variants),
        len(candidates),
        len(eligible_variants),
        sum(reasons.values()),
        len(ineligible_variants),
        dict(sorted(reasons.items())),
    )


def read_transcript_fasta(path: Path, required: set[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    def commit() -> None:
        if current is None or current not in required:
            return
        sequence = "".join(chunks).upper().replace("U", "T")
        if current in sequences and sequences[current] != sequence:
            raise ValueError(f"conflicting_fasta_record: {current}")
        sequences[current] = sequence

    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                current = normalize_stable_id(
                    line[1:].split(None, 1)[0].split("|", 1)[0]
                )
                chunks = []
            elif current is None:
                raise ValueError(f"invalid_fasta: sequence before header at line {line_number}")
            else:
                chunks.append(line)
        commit()
    missing = sorted(required - set(sequences))
    if missing:
        raise ValueError(f"missing_transcript_fasta: {missing[0]}")
    return sequences


def gff_attributes(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw.split(";"):
        key, separator, value = item.partition("=")
        if separator:
            result[unquote(key)] = unquote(value)
    return result


def transcript_id(attributes: Mapping[str, str], record_type: str) -> str | None:
    raw = attributes.get("transcript_id")
    if raw is None and record_type in ("transcript", "mRNA", "lnc_RNA"):
        raw = attributes.get("ID")
    if raw is None:
        raw = attributes.get("Parent")
    if raw is None:
        return None
    raw = raw.split(",", 1)[0]
    if ":" in raw and raw.split(":", 1)[0] in ("transcript", "rna", "mRNA"):
        raw = raw.split(":", 1)[1]
    return normalize_stable_id(raw)


def prepare_requests(
    candidates: Sequence[RnaCandidate],
    annotations: Mapping[str, TranscriptAnnotation],
) -> list[PreparedRequest]:
    requests: list[PreparedRequest] = []
    for candidate in candidates:
        annotation = annotations.get(candidate.feature)
        if annotation is None:
            raise ValueError(f"missing_transcript: {candidate.feature}")
        label = f"{candidate.variant}/{candidate.gene}/{candidate.feature}"
        if annotation.gene is not None and annotation.gene != candidate.gene:
            raise ValueError(
                f"transcript_gene_mismatch: {label}: input={candidate.gene}, GFF3={annotation.gene}"
            )
        input_chromosome = candidate.chromosome.removeprefix("chr").upper().replace("MT", "M")
        annotation_chromosome = annotation.chromosome.removeprefix("chr").upper().replace("MT", "M")
        if input_chromosome != annotation_chromosome:
            raise ValueError(
                f"transcript_chromosome_mismatch: {label}: "
                f"input={candidate.chromosome}, GFF3={annotation.chromosome}"
            )
        if candidate.end != candidate.start + 1 or candidate.start < 0:
            raise ValueError(
                f"not_snv_interval: {label}: [{candidate.start}, {candidate.end})"
            )
        if candidate.ref == candidate.alt:
            raise ValueError(f"identical_alleles: {label}: {candidate.ref}>{candidate.alt}")
        if not 1 <= candidate.cdna_position <= len(annotation.sequence):
            raise ValueError(
                f"cdna_position_out_of_bounds: {label}: "
                f"{candidate.cdna_position}, length={len(annotation.sequence)}"
            )
        observed_position = annotation.genomic_position(candidate.target_index)
        if observed_position != candidate.start:
            raise ValueError(
                f"cdna_genomic_position_mismatch: {label}: "
                f"input={candidate.start}, GFF3={observed_position}"
            )
        transcript_ref = (
            candidate.ref
            if annotation.strand == "+"
            else candidate.ref.translate(COMPLEMENT)
        )
        transcript_alt = (
            candidate.alt
            if annotation.strand == "+"
            else candidate.alt.translate(COMPLEMENT)
        )
        observed_ref = annotation.sequence[candidate.target_index]
        if observed_ref != transcript_ref:
            raise ValueError(
                f"transcript_reference_mismatch: {label}: "
                f"input={transcript_ref}, FASTA={observed_ref}"
            )
        requests.append(
            PreparedRequest(candidate, annotation, transcript_ref, transcript_alt)
        )
    return requests


def shard_candidates(
    candidates: Sequence[RnaCandidate], *, num_shards: int, shard_index: int
) -> list[RnaCandidate]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    ordered = sorted(candidates, key=candidate_sort_key)
    start = len(ordered) * shard_index // num_shards
    end = len(ordered) * (shard_index + 1) // num_shards
    return ordered[start:end]


def output_candidate_key(row: Mapping[str, str]) -> tuple[Any, ...]:
    return (
        row["variant"], normalize_stable_id(row["gene"]),
        normalize_stable_id(row["feature"]), row["chromosome"], int(row["start"]),
        int(row["end"]), int(row["cdna_position"]), row["ref"], row["alt"],
    )


def collate_rna_scores(
    candidates: Sequence[RnaCandidate],
    *,
    output_dir: Path,
    output_path: Path,
    num_shards: int,
    columns: Sequence[str],
) -> Path:
    ordered = sorted(candidates, key=candidate_sort_key)
    rows_by_key: dict[tuple[Any, ...], dict[str, str]] = {}
    for shard_index in range(num_shards):
        path = shard_path(output_dir, num_shards, shard_index)
        if not path.is_file():
            raise ValueError(f"missing shard {shard_index}: {path}")
        rows = read_score_table(path, columns)
        expected = shard_candidates(
            ordered, num_shards=num_shards, shard_index=shard_index
        )
        observed_keys = [output_candidate_key(row) for row in rows]
        expected_keys = [candidate.key for candidate in expected]
        if len(observed_keys) != len(set(observed_keys)):
            raise ValueError(f"duplicate row within shard: {path}")
        if set(observed_keys) != set(expected_keys):
            missing = list(set(expected_keys) - set(observed_keys))
            extra = list(set(observed_keys) - set(expected_keys))
            raise ValueError(
                f"shard key coverage mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}"
            )
        for key, row in zip(observed_keys, rows, strict=True):
            if key in rows_by_key:
                raise ValueError(f"duplicate row across shards: {key!r}")
            rows_by_key[key] = row

    expected_keys = [candidate.key for candidate in ordered]
    missing = [key for key in expected_keys if key not in rows_by_key]
    extra = list(set(rows_by_key) - set(expected_keys))
    if missing or extra:
        raise ValueError(
            f"final row coverage mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}"
        )
    write_score_table(
        output_path, columns, [rows_by_key[key] for key in expected_keys]
    )
    return output_path


OrthrusCandidate = RnaCandidate
read_orthrus_candidates = read_rna_candidates

__all__ = [
    "ComponentScores", "InputInventory", "OrthrusCandidate", "PreparedRequest",
    "RnaCandidate", "TranscriptAnnotation", "candidate_sort_key",
    "collate_rna_scores", "gff_attributes", "natural_chromosome_key",
    "normalize_stable_id", "open_text", "output_candidate_key",
    "prepare_requests", "read_orthrus_candidates", "read_rna_candidates",
    "read_transcript_fasta", "shard_candidates", "transcript_id",
]
