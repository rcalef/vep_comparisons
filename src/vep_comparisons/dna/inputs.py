"""Genomic candidates, reference sequences, and gene annotations."""

from __future__ import annotations

import csv
import gzip
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TextIO

NUCLEOTIDES = frozenset("ACGT")
REQUIRED_COLUMNS = (
    "variant", "gene", "feature", "chromosome", "start", "end", "ref", "alt",
)


@dataclass(frozen=True)
class DnaCandidate:
    variant: str
    gene: str
    feature: str
    chromosome: str
    start: int
    end: int
    ref: str
    alt: str
    input_index: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.variant, self.gene)

    @property
    def variant_signature(self) -> tuple[str, int, int, str, str]:
        return (self.chromosome, self.start, self.end, self.ref, self.alt)


@dataclass(frozen=True)
class GeneSpan:
    gene: str
    chromosome: str
    start: int
    end: int
    strand: str


class ReferenceGenome(Protocol):
    @property
    def references(self) -> Sequence[str]: ...

    def get_reference_length(self, chromosome: str) -> int: ...

    def fetch(self, chromosome: str, start: int, end: int) -> str: ...


class IndexedReference:
    def __init__(self, path: Path) -> None:
        import pysam

        self._fasta = pysam.FastaFile(str(path))

    @property
    def references(self) -> Sequence[str]:
        return self._fasta.references

    def get_reference_length(self, chromosome: str) -> int:
        return self._fasta.get_reference_length(chromosome)

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        return self._fasta.fetch(chromosome, start, end).upper()

    def __enter__(self) -> IndexedReference:
        return self

    def __exit__(self, *_: object) -> None:
        self._fasta.close()


def normalize_header(name: str) -> str:
    normalized = name.lstrip("#").strip().lower()
    return "variant" if normalized == "uploaded_variation" else normalized


def normalize_gene(value: str) -> str:
    return value.split(".", maxsplit=1)[0]


def _open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")


def read_dna_candidates(path: Path) -> list[DnaCandidate]:
    candidates: list[DnaCandidate] = []
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"missing_header: {path}")
        normalized = [normalize_header(name) for name in reader.fieldnames]
        if len(normalized) != len(set(normalized)):
            raise ValueError("duplicate_column")
        missing = sorted(set(REQUIRED_COLUMNS) - set(normalized))
        if missing:
            raise ValueError(f"missing_columns: {', '.join(missing)}")

        for input_index, raw in enumerate(reader):
            row = {normalize_header(key): value for key, value in raw.items()}
            label = row["variant"] or f"row {input_index + 2}"
            empty = [
                name
                for name in REQUIRED_COLUMNS
                if name != "alt" and row[name] in (None, "", "-")
            ]
            if empty:
                raise ValueError(f"missing_value: {label}: {', '.join(empty)}")
            try:
                start, end = int(row["start"]), int(row["end"])
            except ValueError:
                raise ValueError(
                    f"invalid_coordinate: {label}: start={row['start']!r}, end={row['end']!r}"
                ) from None
            candidates.append(
                DnaCandidate(
                    row["variant"],
                    normalize_gene(row["gene"]),
                    row["feature"],
                    row["chromosome"],
                    start,
                    end,
                    row["ref"].upper(),
                    (row.get("alt") or "").upper(),
                    input_index,
                )
            )
    return candidates


def filter_candidates_by_alt(
    candidates: Sequence[DnaCandidate],
) -> tuple[list[DnaCandidate], list[DnaCandidate]]:
    retained = [candidate for candidate in candidates if candidate.alt in NUCLEOTIDES]
    filtered = [candidate for candidate in candidates if candidate.alt not in NUCLEOTIDES]
    return retained, filtered


def contig_aliases(name: str) -> tuple[str, ...]:
    aliases = [name, name[3:] if name.startswith("chr") else f"chr{name}"]
    if name in ("M", "MT", "chrM", "chrMT"):
        aliases.extend(("M", "MT", "chrM", "chrMT"))
    return tuple(dict.fromkeys(aliases))


def resolve_contig(name: str, references: Sequence[str]) -> str | None:
    available = set(references)
    return next((alias for alias in contig_aliases(name) if alias in available), None)


def validate_and_resolve_candidates(
    candidates: Sequence[DnaCandidate], reference: ReferenceGenome
) -> list[DnaCandidate]:
    seen_keys: set[tuple[str, str]] = set()
    signatures: dict[str, tuple[str, int, int, str, str]] = {}
    resolved: list[DnaCandidate] = []
    for candidate in candidates:
        if candidate.key in seen_keys:
            raise ValueError(f"duplicate_variant_gene: {candidate.key!r}")
        seen_keys.add(candidate.key)
        if candidate.end != candidate.start + 1:
            raise ValueError(
                f"not_snv_interval: {candidate.variant}: [{candidate.start}, {candidate.end})"
            )
        if len(candidate.ref) != 1 or len(candidate.alt) != 1:
            raise ValueError(f"not_snv_alleles: {candidate.variant}: {candidate.ref}>{candidate.alt}")
        if candidate.ref not in NUCLEOTIDES or candidate.alt not in NUCLEOTIDES:
            raise ValueError(f"invalid_allele: {candidate.variant}: {candidate.ref}>{candidate.alt}")
        if candidate.ref == candidate.alt:
            raise ValueError(f"identical_alleles: {candidate.variant}: {candidate.ref}>{candidate.alt}")

        chromosome = resolve_contig(candidate.chromosome, reference.references)
        if chromosome is None:
            raise ValueError(f"missing_contig: {candidate.variant}: {candidate.chromosome}")
        updated = replace(candidate, chromosome=chromosome)
        length = reference.get_reference_length(chromosome)
        if not 0 <= updated.start < updated.end <= length:
            raise ValueError(
                f"coordinate_out_of_bounds: {candidate.variant}: "
                f"{chromosome}:{candidate.start}-{candidate.end}, length={length}"
            )
        previous = signatures.setdefault(candidate.variant, updated.variant_signature)
        if previous != updated.variant_signature:
            raise ValueError(
                f"inconsistent_variant: {candidate.variant}: {previous!r} != {updated.variant_signature!r}"
            )
        resolved.append(updated)

    checked: set[str] = set()
    for candidate in sorted(
        resolved, key=lambda item: (item.chromosome, item.start, item.variant)
    ):
        if candidate.variant in checked:
            continue
        checked.add(candidate.variant)
        observed = reference.fetch(candidate.chromosome, candidate.start, candidate.end)
        if observed != candidate.ref:
            raise ValueError(
                f"reference_mismatch: {candidate.variant}: input={candidate.ref}, FASTA={observed!r}"
            )
    return resolved


_ATTRIBUTE_RE = re.compile(r'(\S+)\s+"([^"]*)"')


def read_gene_spans(
    path: Path,
    references: Sequence[str],
    *,
    required_genes: set[str] | None = None,
) -> dict[str, GeneSpan]:
    spans: dict[str, GeneSpan] = {}
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attributes = dict(_ATTRIBUTE_RE.findall(fields[8]))
            raw_gene = attributes.get("gene_id")
            if raw_gene is None:
                raise ValueError(f"gtf_missing_gene_id: line {line_number}")
            gene = normalize_gene(raw_gene)
            if required_genes is not None and gene not in required_genes:
                continue
            chromosome = resolve_contig(fields[0], references)
            if chromosome is None:
                raise ValueError(f"gtf_missing_contig: {gene}: {fields[0]}")
            try:
                start, end = int(fields[3]) - 1, int(fields[4])
            except ValueError:
                raise ValueError(f"gtf_invalid_coordinate: line {line_number}") from None
            span = GeneSpan(gene, chromosome, start, end, fields[6])
            if gene in spans and spans[gene] != span:
                raise ValueError(f"conflicting_gene_span: {gene}: {spans[gene]!r} != {span!r}")
            spans[gene] = span
    return spans


__all__ = [
    "DnaCandidate", "GeneSpan", "IndexedReference", "ReferenceGenome",
    "contig_aliases", "filter_candidates_by_alt", "normalize_gene",
    "read_dna_candidates", "read_gene_spans", "resolve_contig",
    "validate_and_resolve_candidates",
]
