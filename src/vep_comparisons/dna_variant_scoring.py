"""Planning, validation, sharding, and publication for genomic SNV scoring.

Coordinates in this module are always zero-based and half-open.  Heavy model
dependencies are deliberately kept out of this file so all input failures are
reported before CUDA is initialized.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shlex
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, TextIO


MODEL_NAME = "ntv3-100m-pre"
SUPPORTED_MODELS = frozenset(("ntv3-100m-pre", "ntv3-650m-pre"))
LICENSE_NOTICE = "NTv3 model and derived outputs are restricted to non-commercial use."
NUCLEOTIDES = frozenset("ACGT")
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
REQUIRED_COLUMNS = (
    "variant",
    "gene",
    "feature",
    "chromosome",
    "start",
    "end",
    "ref",
    "alt",
)
OUTPUT_COLUMNS = (
    *REQUIRED_COLUMNS,
    "model",
    "window_length",
    "centered_window_start",
    "centered_window_end",
    "centered_left_pad",
    "centered_right_pad",
    "score_forward",
    "score_reverse",
    "score",
    "gene_context_status",
    "gene_window_start",
    "gene_window_end",
    "gene_score_forward",
    "gene_score_reverse",
    "gene_score",
)
GENE_CONTEXT_STATUSES = frozenset(
    (
        "centered_full_gene",
        "shifted_full_gene",
        "gene_too_long",
        "missing_gene_span",
        "not_requested",
    )
)


@dataclass(frozen=True)
class ValidationIssue:
    category: str
    detail: str


class InputValidationError(ValueError):
    """Aggregated input failures, with bounded examples in the message."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = list(issues)
        counts = Counter(issue.category for issue in self.issues)
        lines = [
            f"Input validation failed with {len(self.issues)} error(s) "
            f"across {len(counts)} category/categories:"
        ]
        for category, count in sorted(counts.items()):
            lines.append(f"- {category}: {count}")
            examples = [
                issue.detail for issue in self.issues if issue.category == category
            ][:3]
            lines.extend(f"  example: {example}" for example in examples)
        super().__init__("\n".join(lines))


class ShardCompatibilityError(ValueError):
    pass


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


@dataclass(frozen=True)
class WindowPlan:
    chromosome: str
    start: int
    end: int
    target_index: int
    left_pad: int
    right_pad: int


@dataclass(frozen=True)
class ScoringContext:
    chromosome: str
    window_start: int
    window_length: int
    target_index: int
    ref: str
    alt: str

    @property
    def key(self) -> tuple[str, int, int, int, str, str]:
        return (
            self.chromosome,
            self.window_start,
            self.window_length,
            self.target_index,
            self.ref,
            self.alt,
        )


@dataclass(frozen=True)
class OrientationScores:
    forward: float
    reverse: float

    @property
    def mean(self) -> float:
        return (self.forward + self.reverse) / 2.0


@dataclass(frozen=True)
class CandidatePlan:
    candidate: DnaCandidate
    centered: WindowPlan
    gene_context_status: str
    gene_window: WindowPlan | None


@dataclass(frozen=True)
class DnaScoringSummary:
    output_path: Path
    metadata_path: Path
    candidates: int
    unique_variants: int
    contexts: int
    runtime_seconds: float
    reused: bool = False


class ReferenceGenome(Protocol):
    @property
    def references(self) -> Sequence[str]: ...

    def get_reference_length(self, chromosome: str) -> int: ...

    def fetch(self, chromosome: str, start: int, end: int) -> str: ...


class DnaModel(Protocol):
    def score_contexts(
        self,
        contexts: Sequence[ScoringContext],
        *,
        reference: ReferenceGenome,
        batch_size: int,
    ) -> Mapping[tuple[str, int, int, int, str, str], OrientationScores]: ...


ModelFactory = Callable[[Path, Path, Path, str, str], DnaModel]


class IndexedReference:
    """Small wrapper around an indexed FASTA, including BGZF FASTA files."""

    def __init__(self, path: Path) -> None:
        import pysam

        self.path = path
        self._fasta = pysam.FastaFile(str(path))

    @property
    def references(self) -> Sequence[str]:
        return self._fasta.references

    def get_reference_length(self, chromosome: str) -> int:
        return self._fasta.get_reference_length(chromosome)

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        return self._fasta.fetch(chromosome, start, end).upper()

    def close(self) -> None:
        self._fasta.close()

    def __enter__(self) -> IndexedReference:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def normalize_header(name: str) -> str:
    normalized = name.lstrip("#").strip().lower()
    return "variant" if normalized == "uploaded_variation" else normalized


def normalize_gene(value: str) -> str:
    return value.split(".", maxsplit=1)[0]


def _open_text(path: Path, mode: str = "rt") -> TextIO:
    return gzip.open(path, mode, newline="") if path.suffix == ".gz" else path.open(mode, newline="")


def read_dna_candidates(path: Path) -> list[DnaCandidate]:
    """Read every variant-gene row and aggregate parse/shape failures."""

    issues: list[ValidationIssue] = []
    candidates: list[DnaCandidate] = []
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise InputValidationError([ValidationIssue("missing_header", str(path))])
        normalized = [normalize_header(name) for name in reader.fieldnames]
        duplicates = [name for name, count in Counter(normalized).items() if count > 1]
        missing = sorted(set(REQUIRED_COLUMNS) - set(normalized))
        if duplicates:
            issues.append(ValidationIssue("duplicate_column", ", ".join(duplicates)))
        if missing:
            issues.append(ValidationIssue("missing_columns", ", ".join(missing)))
        if issues:
            raise InputValidationError(issues)

        for input_index, raw in enumerate(reader):
            row = {normalize_header(key): value for key, value in raw.items()}
            label = row.get("variant") or f"row {input_index + 2}"
            # ALT is allowed to be malformed here because the DNA pipeline
            # intentionally filters non-ACGT alternate alleles before the
            # remaining candidate validation.
            empty = [
                name
                for name in REQUIRED_COLUMNS
                if name != "alt" and row.get(name) in (None, "", "-")
            ]
            if empty:
                issues.append(ValidationIssue("missing_value", f"{label}: {', '.join(empty)}"))
                continue
            try:
                start, end = int(row["start"]), int(row["end"])
            except ValueError:
                issues.append(
                    ValidationIssue(
                        "invalid_coordinate",
                        f"{label}: start={row['start']!r}, end={row['end']!r}",
                    )
                )
                continue
            candidates.append(
                DnaCandidate(
                    variant=row["variant"],
                    gene=normalize_gene(row["gene"]),
                    feature=row["feature"],
                    chromosome=row["chromosome"],
                    start=start,
                    end=end,
                    ref=row["ref"].upper(),
                    alt=(row.get("alt") or "").upper(),
                    input_index=input_index,
                )
            )
    if issues:
        raise InputValidationError(issues)
    return candidates


def filter_candidates_by_alt(
    candidates: Sequence[DnaCandidate],
) -> tuple[list[DnaCandidate], list[DnaCandidate]]:
    """Partition candidates by the supported single-nucleotide ALT alphabet."""

    retained: list[DnaCandidate] = []
    filtered: list[DnaCandidate] = []
    for candidate in candidates:
        (retained if candidate.alt in NUCLEOTIDES else filtered).append(candidate)
    return retained, filtered


def contig_aliases(name: str) -> tuple[str, ...]:
    aliases = [name]
    if name.startswith("chr"):
        aliases.append(name[3:])
    else:
        aliases.append(f"chr{name}")
    if name in ("M", "MT", "chrM", "chrMT"):
        aliases.extend(("M", "MT", "chrM", "chrMT"))
    return tuple(dict.fromkeys(aliases))


def resolve_contig(name: str, references: Sequence[str]) -> str | None:
    available = set(references)
    return next((alias for alias in contig_aliases(name) if alias in available), None)


def validate_and_resolve_candidates(
    candidates: Sequence[DnaCandidate], reference: ReferenceGenome
) -> list[DnaCandidate]:
    """Validate candidate invariants and REF bases, returning FASTA contig names."""

    issues: list[ValidationIssue] = []
    seen_keys: set[tuple[str, str]] = set()
    signatures: dict[str, tuple[str, int, int, str, str]] = {}
    resolved: list[DnaCandidate] = []
    for candidate in candidates:
        if candidate.key in seen_keys:
            issues.append(ValidationIssue("duplicate_variant_gene", repr(candidate.key)))
        seen_keys.add(candidate.key)
        if candidate.end != candidate.start + 1:
            issues.append(
                ValidationIssue(
                    "not_snv_interval",
                    f"{candidate.variant}: [{candidate.start}, {candidate.end})",
                )
            )
        if len(candidate.ref) != 1 or len(candidate.alt) != 1:
            issues.append(
                ValidationIssue("not_snv_alleles", f"{candidate.variant}: {candidate.ref}>{candidate.alt}")
            )
        elif candidate.ref not in NUCLEOTIDES or candidate.alt not in NUCLEOTIDES:
            issues.append(
                ValidationIssue("invalid_allele", f"{candidate.variant}: {candidate.ref}>{candidate.alt}")
            )
        elif candidate.ref == candidate.alt:
            issues.append(
                ValidationIssue("identical_alleles", f"{candidate.variant}: {candidate.ref}>{candidate.alt}")
            )
        chromosome = resolve_contig(candidate.chromosome, reference.references)
        if chromosome is None:
            issues.append(
                ValidationIssue("missing_contig", f"{candidate.variant}: {candidate.chromosome}")
            )
            continue
        updated = replace(candidate, chromosome=chromosome)
        length = reference.get_reference_length(chromosome)
        if not (0 <= candidate.start < candidate.end <= length):
            issues.append(
                ValidationIssue(
                    "coordinate_out_of_bounds",
                    f"{candidate.variant}: {chromosome}:{candidate.start}-{candidate.end}, length={length}",
                )
            )
        resolved.append(updated)

        previous = signatures.setdefault(candidate.variant, updated.variant_signature)
        if previous != updated.variant_signature:
            issues.append(
                ValidationIssue(
                    "inconsistent_variant",
                    f"{candidate.variant}: {previous!r} != {updated.variant_signature!r}",
                )
            )

    # Coordinate sorting substantially improves indexed BGZF locality.
    valid_coordinates = sorted(
        (candidate for candidate in resolved if 0 <= candidate.start < candidate.end <= reference.get_reference_length(candidate.chromosome)),
        key=lambda item: (item.chromosome, item.start, item.variant),
    )
    checked: set[str] = set()
    for candidate in valid_coordinates:
        if candidate.variant in checked or len(candidate.ref) != 1:
            continue
        checked.add(candidate.variant)
        observed = reference.fetch(candidate.chromosome, candidate.start, candidate.end)
        if observed != candidate.ref:
            issues.append(
                ValidationIssue(
                    "reference_mismatch",
                    f"{candidate.variant}: input={candidate.ref}, FASTA={observed!r}",
                )
            )
    if issues:
        raise InputValidationError(issues)
    return resolved


_ATTRIBUTE_RE = re.compile(r'(\S+)\s+"([^"]*)"')


def read_gene_spans(
    path: Path,
    references: Sequence[str],
    *,
    required_genes: set[str] | None = None,
) -> dict[str, GeneSpan]:
    """Parse GTF gene records, converting one-based closed coordinates."""

    spans: dict[str, GeneSpan] = {}
    issues: list[ValidationIssue] = []
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attributes = dict(_ATTRIBUTE_RE.findall(fields[8]))
            raw_gene = attributes.get("gene_id")
            if not raw_gene:
                issues.append(ValidationIssue("gtf_missing_gene_id", f"line {line_number}"))
                continue
            gene = normalize_gene(raw_gene)
            if required_genes is not None and gene not in required_genes:
                continue
            chromosome = resolve_contig(fields[0], references)
            if chromosome is None:
                issues.append(
                    ValidationIssue("gtf_missing_contig", f"{gene}: {fields[0]}")
                )
                continue
            try:
                start, end = int(fields[3]) - 1, int(fields[4])
            except ValueError:
                issues.append(ValidationIssue("gtf_invalid_coordinate", f"line {line_number}"))
                continue
            span = GeneSpan(gene, chromosome, start, end, fields[6])
            previous = spans.get(gene)
            if previous is not None and previous != span:
                issues.append(
                    ValidationIssue("conflicting_gene_span", f"{gene}: {previous!r} != {span!r}")
                )
            else:
                spans[gene] = span
    if issues:
        raise InputValidationError(issues)
    return spans


def validate_window_policy(window_length: int, min_variant_margin: int) -> None:
    issues: list[ValidationIssue] = []
    if window_length <= 0 or window_length % 128:
        issues.append(
            ValidationIssue("invalid_window_length", f"{window_length}; must be positive and divisible by 128")
        )
    if min_variant_margin < 0 or min_variant_margin >= window_length / 2:
        issues.append(
            ValidationIssue(
                "invalid_variant_margin",
                f"{min_variant_margin}; require 0 <= margin < window_length / 2",
            )
        )
    if issues:
        raise InputValidationError(issues)


def centered_window(candidate: DnaCandidate, contig_length: int, window_length: int) -> WindowPlan:
    target_index = window_length // 2
    start = candidate.start - target_index
    end = start + window_length
    return WindowPlan(
        candidate.chromosome,
        start,
        end,
        target_index,
        max(0, -start),
        max(0, end - contig_length),
    )


def _contains_gene_and_margin(
    start: int,
    *,
    window_length: int,
    variant_start: int,
    gene: GeneSpan,
    margin: int,
) -> bool:
    return (
        start <= gene.start
        and gene.end <= start + window_length
        and start <= variant_start - margin
        and variant_start + 1 + margin <= start + window_length
    )


def plan_gene_window(
    candidate: DnaCandidate,
    centered: WindowPlan,
    gene: GeneSpan | None,
    *,
    contig_length: int,
    window_length: int,
    min_variant_margin: int,
) -> tuple[str, WindowPlan | None]:
    if gene is None:
        return "missing_gene_span", None
    if gene.chromosome != candidate.chromosome:
        raise InputValidationError(
            [
                ValidationIssue(
                    "gene_contig_mismatch",
                    f"{candidate.variant}/{candidate.gene}: variant={candidate.chromosome}, gene={gene.chromosome}",
                )
            ]
        )
    if _contains_gene_and_margin(
        centered.start,
        window_length=window_length,
        variant_start=candidate.start,
        gene=gene,
        margin=min_variant_margin,
    ):
        return "centered_full_gene", centered

    lower = max(
        gene.end - window_length,
        candidate.start + 1 + min_variant_margin - window_length,
        0,
    )
    upper = min(
        gene.start,
        candidate.start - min_variant_margin,
        contig_length - window_length,
    )
    if lower > upper:
        return "gene_too_long", None
    start = min(max(centered.start, lower), upper)
    target_index = candidate.start - start
    return (
        "shifted_full_gene",
        WindowPlan(candidate.chromosome, start, start + window_length, target_index, 0, 0),
    )


def plan_candidates(
    candidates: Sequence[DnaCandidate],
    gene_spans: Mapping[str, GeneSpan],
    reference: ReferenceGenome,
    *,
    window_length: int,
    min_variant_margin: int,
    include_gene_context: bool = True,
) -> list[CandidatePlan]:
    plans: list[CandidatePlan] = []
    issues: list[ValidationIssue] = []
    centered_by_variant: dict[str, WindowPlan] = {}
    for candidate in candidates:
        contig_length = reference.get_reference_length(candidate.chromosome)
        centered = centered_by_variant.setdefault(
            candidate.variant, centered_window(candidate, contig_length, window_length)
        )
        if not include_gene_context:
            plans.append(CandidatePlan(candidate, centered, "not_requested", None))
            continue
        try:
            status, gene_window = plan_gene_window(
                candidate,
                centered,
                gene_spans.get(candidate.gene),
                contig_length=contig_length,
                window_length=window_length,
                min_variant_margin=min_variant_margin,
            )
        except InputValidationError as error:
            issues.extend(error.issues)
            continue
        plans.append(CandidatePlan(candidate, centered, status, gene_window))
    if issues:
        raise InputValidationError(issues)
    return plans


def context_for(candidate: DnaCandidate, window: WindowPlan) -> ScoringContext:
    return ScoringContext(
        candidate.chromosome,
        window.start,
        window.end - window.start,
        window.target_index,
        candidate.ref,
        candidate.alt,
    )


def deduplicate_contexts(plans: Sequence[CandidatePlan]) -> list[ScoringContext]:
    result: dict[tuple[str, int, int, int, str, str], ScoringContext] = {}
    for plan in plans:
        centered = context_for(plan.candidate, plan.centered)
        result.setdefault(centered.key, centered)
        if plan.gene_context_status == "shifted_full_gene":
            assert plan.gene_window is not None
            gene_context = context_for(plan.candidate, plan.gene_window)
            result.setdefault(gene_context.key, gene_context)
    return list(result.values())


def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(COMPLEMENT)[::-1]


def complemented(allele: str) -> str:
    return allele.translate(COMPLEMENT)


def extract_window(reference: ReferenceGenome, context: ScoringContext) -> str:
    contig_length = reference.get_reference_length(context.chromosome)
    fetch_start = max(0, context.window_start)
    fetch_end = min(contig_length, context.window_start + context.window_length)
    sequence = (
        "N" * max(0, -context.window_start)
        + reference.fetch(context.chromosome, fetch_start, fetch_end).upper()
        + "N" * max(0, context.window_start + context.window_length - contig_length)
    )
    if len(sequence) != context.window_length:
        raise RuntimeError(f"window extraction returned {len(sequence)}, expected {context.window_length}")
    if sequence[context.target_index] != context.ref:
        raise RuntimeError(
            f"window REF mismatch for {context.key}: sequence={sequence[context.target_index]!r}"
        )
    return sequence


def shard_variant_ids(
    candidates: Sequence[DnaCandidate],
    references: Sequence[str],
    *,
    num_shards: int,
    shard_index: int,
) -> list[str]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    contig_order = {name: index for index, name in enumerate(references)}
    representative: dict[str, DnaCandidate] = {}
    for candidate in candidates:
        representative.setdefault(candidate.variant, candidate)
    ordered = sorted(
        representative.values(),
        key=lambda item: (contig_order[item.chromosome], item.start, item.end, item.ref, item.alt, item.variant),
    )
    start = len(ordered) * shard_index // num_shards
    end = len(ordered) * (shard_index + 1) // num_shards
    return [candidate.variant for candidate in ordered[start:end]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, checksum: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if checksum:
        result["sha256"] = sha256_file(resolved)
    return result


def shard_paths(output_dir: Path, num_shards: int, shard_index: int) -> tuple[Path, Path]:
    stem = f"shard-{shard_index:05d}-of-{num_shards:05d}"
    return output_dir / f"{stem}.tsv.gz", output_dir / f"{stem}.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _score_fields(scores: OrientationScores) -> tuple[float, float, float]:
    values = (scores.forward, scores.reverse, scores.mean)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"model returned non-finite scores: {values!r}")
    return values


def assemble_rows(
    plans: Sequence[CandidatePlan],
    scores: Mapping[tuple[str, int, int, int, str, str], OrientationScores],
    *,
    window_length: int,
    model_name: str = MODEL_NAME,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in sorted(plans, key=lambda item: item.candidate.input_index):
        candidate = plan.candidate
        centered_scores = scores[context_for(candidate, plan.centered).key]
        score_forward, score_reverse, score = _score_fields(centered_scores)
        gene_window = plan.gene_window
        if gene_window is None:
            gene_values: tuple[float | None, float | None, float | None] = (None, None, None)
        elif plan.gene_context_status == "centered_full_gene":
            gene_values = (score_forward, score_reverse, score)
        else:
            gene_values = _score_fields(scores[context_for(candidate, gene_window).key])
        rows.append(
            {
                "variant": candidate.variant,
                "gene": candidate.gene,
                "feature": candidate.feature,
                "chromosome": candidate.chromosome,
                "start": candidate.start,
                "end": candidate.end,
                "ref": candidate.ref,
                "alt": candidate.alt,
                "model": model_name,
                "window_length": window_length,
                "centered_window_start": plan.centered.start,
                "centered_window_end": plan.centered.end,
                "centered_left_pad": plan.centered.left_pad,
                "centered_right_pad": plan.centered.right_pad,
                "score_forward": score_forward,
                "score_reverse": score_reverse,
                "score": score,
                "gene_context_status": plan.gene_context_status,
                "gene_window_start": None if gene_window is None else gene_window.start,
                "gene_window_end": None if gene_window is None else gene_window.end,
                "gene_score_forward": gene_values[0],
                "gene_score_reverse": gene_values[1],
                "gene_score": gene_values[2],
            }
        )
    return rows


def _software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version()}
    for package in ("vep-comparisons", "transformers", "torch", "pysam"):
        try:
            from importlib.metadata import version

            versions[package] = version(package)
        except Exception:
            versions[package] = None
    try:
        import torch

        versions["cuda"] = torch.version.cuda
    except Exception:
        versions["cuda"] = None
    return versions


def _runtime_device(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {"device": device, "gpu_model": None, "peak_gpu_memory_bytes": None}
    try:
        import torch

        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            result["gpu_model"] = torch.cuda.get_device_name(torch.device(device))
            result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(torch.device(device))
    except Exception:
        pass
    return result


def _default_model_factory(
    model_dir: Path, model_code_dir: Path, manifest_path: Path, device: str, dtype: str
) -> DnaModel:
    from .ntv3_model import NTv3Model

    return NTv3Model(model_dir, model_code_dir, manifest_path, device=device, dtype=dtype)


def score_dna_variants(
    *,
    variants_path: Path,
    reference_path: Path,
    genes_path: Path | None = None,
    model_dir: Path,
    model_code_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    model_name: str = MODEL_NAME,
    window_length: int = 8192,
    min_variant_margin: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: str = "float32",
    num_shards: int = 1,
    shard_index: int = 0,
    command_line: Sequence[str] | None = None,
    model_factory: ModelFactory = _default_model_factory,
) -> DnaScoringSummary:
    """Validate, score, and atomically publish one deterministic shard."""

    started = time.monotonic()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported DNA model: {model_name}")
    validate_window_policy(
        window_length, min_variant_margin if genes_path is not None else 0
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("dtype must be float32 or bfloat16")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")

    # Manifest checks include local file hashes and happen before the model import.
    from .ntv3_model import validate_model_package

    model_identity = validate_model_package(model_dir, model_code_dir, manifest_path)
    if model_identity.get("model") != model_name:
        raise ValueError(
            f"model manifest identifies {model_identity.get('model')!r}, expected {model_name!r}"
        )
    input_identity = file_identity(variants_path)
    reference_identity = file_identity(reference_path)
    genes_identity = None if genes_path is None else file_identity(genes_path)
    run_identity = {
        "schema_version": 1,
        "input": input_identity,
        "reference": reference_identity,
        "genes": genes_identity,
        "gene_context_mode": "centered_only" if genes_path is None else "gene_aware",
        "model_package": model_identity,
        "model": model_name,
        "window_length": window_length,
        "min_variant_margin": (
            min_variant_margin if genes_path is not None else None
        ),
        "dtype": dtype,
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    output_path, metadata_path = shard_paths(output_dir, num_shards, shard_index)
    if output_path.exists() or metadata_path.exists():
        if not output_path.is_file() or not metadata_path.is_file():
            raise ShardCompatibilityError(f"incomplete existing shard: {output_path}, {metadata_path}")
        existing = json.loads(metadata_path.read_text())
        if existing.get("run_identity") != run_identity:
            raise ShardCompatibilityError(f"existing shard metadata is incompatible: {metadata_path}")
        recorded_output = existing.get("output", {})
        if recorded_output.get("sha256") != sha256_file(output_path):
            raise ShardCompatibilityError(f"existing shard output checksum mismatch: {output_path}")
        return DnaScoringSummary(
            output_path,
            metadata_path,
            int(existing["counts"]["candidates"]),
            int(existing["counts"]["unique_variants"]),
            int(existing["counts"]["contexts"]),
            time.monotonic() - started,
            True,
        )

    with IndexedReference(reference_path) as reference:
        raw_candidates = read_dna_candidates(variants_path)
        retained_candidates, filtered_candidates = filter_candidates_by_alt(raw_candidates)
        candidates = validate_and_resolve_candidates(retained_candidates, reference)
        gene_spans = (
            {}
            if genes_path is None
            else read_gene_spans(
                genes_path,
                reference.references,
                required_genes={candidate.gene for candidate in candidates},
            )
        )
        # Gene/variant contig mismatches are fatal even when their row belongs to another shard.
        mismatch = [
            ValidationIssue(
                "gene_contig_mismatch",
                f"{candidate.variant}/{candidate.gene}: variant={candidate.chromosome}, gene={gene_spans[candidate.gene].chromosome}",
            )
            for candidate in candidates
            if candidate.gene in gene_spans and candidate.chromosome != gene_spans[candidate.gene].chromosome
        ]
        if mismatch:
            raise InputValidationError(mismatch)
        selected_ids = set(
            shard_variant_ids(
                candidates, reference.references, num_shards=num_shards, shard_index=shard_index
            )
        )
        selected = [candidate for candidate in candidates if candidate.variant in selected_ids]
        plans = plan_candidates(
            selected,
            gene_spans,
            reference,
            window_length=window_length,
            min_variant_margin=min_variant_margin,
            include_gene_context=genes_path is not None,
        )
        contexts = deduplicate_contexts(plans)

        model = model_factory(model_dir, model_code_dir, manifest_path, device, dtype)
        scored = model.score_contexts(contexts, reference=reference, batch_size=batch_size)
        missing_scores = [context.key for context in contexts if context.key not in scored]
        extra_scores = sorted(set(scored) - {context.key for context in contexts})
        if missing_scores or extra_scores:
            raise RuntimeError(
                f"model score key mismatch: missing={missing_scores[:3]!r}, extra={extra_scores[:3]!r}"
            )
        rows = assemble_rows(
            plans, scored, window_length=window_length, model_name=model_name
        )

    runtime = time.monotonic() - started
    counts = {
        "input_candidates": len(raw_candidates),
        "input_unique_variants": len({candidate.variant for candidate in raw_candidates}),
        "filtered_invalid_alt_candidates": len(filtered_candidates),
        "filtered_invalid_alt_variants": len(
            {candidate.variant for candidate in filtered_candidates}
        ),
        "retained_candidates": len(candidates),
        "retained_unique_variants": len({candidate.variant for candidate in candidates}),
        "candidates": len(selected),
        "unique_variants": len(selected_ids),
        "contexts": len(contexts),
        "orientation_requests": 2 * len(contexts),
        "gene_context_statuses": dict(sorted(Counter(plan.gene_context_status for plan in plans).items())),
        "filtered_invalid_alt_counts": dict(
            sorted(Counter(candidate.alt or "<empty>" for candidate in filtered_candidates).items())
        ),
        "filtered_invalid_alt_examples": [
            candidate.variant for candidate in filtered_candidates[:10]
        ],
    }
    metadata: dict[str, Any] = {
        "run_identity": run_identity,
        "counts": counts,
        "runtime_seconds": runtime,
        "batch_size": batch_size,
        "software": _software_versions(),
        "hardware": _runtime_device(device),
        "command": shlex.join(command_line if command_line is not None else sys.argv),
        "license": LICENSE_NOTICE,
        "output": {"path": str(output_path.resolve()), "columns": list(OUTPUT_COLUMNS)},
    }
    # The TSV is made visible first and the metadata acts as its completion marker.
    _atomic_write_rows(output_path, rows)
    metadata["output"].update(
        {"size": output_path.stat().st_size, "sha256": sha256_file(output_path)}
    )
    _atomic_write_json(metadata_path, metadata)
    return DnaScoringSummary(
        output_path, metadata_path, len(selected), len(selected_ids), len(contexts), runtime
    )


def _read_output_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ShardCompatibilityError(f"missing output header: {path}")
        return reader.fieldnames, list(reader)


def collate_dna_scores(
    *,
    variants_path: Path,
    output_dir: Path,
    output_path: Path,
    num_shards: int,
) -> tuple[Path, Path]:
    """Validate all shards and publish one table in original input order."""

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    raw_candidates = read_dna_candidates(variants_path)
    candidates, filtered_candidates = filter_candidates_by_alt(raw_candidates)
    expected_keys = [candidate.key for candidate in candidates]
    if len(expected_keys) != len(set(expected_keys)):
        raise InputValidationError([ValidationIssue("duplicate_variant_gene", "input keys are not unique")])
    input_identity = file_identity(variants_path)
    common_identity: dict[str, Any] | None = None
    rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
    shard_metadata: list[dict[str, Any]] = []
    for shard_index in range(num_shards):
        shard_output, shard_meta_path = shard_paths(output_dir, num_shards, shard_index)
        if not shard_output.is_file() or not shard_meta_path.is_file():
            raise ShardCompatibilityError(f"missing shard {shard_index}: {shard_output}")
        metadata = json.loads(shard_meta_path.read_text())
        identity = metadata.get("run_identity")
        if not isinstance(identity, dict):
            raise ShardCompatibilityError(f"invalid shard metadata: {shard_meta_path}")
        if identity.get("shard_index") != shard_index or identity.get("num_shards") != num_shards:
            raise ShardCompatibilityError(f"incorrect shard identity: {shard_meta_path}")
        if identity.get("input") != input_identity:
            raise ShardCompatibilityError(f"input identity mismatch: {shard_meta_path}")
        if metadata.get("output", {}).get("sha256") != sha256_file(shard_output):
            raise ShardCompatibilityError(f"shard output checksum mismatch: {shard_output}")
        comparable = {key: value for key, value in identity.items() if key != "shard_index"}
        if common_identity is None:
            common_identity = comparable
        elif comparable != common_identity:
            raise ShardCompatibilityError(f"incompatible shard metadata: {shard_meta_path}")
        header, rows = _read_output_rows(shard_output)
        if tuple(header) != OUTPUT_COLUMNS:
            raise ShardCompatibilityError(f"output schema mismatch: {shard_output}")
        for row in rows:
            key = (row["variant"], normalize_gene(row["gene"]))
            if key in rows_by_key:
                raise ShardCompatibilityError(f"duplicate key across shards: {key!r}")
            rows_by_key[key] = row
        shard_metadata.append(metadata)

    missing = [key for key in expected_keys if key not in rows_by_key]
    extra = sorted(set(rows_by_key) - set(expected_keys))
    if missing or extra:
        raise ShardCompatibilityError(f"final key coverage mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}")
    ordered_rows = [rows_by_key[key] for key in expected_keys]
    _atomic_write_rows(output_path, ordered_rows)
    final_metadata_path = output_path.with_suffix(".json")
    final_metadata = {
        "schema_version": 1,
        "run_identity": common_identity,
        "counts": {
            "candidates": len(ordered_rows),
            "unique_variants": len({row["variant"] for row in ordered_rows}),
            "filtered_invalid_alt_candidates": len(filtered_candidates),
            "filtered_invalid_alt_variants": len(
                {candidate.variant for candidate in filtered_candidates}
            ),
            "shards": num_shards,
        },
        "shard_metadata": [str(shard_paths(output_dir, num_shards, index)[1].resolve()) for index in range(num_shards)],
        "output": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
        "license": LICENSE_NOTICE,
    }
    _atomic_write_json(final_metadata_path, final_metadata)
    return output_path, final_metadata_path
