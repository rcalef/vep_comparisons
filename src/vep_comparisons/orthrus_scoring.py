"""Transcript-aware masked-marginal scoring with the Orthrus RNA MLM.

The module intentionally has no import-time dependency on Torch, Transformers,
or Mamba.  Input and annotation failures are therefore reported before CUDA or
the isolated Orthrus environment is initialized.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO
from urllib.parse import unquote

from .dna_variant_scoring import InputValidationError, ShardCompatibilityError, ValidationIssue


MODEL_NAME = "orthrus-mlm-6-track"
SCHEMA_VERSION = 1
NUCLEOTIDES = "ACGT"
NUCLEOTIDE_INDEX = {base: index for index, base in enumerate(NUCLEOTIDES)}
COMPLEMENT = str.maketrans("ACGT", "TGCA")
REQUIRED_COLUMNS = (
    "variant",
    "gene",
    "feature",
    "chromosome",
    "start",
    "end",
    "cdna_position",
    "ref",
    "alt",
)
OUTPUT_COLUMNS = (
    "variant",
    "gene",
    "feature",
    "chromosome",
    "start",
    "end",
    "cdna_position",
    "ref",
    "alt",
    "strand",
    "transcript_ref",
    "transcript_alt",
    "model",
    "ref_log_probability",
    "alt_log_probability",
    "score",
)
INELIGIBLE_MISSING_CDNA = "missing_cdna_position"
INELIGIBLE_UNSUPPORTED_REF = "unsupported_ref"
INELIGIBLE_UNSUPPORTED_ALT = "unsupported_alt"
_MISSING = frozenset(("", "-", ".", "NA", "NaN", "nan", "null", "None"))
_AUTOSOME_RE = re.compile(r"^(?:chr)?([1-9]|1[0-9]|2[0-2])$")


@dataclass(frozen=True)
class OrthrusCandidate:
    variant: str
    gene: str
    feature: str
    chromosome: str
    start: int
    end: int
    cdna_position: int  # one-based, matching VEP input
    ref: str
    alt: str
    input_index: int

    @property
    def target_index(self) -> int:
        return self.cdna_position - 1

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.variant,
            self.gene,
            self.feature,
            self.chromosome,
            self.start,
            self.end,
            self.cdna_position,
            self.ref,
            self.alt,
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
    exons: tuple[tuple[int, int], ...]  # zero-based genomic intervals in 5'->3' order
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
    candidate: OrthrusCandidate
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


@dataclass(frozen=True)
class OrthrusScoringSummary:
    output_path: Path
    metadata_path: Path
    rows: int
    reused: bool = False


class RequestScorer(Protocol):
    def score_requests(
        self, requests: Sequence[PreparedRequest], *, batch_size: int
    ) -> Mapping[tuple[Any, ...], ComponentScores]: ...


ModelFactory = Callable[[Path, Path, str], RequestScorer]


def _open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")


def normalize_header(name: str) -> str:
    value = name.lstrip("#").strip().lower()
    return "variant" if value == "uploaded_variation" else value


def normalize_stable_id(value: str) -> str:
    value = value.strip()
    return value.split(".", 1)[0]


def natural_chromosome_key(chromosome: str) -> tuple[int, int | str, str]:
    match = _AUTOSOME_RE.fullmatch(chromosome)
    if match:
        return (0, int(match.group(1)), chromosome)
    bare = chromosome[3:] if chromosome.lower().startswith("chr") else chromosome
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    upper = bare.upper()
    if upper in special:
        return (1, special[upper], chromosome)
    return (2, upper, chromosome)


def candidate_sort_key(candidate: OrthrusCandidate) -> tuple[Any, ...]:
    return (
        natural_chromosome_key(candidate.chromosome),
        candidate.start,
        candidate.end,
        candidate.ref,
        candidate.alt,
        candidate.feature,
        candidate.gene,
        candidate.variant,
    )


def _parse_int(value: str, *, category: str, label: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise InputValidationError([ValidationIssue(category, f"{label}: {value!r}")]) from None


def read_orthrus_candidates(path: Path) -> tuple[list[OrthrusCandidate], InputInventory]:
    """Read input rows, partitioning supported SNVs from explicitly excluded rows."""

    issues: list[ValidationIssue] = []
    candidates: list[OrthrusCandidate] = []
    reasons: Counter[str] = Counter()
    all_variants: set[str] = set()
    ineligible_variants: set[str] = set()
    input_rows = 0
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise InputValidationError([ValidationIssue("missing_header", str(path))])
        headers = [normalize_header(name) for name in reader.fieldnames]
        duplicates = sorted(name for name, count in Counter(headers).items() if count > 1)
        missing = sorted(set(REQUIRED_COLUMNS) - set(headers))
        if duplicates:
            issues.append(ValidationIssue("duplicate_column", ", ".join(duplicates)))
        if missing:
            issues.append(ValidationIssue("missing_columns", ", ".join(missing)))
        if issues:
            raise InputValidationError(issues)

        for input_index, raw in enumerate(reader):
            input_rows += 1
            row = {normalize_header(key): (value or "").strip() for key, value in raw.items()}
            label = row.get("variant") or f"row {input_index + 2}"
            required_values = ("variant", "gene", "feature", "chromosome", "start", "end", "ref")
            empty = [name for name in required_values if row.get(name, "") in _MISSING]
            if empty:
                issues.append(ValidationIssue("missing_value", f"{label}: {', '.join(empty)}"))
                continue
            all_variants.add(row["variant"])
            try:
                start = int(row["start"])
                end = int(row["end"])
            except ValueError:
                issues.append(
                    ValidationIssue(
                        "invalid_coordinate", f"{label}: start={row['start']!r}, end={row['end']!r}"
                    )
                )
                continue

            ref, alt = row["ref"].upper(), row.get("alt", "").upper()
            cdna_raw = row.get("cdna_position", "")
            reason: str | None = None
            if cdna_raw in _MISSING:
                reason = INELIGIBLE_MISSING_CDNA
            elif len(ref) != 1 or ref not in NUCLEOTIDES:
                reason = INELIGIBLE_UNSUPPORTED_REF
            elif len(alt) != 1 or alt not in NUCLEOTIDES:
                reason = INELIGIBLE_UNSUPPORTED_ALT
            if reason is not None:
                reasons[reason] += 1
                ineligible_variants.add(row["variant"])
                continue
            try:
                cdna_position = int(cdna_raw)
            except ValueError:
                issues.append(
                    ValidationIssue("invalid_cdna_position", f"{label}: {cdna_raw!r}")
                )
                continue
            candidates.append(
                OrthrusCandidate(
                    variant=row["variant"],
                    gene=normalize_stable_id(row["gene"]),
                    feature=normalize_stable_id(row["feature"]),
                    chromosome=row["chromosome"],
                    start=start,
                    end=end,
                    cdna_position=cdna_position,
                    ref=ref,
                    alt=alt,
                    input_index=input_index,
                )
            )
    if issues:
        raise InputValidationError(issues)
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        duplicates = [repr(key) for key, count in Counter(keys).items() if count > 1]
        raise InputValidationError(
            [ValidationIssue("duplicate_eligible_row", value) for value in duplicates[:10]]
        )
    eligible_variants = {candidate.variant for candidate in candidates}
    inventory = InputInventory(
        input_rows=input_rows,
        input_unique_variants=len(all_variants),
        eligible_rows=len(candidates),
        eligible_unique_variants=len(eligible_variants),
        ineligible_rows=sum(reasons.values()),
        ineligible_unique_variants=len(ineligible_variants),
        ineligible_counts=dict(sorted(reasons.items())),
    )
    return candidates, inventory


def read_transcript_fasta(
    path: Path, required_transcripts: set[str]
) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    def commit() -> None:
        if current is None or current not in required_transcripts:
            return
        sequence = "".join(chunks).upper().replace("U", "T")
        previous = sequences.get(current)
        if previous is not None and previous != sequence:
            raise InputValidationError(
                [ValidationIssue("conflicting_fasta_record", current)]
            )
        sequences[current] = sequence

    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                token = line[1:].split(None, 1)[0].split("|", 1)[0]
                current = normalize_stable_id(token)
                chunks = []
            elif current is None:
                raise InputValidationError(
                    [ValidationIssue("invalid_fasta", f"sequence before header at line {line_number}")]
                )
            else:
                chunks.append(line)
        commit()
    missing = sorted(required_transcripts - set(sequences))
    if missing:
        raise InputValidationError(
            [ValidationIssue("missing_transcript_fasta", transcript) for transcript in missing]
        )
    return sequences


def _gff_attributes(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw.split(";"):
        if not item:
            continue
        key, separator, value = item.partition("=")
        if separator:
            result[unquote(key)] = unquote(value)
    return result


def _attribute_transcript(attributes: Mapping[str, str], *, record_type: str) -> str | None:
    raw = attributes.get("transcript_id")
    if raw is None and record_type in ("transcript", "mRNA", "lnc_RNA"):
        raw = attributes.get("ID")
    if raw is None:
        raw = attributes.get("Parent")
    if raw is None:
        return None
    raw = raw.split(",", 1)[0]
    if ":" in raw:
        prefix, value = raw.split(":", 1)
        if prefix in ("transcript", "rna", "mRNA"):
            raw = value
    return normalize_stable_id(raw)


@dataclass
class _GffTranscript:
    chromosome: str
    strand: str
    gene: str | None
    exons: list[tuple[int, int]]
    cds: list[tuple[int, int]]


def read_transcript_annotations(
    gff3_path: Path,
    fasta_path: Path,
    required_transcripts: set[str],
) -> dict[str, TranscriptAnnotation]:
    """Project GFF3 exon/CDS features into transcript-oriented six-track inputs."""

    records: dict[str, _GffTranscript] = {}
    features: dict[str, dict[str, list[tuple[int, int, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    issues: list[ValidationIssue] = []
    with _open_text(gff3_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                issues.append(ValidationIssue("invalid_gff3_record", f"line {line_number}"))
                continue
            record_type = fields[2]
            if record_type not in ("transcript", "mRNA", "lnc_RNA", "exon", "CDS"):
                continue
            attributes = _gff_attributes(fields[8])
            transcript = _attribute_transcript(attributes, record_type=record_type)
            if transcript is None or transcript not in required_transcripts:
                continue
            chromosome, strand = fields[0], fields[6]
            try:
                start, end = int(fields[3]) - 1, int(fields[4])
            except ValueError:
                issues.append(ValidationIssue("invalid_gff3_coordinate", f"line {line_number}"))
                continue
            if start < 0 or end <= start or strand not in ("+", "-"):
                issues.append(ValidationIssue("invalid_gff3_feature", f"line {line_number}"))
                continue
            if record_type in ("transcript", "mRNA", "lnc_RNA"):
                raw_gene = attributes.get("gene_id") or attributes.get("gene") or attributes.get("Parent")
                if raw_gene and ":" in raw_gene:
                    raw_gene = raw_gene.split(":", 1)[1]
                gene = None if raw_gene is None else normalize_stable_id(raw_gene)
                new = _GffTranscript(chromosome, strand, gene, [], [])
                previous = records.get(transcript)
                if previous is not None and (
                    previous.chromosome, previous.strand, previous.gene
                ) != (new.chromosome, new.strand, new.gene):
                    issues.append(ValidationIssue("conflicting_transcript_record", transcript))
                else:
                    records[transcript] = new
            else:
                features[transcript][record_type].append((start, end, chromosome, strand))
    if issues:
        raise InputValidationError(issues)
    sequences = read_transcript_fasta(fasta_path, required_transcripts)
    missing_records = sorted(required_transcripts - set(records))
    if missing_records:
        raise InputValidationError(
            [ValidationIssue("missing_transcript_gff3", transcript) for transcript in missing_records]
        )

    annotations: dict[str, TranscriptAnnotation] = {}
    issues = []
    for transcript in sorted(required_transcripts):
        record = records[transcript]
        exon_rows = features[transcript].get("exon", [])
        cds_rows = features[transcript].get("CDS", [])
        if not exon_rows:
            issues.append(ValidationIssue("missing_exons", transcript))
            continue
        if any(chrom != record.chromosome or strand != record.strand for _, _, chrom, strand in exon_rows + cds_rows):
            issues.append(ValidationIssue("inconsistent_feature_parent", transcript))
            continue
        exons_genomic = sorted((start, end) for start, end, _, _ in exon_rows)
        if any(left[1] > right[0] for left, right in zip(exons_genomic, exons_genomic[1:])):
            issues.append(ValidationIssue("overlapping_exons", transcript))
            continue
        exons = tuple(exons_genomic if record.strand == "+" else reversed(exons_genomic))
        length = sum(end - start for start, end in exons)
        sequence = sequences[transcript]
        if len(sequence) != length:
            issues.append(
                ValidationIssue(
                    "transcript_length_mismatch",
                    f"{transcript}: FASTA={len(sequence)}, exons={length}",
                )
            )
            continue

        splice = [0] * length
        cumulative = 0
        for start, end in exons:
            cumulative += end - start
            splice[cumulative - 1] = 1

        exon_offsets: list[tuple[int, int, int]] = []
        cursor = 0
        for exon_start, exon_end in exons:
            exon_offsets.append((exon_start, exon_end, cursor))
            cursor += exon_end - exon_start
        projected_cds: list[tuple[int, int]] = []
        cds_bases = 0
        for cds_start, cds_end, _, _ in cds_rows:
            cds_bases += cds_end - cds_start
            for exon_start, exon_end, offset in exon_offsets:
                intersection_start = max(cds_start, exon_start)
                intersection_end = min(cds_end, exon_end)
                if intersection_start >= intersection_end:
                    continue
                if record.strand == "+":
                    projected_start = offset + intersection_start - exon_start
                    projected_end = offset + intersection_end - exon_start
                else:
                    projected_start = offset + exon_end - intersection_end
                    projected_end = offset + exon_end - intersection_start
                projected_cds.append((projected_start, projected_end))
        projected_cds.sort()
        projected_bases = sum(end - start for start, end in projected_cds)
        has_overlap = any(left[1] > right[0] for left, right in zip(projected_cds, projected_cds[1:]))
        if projected_bases != cds_bases or has_overlap:
            issues.append(ValidationIssue("cds_outside_or_overlapping_exons", transcript))
            continue
        if any(left[1] != right[0] for left, right in zip(projected_cds, projected_cds[1:])):
            issues.append(ValidationIssue("noncontiguous_spliced_cds", transcript))
            continue
        cds_track = [0] * length
        if projected_cds:
            cds_start = projected_cds[0][0]
            cds_end = projected_cds[-1][1]
            for position in range(cds_start, cds_end, 3):
                cds_track[position] = 1
        annotations[transcript] = TranscriptAnnotation(
            transcript=transcript,
            gene=record.gene,
            chromosome=record.chromosome,
            strand=record.strand,
            sequence=sequence,
            exons=exons,
            cds_track=tuple(cds_track),
            splice_track=tuple(splice),
        )
    if issues:
        raise InputValidationError(issues)
    return annotations


def _same_chromosome(left: str, right: str) -> bool:
    def bare(value: str) -> str:
        return value[3:] if value.lower().startswith("chr") else value

    return bare(left).upper().replace("MT", "M") == bare(right).upper().replace("MT", "M")


def prepare_requests(
    candidates: Sequence[OrthrusCandidate],
    annotations: Mapping[str, TranscriptAnnotation],
) -> list[PreparedRequest]:
    issues: list[ValidationIssue] = []
    requests: list[PreparedRequest] = []
    for candidate in candidates:
        annotation = annotations.get(candidate.feature)
        if annotation is None:
            issues.append(ValidationIssue("missing_transcript", candidate.feature))
            continue
        label = f"{candidate.variant}/{candidate.gene}/{candidate.feature}"
        if annotation.gene is not None and annotation.gene != candidate.gene:
            issues.append(
                ValidationIssue(
                    "transcript_gene_mismatch",
                    f"{label}: input={candidate.gene}, GFF3={annotation.gene}",
                )
            )
        if not _same_chromosome(candidate.chromosome, annotation.chromosome):
            issues.append(
                ValidationIssue(
                    "transcript_chromosome_mismatch",
                    f"{label}: input={candidate.chromosome}, GFF3={annotation.chromosome}",
                )
            )
        if candidate.end != candidate.start + 1 or candidate.start < 0:
            issues.append(
                ValidationIssue("not_snv_interval", f"{label}: [{candidate.start}, {candidate.end})")
            )
        if candidate.ref == candidate.alt:
            issues.append(
                ValidationIssue("identical_alleles", f"{label}: {candidate.ref}>{candidate.alt}")
            )
        if not 1 <= candidate.cdna_position <= len(annotation.sequence):
            issues.append(
                ValidationIssue(
                    "cdna_position_out_of_bounds",
                    f"{label}: {candidate.cdna_position}, length={len(annotation.sequence)}",
                )
            )
            continue
        observed_position = annotation.genomic_position(candidate.target_index)
        if observed_position != candidate.start:
            issues.append(
                ValidationIssue(
                    "cdna_genomic_position_mismatch",
                    f"{label}: input={candidate.start}, GFF3={observed_position}",
                )
            )
        transcript_ref = (
            candidate.ref if annotation.strand == "+" else candidate.ref.translate(COMPLEMENT)
        )
        transcript_alt = (
            candidate.alt if annotation.strand == "+" else candidate.alt.translate(COMPLEMENT)
        )
        observed_ref = annotation.sequence[candidate.target_index]
        if observed_ref != transcript_ref:
            issues.append(
                ValidationIssue(
                    "transcript_reference_mismatch",
                    f"{label}: input={transcript_ref}, FASTA={observed_ref}",
                )
            )
        requests.append(PreparedRequest(candidate, annotation, transcript_ref, transcript_alt))
    if issues:
        raise InputValidationError(issues)
    return requests


def build_six_track_input(
    annotation: TranscriptAnnotation, target_index: int, *, prefix_only: bool = True
) -> list[list[float]]:
    end = target_index + 1 if prefix_only else len(annotation.sequence)
    result: list[list[float]] = []
    for index, base in enumerate(annotation.sequence[:end]):
        row = [0.0] * 6
        if index != target_index and base in NUCLEOTIDE_INDEX:
            row[NUCLEOTIDE_INDEX[base]] = 1.0
        row[4] = float(annotation.cds_track[index])
        row[5] = float(annotation.splice_track[index])
        result.append(row)
    return result


class OrthrusModel:
    """Thin inference-only adapter around the local Hugging Face checkpoint."""

    def __init__(self, checkpoint: Path, manifest_path: Path, device: str = "cuda") -> None:
        validate_orthrus_checkpoint(checkpoint, manifest_path)
        import torch
        from transformers import AutoModel

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.model = AutoModel.from_pretrained(
            str(checkpoint.resolve()),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(self.device, dtype=torch.float32).eval()
        config = self.model.config
        if config.n_tracks != 6 or not config.has_mlm_head or config.mlm_head_dim != 4:
            raise ValueError("Orthrus checkpoint must have six tracks and a four-class MLM head")

    def score_requests(
        self,
        requests: Sequence[PreparedRequest],
        *,
        batch_size: int,
        prefix_only: bool = True,
    ) -> Mapping[tuple[Any, ...], ComponentScores]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        torch = self._torch
        results: dict[tuple[Any, ...], ComponentScores] = {}
        ordered = sorted(
            requests,
            key=lambda request: (
                request.candidate.target_index + 1 if prefix_only else len(request.annotation.sequence),
                candidate_sort_key(request.candidate),
            ),
        )
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset : offset + batch_size]
            arrays = [
                build_six_track_input(
                    request.annotation, request.candidate.target_index, prefix_only=prefix_only
                )
                for request in batch
            ]
            lengths = torch.tensor([len(array) for array in arrays], dtype=torch.long, device=self.device)
            max_length = int(lengths.max().item())
            inputs = torch.zeros((len(batch), max_length, 6), dtype=torch.float32, device=self.device)
            for index, array in enumerate(arrays):
                inputs[index, : len(array)] = torch.tensor(array, dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                logits = self.model.predict_tokens(inputs, lengths, channel_last=True)
                target_positions = torch.tensor(
                    [request.candidate.target_index for request in batch],
                    dtype=torch.long,
                    device=self.device,
                )
                selected = logits[torch.arange(len(batch), device=self.device), target_positions]
                log_probabilities = torch.log_softmax(selected.float(), dim=-1).cpu()
            for index, request in enumerate(batch):
                values = log_probabilities[index]
                scores = ComponentScores(
                    float(values[NUCLEOTIDE_INDEX[request.transcript_ref]].item()),
                    float(values[NUCLEOTIDE_INDEX[request.transcript_alt]].item()),
                )
                if not all(math.isfinite(value) for value in (scores.ref_log_probability, scores.alt_log_probability, scores.score)):
                    raise RuntimeError(f"model returned non-finite scores for {request.key!r}")
                results[request.key] = scores
        return results


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_orthrus_checkpoint(checkpoint: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model") != MODEL_NAME:
        raise ValueError(f"checkpoint manifest must identify {MODEL_NAME!r}")
    expected_revision = manifest.get("revision")
    if not isinstance(expected_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise ValueError("checkpoint manifest has an invalid revision")
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkpoint), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot verify checkpoint Git revision: {checkpoint}") from error
    if revision != expected_revision:
        raise ValueError(f"checkpoint revision mismatch: expected {expected_revision}, observed {revision}")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("checkpoint manifest has no file hashes")
    for relative, expected_hash in sorted(hashes.items()):
        path = checkpoint / relative
        if not path.is_file():
            raise ValueError(f"checkpoint file is missing: {path}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ValueError(
                f"checkpoint hash mismatch for {relative}: expected {expected_hash}, observed {observed}"
            )
    config = json.loads((checkpoint / "config.json").read_text())
    if (
        config.get("n_tracks") != 6
        or config.get("has_mlm_head") is not True
        or config.get("mlm_head_dim") != 4
    ):
        raise ValueError("Orthrus checkpoint must have six tracks and a four-class MLM head")
    expected = manifest.get("expected", {})
    for key in ("n_tracks", "has_mlm_head", "mlm_head_dim"):
        if key in expected and config.get(key) != expected[key]:
            raise ValueError(f"checkpoint config mismatch for {key}")
    return {
        "model": MODEL_NAME,
        "repository": manifest.get("repository"),
        "revision": revision,
        "sha256": dict(sorted(hashes.items())),
    }


def shard_candidates(
    candidates: Sequence[OrthrusCandidate], *, num_shards: int, shard_index: int
) -> list[OrthrusCandidate]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    ordered = sorted(candidates, key=candidate_sort_key)
    start = len(ordered) * shard_index // num_shards
    end = len(ordered) * (shard_index + 1) // num_shards
    return ordered[start:end]


def shard_paths(output_dir: Path, num_shards: int, shard_index: int) -> tuple[Path, Path]:
    stem = f"shard-{shard_index:05d}-of-{num_shards:05d}"
    return output_dir / f"{stem}.tsv.gz", output_dir / f"{stem}.json"


def _range_key(candidate: OrthrusCandidate) -> list[Any]:
    chromosome = natural_chromosome_key(candidate.chromosome)
    return [list(chromosome), *candidate_sort_key(candidate)[1:]]


def expected_range(candidates: Sequence[OrthrusCandidate]) -> dict[str, Any]:
    return {
        "first": None if not candidates else _range_key(candidates[0]),
        "last": None if not candidates else _range_key(candidates[-1]),
    }


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


def _read_output(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ShardCompatibilityError(f"output schema mismatch: {path}")
        return list(reader)


def _candidate_key_from_output(row: Mapping[str, str]) -> tuple[Any, ...]:
    try:
        return (
            row["variant"],
            normalize_stable_id(row["gene"]),
            normalize_stable_id(row["feature"]),
            row["chromosome"],
            int(row["start"]),
            int(row["end"]),
            int(row["cdna_position"]),
            row["ref"],
            row["alt"],
        )
    except (KeyError, ValueError) as error:
        raise ShardCompatibilityError(f"invalid output row: {row!r}") from error


def _identity(
    variants_path: Path,
    fasta_path: Path,
    gff3_path: Path,
    checkpoint_identity: Mapping[str, Any],
    num_shards: int,
) -> dict[str, Any]:
    return {
        "variants": file_identity(variants_path),
        "transcript_fasta": file_identity(fasta_path),
        "gff3": file_identity(gff3_path),
        "checkpoint": checkpoint_identity,
        "num_shards": num_shards,
    }


def _counts(inventory: InputInventory, selected: Sequence[OrthrusCandidate]) -> dict[str, int]:
    return {
        "input_rows": inventory.input_rows,
        "input_unique_variants": inventory.input_unique_variants,
        "eligible_rows": inventory.eligible_rows,
        "eligible_unique_variants": inventory.eligible_unique_variants,
        "ineligible_rows": inventory.ineligible_rows,
        "ineligible_unique_variants": inventory.ineligible_unique_variants,
        "output_rows": len(selected),
        "output_unique_variants": len({candidate.variant for candidate in selected}),
    }


def _default_model_factory(checkpoint: Path, manifest_path: Path, device: str) -> RequestScorer:
    return OrthrusModel(checkpoint, manifest_path, device)


def score_orthrus_variants(
    *,
    variants_path: Path,
    transcript_fasta_path: Path,
    gff3_path: Path,
    checkpoint: Path,
    manifest_path: Path,
    output_dir: Path,
    num_shards: int = 1,
    shard_index: int = 0,
    device: str = "cuda",
    batch_size: int = 32,
    model_factory: ModelFactory = _default_model_factory,
) -> OrthrusScoringSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    checkpoint_identity = validate_orthrus_checkpoint(checkpoint, manifest_path)
    candidates, inventory = read_orthrus_candidates(variants_path)
    selected = shard_candidates(candidates, num_shards=num_shards, shard_index=shard_index)
    required = {candidate.feature for candidate in selected}
    annotations = read_transcript_annotations(gff3_path, transcript_fasta_path, required)
    requests = prepare_requests(selected, annotations)
    identity = _identity(
        variants_path, transcript_fasta_path, gff3_path, checkpoint_identity, num_shards
    )
    output_path, metadata_path = shard_paths(output_dir, num_shards, shard_index)
    shard = {
        "index": shard_index,
        "count": num_shards,
        "expected_genomic_range": expected_range(selected),
    }
    if output_path.exists() or metadata_path.exists():
        if not output_path.is_file() or not metadata_path.is_file():
            raise ShardCompatibilityError(f"incomplete existing shard: {output_path}, {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("identity") != identity
            or metadata.get("shard") != shard
            or metadata.get("counts") != _counts(inventory, selected)
            or metadata.get("ineligible_counts") != dict(inventory.ineligible_counts)
        ):
            raise ShardCompatibilityError(f"existing shard metadata is incompatible: {metadata_path}")
        if metadata.get("output", {}).get("sha256") != sha256_file(output_path):
            raise ShardCompatibilityError(f"existing shard output checksum mismatch: {output_path}")
        return OrthrusScoringSummary(output_path, metadata_path, len(selected), True)

    scorer = model_factory(checkpoint, manifest_path, device)
    scores = scorer.score_requests(requests, batch_size=batch_size)
    expected_keys = {request.key for request in requests}
    if set(scores) != expected_keys:
        missing = list(expected_keys - set(scores))[:3]
        extra = list(set(scores) - expected_keys)[:3]
        raise RuntimeError(f"model score key mismatch: missing={missing!r}, extra={extra!r}")
    rows: list[dict[str, Any]] = []
    for request in requests:
        candidate = request.candidate
        score = scores[request.key]
        rows.append(
            {
                "variant": candidate.variant,
                "gene": candidate.gene,
                "feature": candidate.feature,
                "chromosome": candidate.chromosome,
                "start": candidate.start,
                "end": candidate.end,
                "cdna_position": candidate.cdna_position,
                "ref": candidate.ref,
                "alt": candidate.alt,
                "strand": request.annotation.strand,
                "transcript_ref": request.transcript_ref,
                "transcript_alt": request.transcript_alt,
                "model": MODEL_NAME,
                "ref_log_probability": score.ref_log_probability,
                "alt_log_probability": score.alt_log_probability,
                "score": score.score,
            }
        )
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "shard": shard,
        "counts": _counts(inventory, selected),
        "ineligible_counts": dict(inventory.ineligible_counts),
        "output": {"columns": list(OUTPUT_COLUMNS), "rows": len(rows)},
    }
    _atomic_write_rows(output_path, rows)
    metadata["output"]["sha256"] = sha256_file(output_path)
    _atomic_write_json(metadata_path, metadata)
    return OrthrusScoringSummary(output_path, metadata_path, len(rows))


def collate_orthrus_scores(
    *,
    variants_path: Path,
    transcript_fasta_path: Path,
    gff3_path: Path,
    checkpoint: Path,
    manifest_path: Path,
    output_dir: Path,
    output_path: Path,
    num_shards: int,
) -> tuple[Path, Path]:
    checkpoint_identity = validate_orthrus_checkpoint(checkpoint, manifest_path)
    candidates, inventory = read_orthrus_candidates(variants_path)
    ordered = sorted(candidates, key=candidate_sort_key)
    annotations = read_transcript_annotations(
        gff3_path, transcript_fasta_path, {candidate.feature for candidate in ordered}
    )
    prepare_requests(ordered, annotations)
    identity = _identity(
        variants_path, transcript_fasta_path, gff3_path, checkpoint_identity, num_shards
    )
    rows_by_key: dict[tuple[Any, ...], dict[str, str]] = {}
    for shard_index in range(num_shards):
        expected = shard_candidates(ordered, num_shards=num_shards, shard_index=shard_index)
        shard_output, shard_metadata_path = shard_paths(output_dir, num_shards, shard_index)
        if not shard_output.is_file() or not shard_metadata_path.is_file():
            raise ShardCompatibilityError(f"missing shard {shard_index}: {shard_output}")
        metadata = json.loads(shard_metadata_path.read_text())
        expected_shard = {
            "index": shard_index,
            "count": num_shards,
            "expected_genomic_range": expected_range(expected),
        }
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ShardCompatibilityError(f"schema version mismatch: {shard_metadata_path}")
        if metadata.get("identity") != identity or metadata.get("shard") != expected_shard:
            raise ShardCompatibilityError(f"incompatible shard metadata: {shard_metadata_path}")
        if metadata.get("counts") != _counts(inventory, expected):
            raise ShardCompatibilityError(f"shard count mismatch: {shard_metadata_path}")
        if metadata.get("ineligible_counts") != dict(inventory.ineligible_counts):
            raise ShardCompatibilityError(f"ineligible count mismatch: {shard_metadata_path}")
        if metadata.get("output", {}).get("sha256") != sha256_file(shard_output):
            raise ShardCompatibilityError(f"shard output checksum mismatch: {shard_output}")
        rows = _read_output(shard_output)
        if len(rows) != len(expected):
            raise ShardCompatibilityError(f"shard row count mismatch: {shard_output}")
        for row in rows:
            key = _candidate_key_from_output(row)
            if key in rows_by_key:
                raise ShardCompatibilityError(f"duplicate row across shards: {key!r}")
            rows_by_key[key] = row

    expected_keys = [candidate.key for candidate in ordered]
    missing = [key for key in expected_keys if key not in rows_by_key]
    extra = list(set(rows_by_key) - set(expected_keys))
    if missing or extra:
        raise ShardCompatibilityError(
            f"final row coverage mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}"
        )
    final_rows = [rows_by_key[key] for key in expected_keys]
    _atomic_write_rows(output_path, final_rows)
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "counts": {
            **_counts(inventory, ordered),
            "shards": num_shards,
        },
        "ineligible_counts": dict(inventory.ineligible_counts),
        "output": {
            "columns": list(OUTPUT_COLUMNS),
            "rows": len(final_rows),
            "sha256": sha256_file(output_path),
        },
    }
    _atomic_write_json(metadata_path, metadata)
    return output_path, metadata_path
