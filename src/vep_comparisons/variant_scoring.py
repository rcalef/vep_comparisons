"""Validation and orchestration for zero-shot protein variant scoring."""

from __future__ import annotations

import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pysam

from .variant_models import (
    MODEL_REGISTRY,
    ModelSpec,
    PositionRequest,
    VariantModel,
    load_variant_model,
)


REQUIRED_VARIANT_COLUMNS = (
    "variant",
    "gene",
    "feature",
    "consequence",
    "protein_position",
    "amino_acids",
    "biotype",
)
OUTPUT_COLUMNS = (
    "variant",
    "gene",
    "feature",
    "protein_position",
    "amino_acids",
    "model",
    "score",
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


@dataclass(frozen=True)
class ValidationIssue:
    category: str
    detail: str


class InputValidationError(ValueError):
    """All input failures, summarized without silently dropping any candidate."""

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


@dataclass(frozen=True)
class ScoringSummary:
    model: str
    candidates: int
    unique_positions: int
    scored: int
    null: int
    output_path: Path
    runtime_seconds: float


def normalize_transcript(value: str) -> str:
    """Extract a versionless transcript ID from a FASTA or variant value."""

    identifiers = value.split(maxsplit=1)[0].split("|")
    identifier = next(
        (identifier for identifier in identifiers if identifier.startswith("ENST")),
        identifiers[0],
    )
    return identifier.split(".", maxsplit=1)[0]


def read_transcript_fasta(path: Path) -> dict[str, str]:
    """Read FASTA records keyed by versionless transcript ID."""

    records: dict[str, str] = {}
    with pysam.FastxFile(path) as fasta:
        for record in fasta:
            transcript = normalize_transcript(record.name)
            if transcript in records:
                raise InputValidationError(
                    [ValidationIssue("duplicate_transcript", transcript)]
                )
            records[transcript] = record.sequence

    return records


def read_variant_candidates(path: Path) -> list[Candidate]:
    """Select protein-coding missense rows and parse their substitutions."""

    frame = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        infer_schema=False,
        null_values="-",
    )
    rename = {}
    for column in frame.columns:
        normalized = column.lstrip("#").lower()
        rename[column] = "variant" if normalized == "uploaded_variation" else normalized
    frame = frame.rename(rename)
    frame = (
        frame
        .filter(
            pl.col("biotype") == "protein_coding",
            pl.col("consequence")
            .str.split(",")
            .list.contains("missense_variant"),
        )
        .select(REQUIRED_VARIANT_COLUMNS)
        .with_columns(pl.col("protein_position").cast(pl.Int64))
    )

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
) -> list[ValidationIssue]:
    """Validate every eligible row, including rows too long to score."""

    issues: list[ValidationIssue] = []
    required_transcripts = sorted({candidate.transcript for candidate in candidates})
    for transcript in required_transcripts:
        sequence = sequences.get(transcript)
        if sequence is None:
            issues.append(ValidationIssue("missing_sequence", transcript))
            continue
        if require_structure:
            if structure_tokens is None or transcript not in structure_tokens:
                issues.append(ValidationIssue("missing_structure_tokens", transcript))
            else:
                tokens = structure_tokens[transcript]
                if len(tokens) != len(sequence):
                    issues.append(
                        ValidationIssue(
                            "sequence_structure_length_mismatch",
                            f"{transcript}: sequence={len(sequence)}, 3Di={len(tokens)}",
                        )
                    )

    for candidate in candidates:
        sequence = sequences.get(candidate.transcript)
        if sequence is None:
            continue
        if not 1 <= candidate.position <= len(sequence):
            issues.append(
                ValidationIssue(
                    "position_out_of_range",
                    f"variant={candidate.variant}, transcript={candidate.transcript}, "
                    f"position={candidate.position}, length={len(sequence)}",
                )
            )
        elif sequence[candidate.position - 1].upper() != candidate.ref:
            issues.append(
                ValidationIssue(
                    "reference_mismatch",
                    f"variant={candidate.variant}, transcript={candidate.transcript}, "
                    f"position={candidate.position}, VEP={candidate.ref}, "
                    f"FASTA={sequence[candidate.position - 1]!r}",
                )
            )
    return issues


def build_position_requests(
    candidates: Sequence[Candidate],
    sequences: Mapping[str, str],
    *,
    structure_tokens: Mapping[str, str] | None,
    max_sequence_length: int,
) -> list[PositionRequest]:
    seen: set[tuple[str, int]] = set()
    requests: list[PositionRequest] = []
    for candidate in candidates:
        key = (candidate.transcript, candidate.position)
        sequence = sequences[candidate.transcript]
        if key in seen or len(sequence) > max_sequence_length:
            continue
        seen.add(key)
        requests.append(
            PositionRequest(
                transcript=candidate.transcript,
                sequence=sequence,
                position=candidate.position,
                structure_tokens=(
                    structure_tokens[candidate.transcript]
                    if structure_tokens is not None
                    else None
                ),
            )
        )
    return requests


ModelFactory = Callable[[ModelSpec, Path, str, str], VariantModel]

def _default_model_factory(
    spec: ModelSpec, model_root: Path, device: str, dtype: str
) -> VariantModel:
    return load_variant_model(spec, model_root, device=device, dtype=dtype)


def score_protein_variants(
    *,
    variants_path: Path,
    sequences_path: Path,
    model_name: str,
    model_root: Path,
    output: Path,
    structure_tokens_path: Path | None = None,
    device: str = "cuda",
    dtype: str = "float32",
    max_sequence_length: int | None = None,
    batch_size: int = 1,
    model_factory: ModelFactory = _default_model_factory,
) -> ScoringSummary:
    """Validate, score one checkpoint, and atomically publish one table."""

    started = time.monotonic()
    spec = MODEL_REGISTRY[model_name]
    limit = spec.capacity if max_sequence_length is None else max_sequence_length
    if limit < 1 or limit > spec.capacity:
        raise InputValidationError(
            [
                ValidationIssue(
                    "invalid_max_sequence_length",
                    f"requested={limit}, supported=1..{spec.capacity} for {model_name}",
                )
            ]
        )
    if batch_size < 1:
        raise InputValidationError(
            [
                ValidationIssue("invalid_batch_size", str(batch_size))
            ]
        )
    require_structure = spec.family == "saprot"
    if require_structure and structure_tokens_path is None:
        raise InputValidationError(
            [
                ValidationIssue(
                    "missing_structure_tokens_argument",
                    f"--structure-tokens is required for {model_name}",
                )
            ]
        )

    candidates = read_variant_candidates(variants_path)
    sequences = read_transcript_fasta(sequences_path)
    structures = (
        read_transcript_fasta(structure_tokens_path)
        if require_structure and structure_tokens_path is not None
        else None
    )
    issues = validate_candidates(
        candidates,
        sequences,
        structure_tokens=structures,
        require_structure=require_structure,
    )
    if issues:
        raise InputValidationError(issues)

    requests = build_position_requests(
        candidates,
        sequences,
        structure_tokens=structures,
        max_sequence_length=limit,
    )
    print(
        f"model={model_name} candidates={len(candidates)} "
        f"unique_scoreable_positions={len(requests)} max_length={limit} "
        f"batch_size={batch_size}",
        file=sys.stderr,
    )
    print(
        f"validation=passed transcripts={len({c.transcript for c in candidates})}",
        file=sys.stderr,
    )

    model = model_factory(spec, model_root, device, dtype)
    position_scores = model.score_positions(requests, batch_size=batch_size)

    scores: list[float | None] = []
    for candidate in candidates:
        if len(sequences[candidate.transcript]) > limit:
            scores.append(None)
            continue
        amino_acid_scores = position_scores[(candidate.transcript, candidate.position)]
        scores.append(amino_acid_scores[candidate.alt] - amino_acid_scores[candidate.ref])

    output_frame = (
        pl.DataFrame({
            "variant": [candidate.variant for candidate in candidates],
            "gene": [candidate.gene for candidate in candidates],
            "feature": [candidate.feature for candidate in candidates],
            "protein_position": [candidate.position for candidate in candidates],
            "amino_acids": [candidate.amino_acids for candidate in candidates],
            "model": [model_name] * len(candidates),
            "score": pl.Series(scores, dtype=pl.Float64),
        })
        .select(OUTPUT_COLUMNS)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.write_csv(output, separator="\t", compression="gzip")

    runtime = time.monotonic() - started
    scored = sum(score is not None for score in scores)
    summary = ScoringSummary(
        model=model_name,
        candidates=len(candidates),
        unique_positions=len(requests),
        scored=scored,
        null=len(scores) - scored,
        output_path=output,
        runtime_seconds=runtime,
    )
    print(
        f"scored={summary.scored} null={summary.null} "
        f"runtime_seconds={runtime:.2f} output={output}",
        file=sys.stderr,
    )
    return summary
