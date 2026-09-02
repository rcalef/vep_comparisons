"""Orthrus six-track transcript annotation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..common import (
    TranscriptAnnotation,
    gff_attributes,
    normalize_stable_id,
    open_text,
    read_transcript_fasta,
    transcript_id,
)


@dataclass
class _GffTranscript:
    chromosome: str
    strand: str
    gene: str | None


def _gene(attributes: dict[str, str]) -> str | None:
    raw = attributes.get("gene_id") or attributes.get("gene") or attributes.get("Parent")
    if raw and ":" in raw:
        raw = raw.split(":", 1)[1]
    return None if raw is None else normalize_stable_id(raw)


def read_transcript_annotations(
    gff3_path: Path,
    fasta_path: Path,
    required_transcripts: set[str],
) -> dict[str, TranscriptAnnotation]:
    records: dict[str, _GffTranscript] = {}
    features: dict[str, dict[str, list[tuple[int, int, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open_text(gff3_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"invalid_gff3_record: line {line_number}")
            record_type = fields[2]
            if record_type not in ("transcript", "mRNA", "lnc_RNA", "exon", "CDS"):
                continue
            attributes = gff_attributes(fields[8])
            transcript = transcript_id(attributes, record_type)
            if transcript is None or transcript not in required_transcripts:
                continue
            try:
                start, end = int(fields[3]) - 1, int(fields[4])
            except ValueError:
                raise ValueError(f"invalid_gff3_coordinate: line {line_number}") from None
            chromosome, strand = fields[0], fields[6]
            if start < 0 or end <= start or strand not in ("+", "-"):
                raise ValueError(f"invalid_gff3_feature: line {line_number}")
            if record_type in ("transcript", "mRNA", "lnc_RNA"):
                record = _GffTranscript(chromosome, strand, _gene(attributes))
                if transcript in records and records[transcript] != record:
                    raise ValueError(f"conflicting_transcript_record: {transcript}")
                records[transcript] = record
            else:
                features[transcript][record_type].append(
                    (start, end, chromosome, strand)
                )

    sequences = read_transcript_fasta(fasta_path, required_transcripts)
    missing = sorted(required_transcripts - set(records))
    if missing:
        raise ValueError(f"missing_transcript_gff3: {missing[0]}")

    annotations: dict[str, TranscriptAnnotation] = {}
    for transcript in sorted(required_transcripts):
        record = records[transcript]
        exon_rows = features[transcript].get("exon", [])
        cds_rows = features[transcript].get("CDS", [])
        if not exon_rows:
            raise ValueError(f"missing_exons: {transcript}")
        if any(
            chromosome != record.chromosome or strand != record.strand
            for _, _, chromosome, strand in exon_rows + cds_rows
        ):
            raise ValueError(f"inconsistent_feature_parent: {transcript}")
        genomic_exons = sorted((start, end) for start, end, _, _ in exon_rows)
        if any(
            left[1] > right[0]
            for left, right in zip(genomic_exons, genomic_exons[1:])
        ):
            raise ValueError(f"overlapping_exons: {transcript}")
        exons = tuple(
            genomic_exons if record.strand == "+" else reversed(genomic_exons)
        )
        sequence = sequences[transcript]
        length = sum(end - start for start, end in exons)
        if len(sequence) != length:
            raise ValueError(
                f"transcript_length_mismatch: {transcript}: "
                f"FASTA={len(sequence)}, exons={length}"
            )

        splice = [0] * length
        cursor = 0
        exon_offsets: list[tuple[int, int, int]] = []
        for start, end in exons:
            exon_offsets.append((start, end, cursor))
            cursor += end - start
            splice[cursor - 1] = 1

        projected_cds: list[tuple[int, int]] = []
        cds_bases = sum(end - start for start, end, _, _ in cds_rows)
        for cds_start, cds_end, _, _ in cds_rows:
            for exon_start, exon_end, offset in exon_offsets:
                intersection_start = max(cds_start, exon_start)
                intersection_end = min(cds_end, exon_end)
                if intersection_start >= intersection_end:
                    continue
                if record.strand == "+":
                    projected = (
                        offset + intersection_start - exon_start,
                        offset + intersection_end - exon_start,
                    )
                else:
                    projected = (
                        offset + exon_end - intersection_end,
                        offset + exon_end - intersection_start,
                    )
                projected_cds.append(projected)
        projected_cds.sort()
        if (
            sum(end - start for start, end in projected_cds) != cds_bases
            or any(
                left[1] > right[0]
                for left, right in zip(projected_cds, projected_cds[1:])
            )
        ):
            raise ValueError(f"cds_outside_or_overlapping_exons: {transcript}")
        if any(
            left[1] != right[0]
            for left, right in zip(projected_cds, projected_cds[1:])
        ):
            raise ValueError(f"noncontiguous_spliced_cds: {transcript}")

        cds_track = [0] * length
        if projected_cds:
            for position in range(projected_cds[0][0], projected_cds[-1][1], 3):
                cds_track[position] = 1
        annotations[transcript] = TranscriptAnnotation(
            transcript,
            record.gene,
            record.chromosome,
            record.strand,
            sequence,
            exons,
            tuple(cds_track),
            tuple(splice),
        )
    return annotations


__all__ = ["TranscriptAnnotation", "read_transcript_annotations"]
