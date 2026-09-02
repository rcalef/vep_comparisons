"""RiNALMo transcript and exon annotation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..common import (
    TranscriptAnnotation,
    gff_attributes,
    normalize_stable_id,
    open_text,
    read_transcript_fasta,
    transcript_id,
)


def read_rinalmo_annotations(
    gff3_path: Path, fasta_path: Path, required_transcripts: set[str]
) -> dict[str, TranscriptAnnotation]:
    records: dict[str, tuple[str, str, str | None]] = {}
    exons: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    with open_text(gff3_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"invalid_gff3_record: line {line_number}")
            record_type = fields[2]
            if record_type not in ("transcript", "mRNA", "lnc_RNA", "exon"):
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
            if record_type == "exon":
                exons[transcript].append((start, end, chromosome, strand))
                continue
            raw_gene = (
                attributes.get("gene_id")
                or attributes.get("gene")
                or attributes.get("Parent")
            )
            if raw_gene and ":" in raw_gene:
                raw_gene = raw_gene.split(":", 1)[1]
            record = (
                chromosome,
                strand,
                None if raw_gene is None else normalize_stable_id(raw_gene),
            )
            if transcript in records and records[transcript] != record:
                raise ValueError(f"conflicting_transcript_record: {transcript}")
            records[transcript] = record

    sequences = read_transcript_fasta(fasta_path, required_transcripts)
    missing = sorted(required_transcripts - set(records))
    if missing:
        raise ValueError(f"missing_transcript_gff3: {missing[0]}")

    result: dict[str, TranscriptAnnotation] = {}
    for transcript in sorted(required_transcripts):
        chromosome, strand, gene = records[transcript]
        rows = exons[transcript]
        if not rows:
            raise ValueError(f"missing_exons: {transcript}")
        if any(
            row_chromosome != chromosome or row_strand != strand
            for _, _, row_chromosome, row_strand in rows
        ):
            raise ValueError(f"inconsistent_feature_parent: {transcript}")
        genomic = sorted((start, end) for start, end, _, _ in rows)
        if any(
            left[1] > right[0] for left, right in zip(genomic, genomic[1:])
        ):
            raise ValueError(f"overlapping_exons: {transcript}")
        ordered = tuple(genomic if strand == "+" else reversed(genomic))
        sequence = sequences[transcript]
        length = sum(end - start for start, end in ordered)
        if len(sequence) != length:
            raise ValueError(
                f"transcript_length_mismatch: {transcript}: "
                f"FASTA={len(sequence)}, exons={length}"
            )
        empty_track = tuple(0 for _ in sequence)
        result[transcript] = TranscriptAnnotation(
            transcript,
            gene,
            chromosome,
            strand,
            sequence,
            ordered,
            empty_track,
            empty_track,
        )
    return result


__all__ = ["read_rinalmo_annotations"]
