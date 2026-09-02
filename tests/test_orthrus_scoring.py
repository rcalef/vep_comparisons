from __future__ import annotations

import csv
import gzip
import json
import subprocess
from pathlib import Path

import pytest
import torch

from vep_comparisons.dna.workflow import InputValidationError, ShardCompatibilityError
from vep_comparisons.rna.orthrus.workflow import (
    OUTPUT_COLUMNS,
    ComponentScores,
    OrthrusCandidate,
    OrthrusModel,
    PreparedRequest,
    TranscriptAnnotation,
    build_six_track_input,
    candidate_sort_key,
    collate_orthrus_scores,
    natural_chromosome_key,
    prepare_requests,
    read_orthrus_candidates,
    read_transcript_annotations,
    score_orthrus_variants,
    shard_candidates,
    validate_orthrus_checkpoint,
)


def write_variants(path: Path, rows: list[dict[str, object]]) -> None:
    columns = (
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
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def variant(
    name: str,
    transcript: str,
    start: int,
    cdna: object,
    ref: str,
    alt: str,
    *,
    gene: str = "ENSG1",
    chromosome: str = "chr1",
) -> dict[str, object]:
    return {
        "variant": name,
        "gene": gene,
        "feature": transcript,
        "chromosome": chromosome,
        "start": start,
        "end": start + 1,
        "cdna_position": cdna,
        "ref": ref,
        "alt": alt,
    }


def write_annotations(tmp_path: Path) -> tuple[Path, Path]:
    fasta = tmp_path / "transcripts.fa.gz"
    with gzip.open(fasta, "wt") as handle:
        handle.write(
            ">ENSTPLUS.7|ENSG1.2|metadata\nACGTGGA\n"
            ">ENSTMINUS.3|ENSG2.1|metadata\nTTTAACC\n"
            ">ENSTLNC.1|ENSG3.1|metadata\nAAAA\n"
        )
    gff = tmp_path / "annotation.gff3.gz"
    with gzip.open(gff, "wt") as handle:
        handle.write("##gff-version 3\n")
        handle.write("chr1\tt\ttranscript\t101\t203\t.\t+\t.\tID=transcript:ENSTPLUS.7;gene_id=ENSG1.2\n")
        handle.write("chr1\tt\texon\t201\t203\t.\t+\t.\tParent=transcript:ENSTPLUS.7\n")
        handle.write("chr1\tt\texon\t101\t104\t.\t+\t.\tParent=transcript:ENSTPLUS.7\n")
        handle.write("chr1\tt\tCDS\t201\t202\t.\t+\t0\tParent=transcript:ENSTPLUS.7\n")
        handle.write("chr1\tt\tCDS\t103\t104\t.\t+\t0\tParent=transcript:ENSTPLUS.7\n")
        handle.write("chr1\tt\ttranscript\t301\t403\t.\t-\t.\tID=transcript:ENSTMINUS.3;gene_id=ENSG2.1\n")
        handle.write("chr1\tt\texon\t301\t303\t.\t-\t.\tParent=transcript:ENSTMINUS.3\n")
        handle.write("chr1\tt\texon\t400\t403\t.\t-\t.\tParent=transcript:ENSTMINUS.3\n")
        handle.write("chr1\tt\tCDS\t302\t303\t.\t-\t0\tParent=transcript:ENSTMINUS.3\n")
        handle.write("chr1\tt\tCDS\t400\t403\t.\t-\t0\tParent=transcript:ENSTMINUS.3\n")
        handle.write("chr2\tt\ttranscript\t11\t14\t.\t+\t.\tID=transcript:ENSTLNC.1;gene_id=ENSG3.1\n")
        handle.write("chr2\tt\texon\t11\t14\t.\t+\t.\tParent=transcript:ENSTLNC.1\n")
    return fasta, gff


def make_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {
        "n_tracks": 6,
        "has_mlm_head": True,
        "mlm_head_dim": 4,
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    (checkpoint / "model.safetensors").write_bytes(b"fake weights")
    (checkpoint / "orthrus_hf.py").write_text("# fake code\n")
    return checkpoint


def test_gff_projection_tracks_both_strands_and_lnc(tmp_path: Path) -> None:
    fasta, gff = write_annotations(tmp_path)
    annotations = read_transcript_annotations(
        gff, fasta, {"ENSTPLUS", "ENSTMINUS", "ENSTLNC"}
    )

    plus = annotations["ENSTPLUS"]
    assert plus.exons == ((100, 104), (200, 203))
    assert plus.splice_track == (0, 0, 0, 1, 0, 0, 1)
    assert plus.cds_track == (0, 0, 1, 0, 0, 1, 0)
    assert [plus.genomic_position(i) for i in range(7)] == [100, 101, 102, 103, 200, 201, 202]

    minus = annotations["ENSTMINUS"]
    assert minus.exons == ((399, 403), (300, 303))
    assert minus.splice_track == (0, 0, 0, 1, 0, 0, 1)
    assert minus.cds_track == (1, 0, 0, 1, 0, 0, 0)
    assert [minus.genomic_position(i) for i in range(7)] == [402, 401, 400, 399, 302, 301, 300]

    lnc = annotations["ENSTLNC"]
    assert lnc.cds_track == (0, 0, 0, 0)
    assert lnc.splice_track == (0, 0, 0, 1)


def test_eligibility_coordinate_conversion_and_minus_complement(tmp_path: Path) -> None:
    fasta, gff = write_annotations(tmp_path)
    variants = tmp_path / "variants.tsv.gz"
    write_variants(
        variants,
        [
            variant("plus", "ENSTPLUS.9", 102, 3, "G", "A"),
            variant("minus", "ENSTMINUS", 401, 2, "A", "C", gene="ENSG2"),
            variant("missing", "ENSTPLUS", 100, "", "A", "C"),
            variant("bad-alt", "ENSTPLUS", 100, 1, "A", "N"),
        ],
    )
    candidates, inventory = read_orthrus_candidates(variants)
    assert inventory.eligible_rows == 2
    assert inventory.ineligible_counts == {
        "missing_cdna_position": 1,
        "unsupported_alt": 1,
    }
    annotations = read_transcript_annotations(
        gff, fasta, {candidate.feature for candidate in candidates}
    )
    requests = prepare_requests(candidates, annotations)
    assert (requests[0].transcript_ref, requests[0].transcript_alt) == ("G", "A")
    assert (requests[1].transcript_ref, requests[1].transcript_alt) == ("T", "G")

    malformed = tmp_path / "malformed.tsv.gz"
    write_variants(malformed, [variant("bad", "ENSTPLUS", 100, "1/2", "A", "C")])
    with pytest.raises(InputValidationError, match="invalid_cdna_position"):
        read_orthrus_candidates(malformed)


def test_ref_and_coordinate_mismatches_fail_immediately(tmp_path: Path) -> None:
    fasta, gff = write_annotations(tmp_path)
    annotations = read_transcript_annotations(gff, fasta, {"ENSTPLUS"})
    rows = [
        OrthrusCandidate("wrong-pos", "ENSG1", "ENSTPLUS", "chr1", 101, 102, 3, "G", "A", 0),
        OrthrusCandidate("wrong-ref", "ENSG1", "ENSTPLUS", "chr1", 102, 103, 3, "C", "A", 1),
    ]
    with pytest.raises(InputValidationError, match="cdna_genomic_position_mismatch"):
        prepare_requests(rows, annotations)


def test_mask_preserves_cds_and_splice_and_prefix_full_are_causal_equivalent() -> None:
    annotation = TranscriptAnnotation(
        "T", "G", "chr1", "+", "ACGTA", ((0, 5),), (1, 0, 0, 1, 0), (0, 0, 1, 0, 1)
    )
    first = PreparedRequest(
        OrthrusCandidate("v1", "G", "T", "chr1", 2, 3, 3, "G", "T", 0),
        annotation,
        "G",
        "T",
    )
    second_annotation = TranscriptAnnotation(
        "U", "G", "chr1", "+", "CA", ((10, 12),), (0, 1), (0, 1)
    )
    second = PreparedRequest(
        OrthrusCandidate("v2", "G", "U", "chr1", 11, 12, 2, "A", "C", 1),
        second_annotation,
        "A",
        "C",
    )
    masked = build_six_track_input(annotation, 2)
    assert masked[2] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert len(masked) == 3

    class CausalModel:
        def predict_tokens(self, inputs, lengths, channel_last=True):
            assert channel_last
            signal = inputs.sum(dim=-1).cumsum(dim=1)
            return torch.stack((signal, signal * 2, -signal, signal + 0.5), dim=-1)

    scorer = OrthrusModel.__new__(OrthrusModel)
    scorer._torch = torch
    scorer.device = torch.device("cpu")
    scorer.model = CausalModel()
    prefix = scorer.score_requests([first, second], batch_size=32, prefix_only=True)
    full = scorer.score_requests([first, second], batch_size=32, prefix_only=False)
    assert prefix == full
    assert prefix[first.key].score == pytest.approx(
        prefix[first.key].alt_log_probability - prefix[first.key].ref_log_probability
    )


def test_natural_order_contiguous_balanced_shards_and_ties() -> None:
    chromosomes = ["chr10", "chr2", "chr1", "chr22"]
    rows = [
        OrthrusCandidate(f"v{i}", "G", f"T{i}", chrom, 100 + i, 101 + i, 1, "A", "C", i)
        for i, chrom in enumerate(chromosomes)
    ]
    tied = [
        OrthrusCandidate("z", "G2", "T2", "chr1", 9, 10, 1, "A", "C", 5),
        OrthrusCandidate("a", "G1", "T1", "chr1", 9, 10, 1, "A", "C", 6),
    ]
    ordered = sorted(rows + tied, key=candidate_sort_key)
    assert [row.chromosome for row in ordered[:4]] == ["chr1", "chr1", "chr1", "chr2"]
    assert [(row.feature, row.gene) for row in ordered[:2]] == [("T1", "G1"), ("T2", "G2")]
    shards = [shard_candidates(ordered, num_shards=4, shard_index=i) for i in range(4)]
    assert [len(shard) for shard in shards] == [1, 2, 1, 2]
    assert sum(shards, []) == ordered
    assert natural_chromosome_key("chr2") < natural_chromosome_key("chr10")


class FakeScorer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def score_requests(self, requests, *, batch_size):
        self.batch_sizes.append(batch_size)
        return {
            request.key: ComponentScores(-2.0 - index, -1.25 - index)
            for index, request in enumerate(requests)
        }


def test_checkpoint_scoring_and_collation(tmp_path: Path) -> None:
    fasta, gff = write_annotations(tmp_path)
    checkpoint = make_checkpoint(tmp_path)
    validate_orthrus_checkpoint(checkpoint)
    variants = tmp_path / "variants.tsv.gz"
    write_variants(
        variants,
        [
            variant("later", "ENSTMINUS", 401, 2, "A", "C", gene="ENSG2"),
            variant("first", "ENSTPLUS", 102, 3, "G", "A"),
            variant("excluded", "ENSTPLUS", 100, "", "A", "C"),
        ],
    )
    output_dir = tmp_path / "shards"
    scorers: list[FakeScorer] = []

    def factory(*args):
        scorer = FakeScorer()
        scorers.append(scorer)
        return scorer

    for index in range(2):
        summary = score_orthrus_variants(
            variants_path=variants,
            transcript_fasta_path=fasta,
            gff3_path=gff,
            checkpoint=checkpoint,
            output_dir=output_dir,
            num_shards=2,
            shard_index=index,
            device="cpu",
            batch_size=32,
            model_factory=factory,
        )
        assert summary.rows == 1
    assert [scorer.batch_sizes for scorer in scorers] == [[32], [32]]
    final = tmp_path / "final.tsv.gz"
    collate_orthrus_scores(
        variants_path=variants,
        transcript_fasta_path=fasta,
        gff3_path=gff,
        checkpoint=checkpoint,
        output_dir=output_dir,
        output_path=final,
        num_shards=2,
    )
    with gzip.open(final, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert tuple(rows[0]) == OUTPUT_COLUMNS
    assert [row["variant"] for row in rows] == ["first", "later"]
    assert float(rows[0]["score"]) == pytest.approx(0.75)


def test_checkpoint_six_track_validation(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    (checkpoint / "config.json").write_text(json.dumps({"n_tracks": 5}))
    with pytest.raises(ValueError, match="six tracks"):
        validate_orthrus_checkpoint(checkpoint)
