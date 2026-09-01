from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest
import torch

from vep_comparisons.dna_variant_scoring import ShardCompatibilityError
from vep_comparisons.orthrus_scoring import OrthrusCandidate, PreparedRequest, TranscriptAnnotation
from vep_comparisons.rinalmo_scoring import (
    MAX_MODEL_TOKENS,
    OUTPUT_COLUMNS,
    TOKEN_IDS,
    TOKENS,
    RiNALMoModel,
    RiNALMoScore,
    build_masked_tokens,
    collate_rinalmo_scores,
    read_rinalmo_annotations,
    score_rinalmo_variants,
    select_transcript_window,
    sha256_file,
    validate_rinalmo_checkpoint,
)


def request(sequence: str, target: int, ref: str = "A", alt: str = "C") -> PreparedRequest:
    annotation = TranscriptAnnotation("T", "G", "chr1", "+", sequence, ((0, len(sequence)),), tuple(0 for _ in sequence), tuple(0 for _ in sequence))
    candidate = OrthrusCandidate("v", "G", "T", "chr1", target, target + 1, target + 1, ref, alt, 0)
    return PreparedRequest(candidate, annotation, ref, alt)


@pytest.mark.parametrize(
    ("length", "target", "expected"),
    [
        (10, 0, (0, 10, 12, True, True)),
        (1022, 500, (0, 1022, 1024, True, True)),
        (1023, 0, (0, 1022, 1023, True, False)),
        (1023, 1022, (1, 1023, 1023, False, True)),
        (1024, 0, (0, 1023, 1024, True, False)),
        (1024, 1023, (1, 1024, 1024, False, True)),
    ],
)
def test_window_boundaries_and_edge_lengths(length, target, expected) -> None:
    window = select_transcript_window(length, target)
    assert (window.start, window.end, window.token_count, window.add_cls, window.add_eos) == expected
    assert window.token_count <= MAX_MODEL_TOKENS


def test_window_interior_balance_and_upstream_tie() -> None:
    window = select_transcript_window(3000, 1500)
    assert (window.start, window.end) == (988, 2012)
    assert 1500 - window.start == window.end - 1 - 1500 + 1


def test_special_tokens_and_mask_alignment() -> None:
    tokens, window = build_masked_tokens(request("ACGT", 2, "G", "T"))
    assert tokens == [TOKEN_IDS["<cls>"], TOKEN_IDS["A"], TOKEN_IDS["C"], TOKEN_IDS["<mask>"], TOKEN_IDS["T"], TOKEN_IDS["<eos>"]]
    assert window.target_token_index == 3
    bad = request("ACGT", 2, "A", "T")
    with pytest.raises(RuntimeError, match="alignment"):
        build_masked_tokens(bad)


class FakeLM:
    def __call__(self, tokens):
        batch, length = tokens.shape
        logits = torch.zeros((batch, length, len(TOKENS)))
        logits[..., TOKEN_IDS["A"]] = -1
        logits[..., TOKEN_IDS["C"]] = 2
        return {"logits": logits}


def fake_model() -> RiNALMoModel:
    model = RiNALMoModel.__new__(RiNALMoModel)
    model._torch = torch
    model.device = torch.device("cpu")
    model.model = FakeLM()
    return model


def test_model_batch_padding_vocab_and_scores() -> None:
    first = request("A", 0)
    second = request("A" * 1025, 500)
    scores = fake_model().score_requests([second, first], batch_size=2)
    assert set(scores) == {first.key, second.key}
    assert scores[first.key].score == pytest.approx(3.0)
    assert scores[first.key].model_token_count == 3
    assert scores[second.key].model_token_count == 1024


def test_model_rejects_shape_and_nonfinite() -> None:
    model = fake_model()
    model.model = lambda tokens: {"logits": torch.zeros((1, 1, 21))}
    with pytest.raises(RuntimeError, match="shape"):
        model.score_requests([request("A", 0)], batch_size=1)
    model.model = lambda tokens: {"logits": torch.full((1, 3, 22), float("nan"))}
    with pytest.raises(RuntimeError, match="non-finite"):
        model.score_requests([request("A", 0)], batch_size=1)


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    variants = tmp_path / "variants.tsv.gz"
    with gzip.open(variants, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "gene", "feature", "chromosome", "start", "end", "cdna_position", "ref", "alt"), delimiter="\t")
        writer.writeheader()
        writer.writerows([
            {"variant": "minus", "gene": "G2", "feature": "M", "chromosome": "chr1", "start": 21, "end": 22, "cdna_position": 2, "ref": "G", "alt": "A"},
            {"variant": "plus", "gene": "G1", "feature": "P", "chromosome": "chr1", "start": 1, "end": 2, "cdna_position": 2, "ref": "C", "alt": "T"},
            {"variant": "excluded", "gene": "G1", "feature": "P", "chromosome": "chr1", "start": 0, "end": 1, "cdna_position": "", "ref": "A", "alt": "C"},
        ])
    fasta = tmp_path / "transcripts.fa.gz"
    with gzip.open(fasta, "wt") as handle:
        handle.write(">P.1\nACGT\n>M.1\nACGT\n")
    gff = tmp_path / "annotation.gff3.gz"
    with gzip.open(gff, "wt") as handle:
        handle.write("chr1\tt\ttranscript\t1\t4\t.\t+\t.\tID=transcript:P;gene_id=G1\n")
        handle.write("chr1\tt\texon\t1\t4\t.\t+\t.\tParent=transcript:P\n")
        handle.write("chr1\tt\tCDS\tbad\tbad\t.\t+\t.\tParent=transcript:P\n")
        handle.write("chr1\tt\ttranscript\t20\t23\t.\t-\t.\tID=transcript:M;gene_id=G2\n")
        handle.write("chr1\tt\texon\t20\t23\t.\t-\t.\tParent=transcript:M\n")
    return variants, fasta, gff


def make_weights(tmp_path: Path) -> tuple[Path, Path]:
    weights = tmp_path / "rinalmo_giga_pretrained.pt"
    weights.write_bytes(b"fake giga")
    expected = {"config_name": "giga", "embed_dim": 1280, "num_blocks": 33, "num_heads": 20, "alphabet_size": 22, "tokens": list(TOKENS), "token_ids": TOKEN_IDS}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model": "rinalmo-giga", "repository": "lbcb-sci/RiNALMo", "revision": "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955", "sha256": {weights.name: sha256_file(weights)}, "expected": expected}))
    return weights, manifest


class FakeScorer:
    def score_requests(self, requests, *, batch_size):
        result = {}
        for item in requests:
            window = select_transcript_window(len(item.annotation.sequence), item.candidate.target_index)
            result[item.key] = RiNALMoScore(-2.0, -1.0, window.start, window.end, window.token_count)
        return result


def test_no_cds_parsing_checkpoint_pipeline_resume_and_collation(tmp_path: Path) -> None:
    variants, fasta, gff = write_fixture(tmp_path)
    annotations = read_rinalmo_annotations(gff, fasta, {"P", "M"})
    assert annotations["M"].genomic_position(1) == 21
    weights, manifest = make_weights(tmp_path)
    assert validate_rinalmo_checkpoint(weights, manifest)["model"] == "rinalmo-giga"
    output_dir = tmp_path / "shards"
    for index in range(2):
        score_rinalmo_variants(variants_path=variants, transcript_fasta_path=fasta, gff3_path=gff, weights=weights, manifest_path=manifest, output_dir=output_dir, num_shards=2, shard_index=index, device="cpu", dtype="float32", model_factory=lambda *args: FakeScorer())
    reused = score_rinalmo_variants(variants_path=variants, transcript_fasta_path=fasta, gff3_path=gff, weights=weights, manifest_path=manifest, output_dir=output_dir, num_shards=2, shard_index=0, device="cpu", dtype="float32", model_factory=lambda *args: pytest.fail("loaded model"))
    assert reused.reused
    final = tmp_path / "final.tsv.gz"
    collate_rinalmo_scores(variants_path=variants, transcript_fasta_path=fasta, gff3_path=gff, weights=weights, manifest_path=manifest, output_dir=output_dir, output_path=final, num_shards=2, dtype="float32")
    with gzip.open(final, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert tuple(rows[0]) == OUTPUT_COLUMNS
    assert [row["variant"] for row in rows] == ["plus", "minus"]
    assert rows[1]["transcript_ref"] == "C"
    assert rows[1]["transcript_alt"] == "T"
    assert rows[0]["score"] == "1.0"
    metadata = json.loads(final.with_suffix(".json").read_text())
    assert metadata["identity"]["dtype"] == "float32"
    assert metadata["ineligible_counts"] == {"missing_cdna_position": 1}
    with pytest.raises(ShardCompatibilityError, match="incompatible"):
        collate_rinalmo_scores(variants_path=variants, transcript_fasta_path=fasta, gff3_path=gff, weights=weights, manifest_path=manifest, output_dir=output_dir, output_path=final, num_shards=2, dtype="bfloat16")


def test_checkpoint_hash_architecture_and_alphabet_failures(tmp_path: Path) -> None:
    weights, manifest = make_weights(tmp_path)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_rinalmo_checkpoint(weights, manifest)
    weights.write_bytes(b"fake giga")
    payload = json.loads(manifest.read_text())
    payload["expected"]["num_blocks"] = 32
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="architecture"):
        validate_rinalmo_checkpoint(weights, manifest)
    payload["expected"]["num_blocks"] = 33
    payload["expected"]["token_ids"]["A"] = 6
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="alphabet"):
        validate_rinalmo_checkpoint(weights, manifest)
