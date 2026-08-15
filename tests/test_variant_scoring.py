from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest
import torch

from vep_comparisons.variant_models import (
    MODEL_REGISTRY,
    PositionRequest,
    batch_requests,
    log_odds_from_logits,
    marginalized_log_odds_from_logits,
)
from vep_comparisons.variant_scoring import (
    InputValidationError,
    read_transcript_fasta,
    score_protein_variants,
)
from vep_comparisons.score_variants_cli import run_cli


def _write_variants(path: Path, rows: list[dict[str, str]]) -> None:
    pl.DataFrame(rows).write_csv(path, separator="\t", compression="gzip")


def _row(
    variant: str,
    transcript: str,
    position: int | str,
    amino_acids: str,
    *,
    consequence: str = "missense_variant",
    biotype: str = "protein_coding",
) -> dict[str, str]:
    return {
        "variant": variant,
        "gene": f"gene-{variant}",
        "feature": transcript,
        "consequence": consequence,
        "protein_position": str(position),
        "amino_acids": amino_acids,
        "biotype": biotype,
    }


def _write_fasta(path: Path, records: dict[str, str]) -> None:
    opener = gzip.open if path.suffix == ".gz" else path.open
    if path.suffix == ".gz":
        with opener(path, "wt") as output:
            for header, sequence in records.items():
                output.write(f">{header}\n{sequence}\n")
    else:
        with opener("wt") as output:
            for header, sequence in records.items():
                output.write(f">{header}\n{sequence}\n")


class FakeModel:
    def __init__(self) -> None:
        self.requests: list[PositionRequest] = []

    def score_positions(self, requests, *, batch_size):
        self.requests.extend(requests)
        # A deterministic amino-acid score makes ALT - REF easy to inspect.
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        return {
            request.key: {aa: float(index) for index, aa in enumerate(alphabet)}
            for request in requests
        }


def _factory_for(fake: FakeModel, calls: list[str]):
    def factory(spec, model_root, device, dtype):
        calls.append(spec.name)
        return fake

    return factory


def test_fasta_removes_versions_and_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    fasta = tmp_path / "translations.fa.gz"
    _write_fasta(
        fasta,
        {
            "ENST000001.8|ENSG000001.2 gene metadata": "ACD",
            "ENSP000003.1|ENST000003.4|ENSG000003.2|GENCODE metadata": "EFG",
            "TX000002.1 more metadata": "MNP",
        },
    )
    assert read_transcript_fasta(fasta) == {
        "ENST000001": "ACD",
        "ENST000003": "EFG",
        "TX000002": "MNP",
    }

    duplicate = tmp_path / "duplicate.fa"
    duplicate.write_text(
        ">ENST000001.1 first\nACD\n>ENST000001.2 second\nACD\n"
    )
    with pytest.raises(InputValidationError, match="duplicate_transcript"):
        read_transcript_fasta(duplicate)


def test_selection_order_deduplication_and_overlength_nulls(tmp_path: Path) -> None:
    variants = tmp_path / "variants.tsv.gz"
    sequences = tmp_path / "sequences.fa"
    output = tmp_path / "scores"
    _write_variants(
        variants,
        [
            _row("first", "ENST000001.2", 1, "A/C", consequence="missense_variant,splice_region_variant"),
            _row("same-position", "ENST000001.2", 1, "A/D"),
            _row("not-missense", "ENST000001", 2, "C/D", consequence="synonymous_variant"),
            _row("not-coding", "ENST000001", 2, "C/D", biotype="lncRNA"),
            _row("overlength", "ENST000002", 4, "E/F"),
            _row("second-position", "ENST000001", 3, "D/E"),
        ],
    )
    _write_fasta(sequences, {"ENST000001.9": "ACD", "ENST000002.1": "ACDE"})
    fake = FakeModel()
    calls: list[str] = []

    summary = score_protein_variants(
        variants_path=variants,
        sequences_path=sequences,
        model_name="esmc-300m",
        model_root=tmp_path,
        output=output,
        max_sequence_length=3,
        batch_size=2,
        model_factory=_factory_for(fake, calls),
    )

    result = pl.read_csv(summary.output_path, separator="\t")
    assert calls == ["esmc-300m"]
    assert [(request.transcript, request.position) for request in fake.requests] == [
        ("ENST000001", 1),
        ("ENST000001", 3),
    ]
    assert result.get_column("variant").to_list() == [
        "first",
        "same-position",
        "overlength",
        "second-position",
    ]
    assert result.get_column("score").to_list() == [1.0, 2.0, None, 1.0]
    assert result.columns == [
        "variant",
        "gene",
        "feature",
        "protein_position",
        "amino_acids",
        "model",
        "score",
    ]
    assert summary.output_path.read_bytes().startswith(b"\x1f\x8b")


@pytest.mark.parametrize(
    ("position", "amino_acids", "category"),
    [
        (4, "D/E", "position_out_of_range"),
        (2, "A/C", "reference_mismatch"),
    ],
)
def test_inconsistent_candidates_fail_before_model_loading_and_leave_no_output(
    tmp_path: Path,
    position: int,
    amino_acids: str,
    category: str,
) -> None:
    variants = tmp_path / "variants.tsv.gz"
    sequences = tmp_path / "sequences.fa"
    output = tmp_path / "scores"
    _write_variants(variants, [_row("bad", "ENST000001", position, amino_acids)])
    _write_fasta(sequences, {"ENST000001": "ACD"})
    calls: list[str] = []

    with pytest.raises(InputValidationError, match=category):
        score_protein_variants(
            variants_path=variants,
            sequences_path=sequences,
            model_name="esmc-300m",
            model_root=tmp_path,
            output=output,
            model_factory=_factory_for(FakeModel(), calls),
        )
    assert calls == []
    assert not Path(f"{output}.tsv.gz").exists()


def test_missing_sequence_fails_before_model_loading(tmp_path: Path) -> None:
    variants = tmp_path / "variants.tsv.gz"
    sequences = tmp_path / "sequences.fa"
    _write_variants(variants, [_row("bad", "ENST000001", 1, "A/C")])
    _write_fasta(sequences, {"ENST000002": "ACD"})
    calls: list[str] = []

    with pytest.raises(InputValidationError, match="missing_sequence"):
        score_protein_variants(
            variants_path=variants,
            sequences_path=sequences,
            model_name="esmc-300m",
            model_root=tmp_path,
            output=tmp_path / "scores",
            model_factory=_factory_for(FakeModel(), calls),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("structure_records", "category"),
    [
        ({}, "missing_structure_tokens"),
        ({"ENST000001": "pp"}, "sequence_structure_length_mismatch"),
    ],
)
def test_saprot_structure_validation_is_fail_fast(
    tmp_path: Path,
    structure_records: dict[str, str],
    category: str,
) -> None:
    variants = tmp_path / "variants.tsv.gz"
    sequences = tmp_path / "sequences.fa"
    structures = tmp_path / "structures.fa"
    _write_variants(variants, [_row("v", "ENST000001", 1, "A/C")])
    _write_fasta(sequences, {"ENST000001": "ACD"})
    _write_fasta(structures, structure_records)
    calls: list[str] = []

    with pytest.raises(InputValidationError, match=category):
        score_protein_variants(
            variants_path=variants,
            sequences_path=sequences,
            structure_tokens_path=structures,
            model_name="saprot-35m",
            model_root=tmp_path,
            output=tmp_path / "scores",
            model_factory=_factory_for(FakeModel(), calls),
        )
    assert calls == []


def test_overlength_saprot_rows_still_require_valid_structure(tmp_path: Path) -> None:
    variants = tmp_path / "variants.tsv.gz"
    sequences = tmp_path / "sequences.fa"
    structures = tmp_path / "structures.fa"
    _write_variants(variants, [_row("v", "ENST000001", 1, "A/C")])
    _write_fasta(sequences, {"ENST000001": "A" * 1025})
    _write_fasta(structures, {})
    with pytest.raises(InputValidationError, match="missing_structure_tokens"):
        score_protein_variants(
            variants_path=variants,
            sequences_path=sequences,
            structure_tokens_path=structures,
            model_name="saprot-35m",
            model_root=tmp_path,
            output=tmp_path / "scores",
            model_factory=_factory_for(FakeModel(), []),
        )


def test_model_capacity_boundaries_and_above_capacity_rejection(tmp_path: Path) -> None:
    for model_name in ("esmc-300m", "saprot-35m"):
        spec = MODEL_REGISTRY[model_name]
        variants = tmp_path / f"{model_name}.tsv.gz"
        sequences = tmp_path / f"{model_name}.fa"
        structures = tmp_path / f"{model_name}.3di.fa"
        sequence = "A" * spec.capacity
        _write_variants(variants, [_row("v", "ENST000001", spec.capacity, "A/C")])
        _write_fasta(sequences, {"ENST000001": sequence})
        kwargs = {}
        if spec.family == "saprot":
            _write_fasta(structures, {"ENST000001": "p" * spec.capacity})
            kwargs["structure_tokens_path"] = structures
        fake = FakeModel()
        score_protein_variants(
            variants_path=variants,
            sequences_path=sequences,
            model_name=model_name,
            model_root=tmp_path,
            output=tmp_path / f"{model_name}-scores",
            batch_size=1,
            model_factory=_factory_for(fake, []),
            **kwargs,
        )
        assert len(fake.requests[0].sequence) == spec.capacity

        with pytest.raises(InputValidationError, match="invalid_max_sequence_length"):
            score_protein_variants(
                variants_path=variants,
                sequences_path=sequences,
                model_name=model_name,
                model_root=tmp_path,
                output=tmp_path / "never-written",
                max_sequence_length=spec.capacity + 1,
                model_factory=_factory_for(FakeModel(), []),
                **kwargs,
            )


def test_masked_marginal_math_and_fixed_size_batching() -> None:
    logits = [0.0, -2.0, 3.0, 1.0, 2.0]
    assert log_odds_from_logits(logits, ref_token=1, alt_token=2) == 5.0
    values = torch.tensor(logits)
    expected = torch.logsumexp(values[[3, 4]], dim=0) - torch.logsumexp(
        values[[1, 2]], dim=0
    )
    assert marginalized_log_odds_from_logits(
        logits, ref_tokens=[1, 2], alt_tokens=[3, 4]
    ) == pytest.approx(expected)

    requests = [
        PositionRequest("ENST000001", "AAA", 1),
        PositionRequest("ENST000001", "AAA", 2),
        PositionRequest("ENST000002", "AAAAA", 1),
    ]
    assert [len(batch) for batch in batch_requests(requests, 2)] == [2, 1]
    assert [request.key for batch in batch_requests(requests, 10) for request in batch] == [
        request.key for request in requests
    ]
    with pytest.raises(ValueError, match="batch_size must be positive"):
        batch_requests(requests, 0)


def test_cli_with_fake_adapter_writes_output_and_reports_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    variants = tmp_path / "variants.tsv.gz"
    sequences = tmp_path / "sequences.fa"
    output = tmp_path / "scores.tsv.gz"
    _write_variants(variants, [_row("v", "ENST000001", 1, "A/C")])
    _write_fasta(sequences, {"ENST000001": "ACD"})

    exit_code = run_cli(
        [
            "--variants",
            str(variants),
            "--sequences",
            str(sequences),
            "--model",
            "esmc-300m",
            "--model-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--batch-size",
            "2",
        ],
        model_factory=_factory_for(FakeModel(), []),
    )

    stderr = capsys.readouterr().err
    assert exit_code == 0
    assert output.exists()
    assert "model=esmc-300m" in stderr
    assert "validation=passed" in stderr
    assert "scored=1 null=0" in stderr
