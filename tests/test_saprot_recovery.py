from __future__ import annotations

import bz2
import gzip
import subprocess
from pathlib import Path

import pytest

from vep_comparisons.protein.recovery.alphafold import (
    AlphaFoldClient,
    AlphaFoldModel,
    select_exact_model,
)
from vep_comparisons.protein.recovery.foldseek import recover_tokens
from vep_comparisons.protein.recovery.workflow import (
    MismatchCandidate,
    RecoveryError,
    merge_tokens,
    read_fasta,
    read_mismatch_candidates,
)


def write_fasta(path: Path, records: dict[str, str]) -> None:
    opener = (
        bz2.open
        if path.suffix == ".bz2"
        else gzip.open
        if path.suffix == ".gz"
        else open
    )
    with opener(path, "wt") as handle:
        for name, sequence in records.items():
            handle.write(f">{name}\n{sequence}\n")


def model(
    *,
    entry: str = "AF-P52569-3-F1",
    sequence: str = "MABCDE",
    start: int = 1,
    end: int | None = None,
    entity_type: str = "protein",
    is_complex: bool = False,
) -> AlphaFoldModel:
    return AlphaFoldModel(
        model_entity_id=entry,
        uniprot_accession="P52569-3",
        sequence=sequence,
        sequence_checksum="checksum",
        sequence_start=start,
        sequence_end=len(sequence) if end is None else end,
        entity_type=entity_type,
        is_complex=is_complex,
        latest_version="6",
        model_created_date="2025-01-01",
        pdb_url=f"https://example.test/{entry}.pdb",
        global_metric_value="80.2",
    )


def candidate(sequence: str = "MABCDE") -> MismatchCandidate:
    return MismatchCandidate(
        transcript="ENST00000004531",
        transcript_version="ENST00000004531.7",
        uniprot_base="P52569",
        sequence=sequence,
        existing_foldseek_length=5,
        variant_count=2,
        variant_positions="2;5",
    )


def test_read_mismatch_candidates_enriches_from_translation(tmp_path: Path) -> None:
    translations = tmp_path / "translations.fa.gz"
    write_fasta(translations, {"ENSP|ENST00000004531.7|GENE": "MABCDE"})
    mismatches = tmp_path / "mismatches.tsv"
    mismatches.write_text(
        "transcript\tuniprot_id\tstructure_token_length\tvariant_count\tvariant_positions\n"
        "ENST00000004531\tP52569-3\t5\t2\t2;5\n",
        encoding="utf-8",
    )

    result = read_mismatch_candidates(mismatches, translations)

    assert result == [candidate()]


def test_read_mismatch_candidates_rejects_sequence_disagreement(
    tmp_path: Path,
) -> None:
    translations = tmp_path / "translations.fa"
    write_fasta(translations, {"ENST00000004531.7": "MABCDE"})
    mismatches = tmp_path / "mismatches.tsv"
    mismatches.write_text(
        "transcript\tuniprot_base\tgencode_sequence\n"
        "ENST00000004531\tP52569\tMABCDF\n",
        encoding="utf-8",
    )

    with pytest.raises(RecoveryError, match="sequence disagrees"):
        read_mismatch_candidates(mismatches, translations)


def test_read_mismatch_candidates_combines_multiple_accession_mappings(
    tmp_path: Path,
) -> None:
    translations = tmp_path / "translations.fa"
    write_fasta(translations, {"ENST00000004531.7": "MABCDE"})
    mismatches = tmp_path / "mismatches.tsv"
    mismatches.write_text(
        "transcript\tuniprot_base\tgencode_length\n"
        "ENST00000004531\tP52569-3\t6\n"
        "ENST00000004531\tQ12345\t6\n",
        encoding="utf-8",
    )

    result = read_mismatch_candidates(mismatches, translations)

    assert len(result) == 1
    assert result[0].uniprot_accessions == ("P52569", "Q12345")


def test_select_exact_model_requires_unique_full_length_monomer() -> None:
    selected = select_exact_model(
        candidate(),
        [
            model(entry="canonical", sequence="MABCD"),
            model(),
            model(entry="complex", is_complex=True),
        ],
    )
    assert selected == model()

    selected = select_exact_model(
        candidate(), [model(entry="first"), model(entry="second")]
    )
    assert selected is None


def test_select_exact_model_reports_fragment_only() -> None:
    selected = select_exact_model(
        candidate("MABCDEFG"), [model(sequence="ABCDE", start=2, end=6)]
    )
    assert selected is None


def test_discovery_uses_uniprot_fallback_only_when_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = AlphaFoldClient(pdb_dir=tmp_path / "pdb")
    calls: list[str] = []
    base = {
        "modelEntityId": "AF-P52569-F1",
        "uniprotAccession": "P52569",
        "sequence": "SHORT",
        "sequenceStart": 1,
        "sequenceEnd": 5,
        "entityType": "protein",
        "isComplex": False,
        "pdbUrl": "https://example.test/base.pdb",
    }
    exact = {
        "modelEntityId": "AF-P52569-3-F1",
        "uniprotAccession": "P52569-3",
        "sequence": "MABCDE",
        "sequenceStart": 1,
        "sequenceEnd": 6,
        "entityType": "protein",
        "isComplex": False,
        "pdbUrl": "https://example.test/isoform.pdb",
    }

    def fake_request_json(url: str, **_: object) -> object:
        calls.append(url)
        if "uniprotkb" in url:
            return {
                "comments": [
                    {
                        "commentType": "ALTERNATIVE PRODUCTS",
                        "isoforms": [{"isoformIds": ["P52569-3"]}],
                    }
                ]
            }
        return [exact] if url.endswith("P52569-3") else [base]

    monkeypatch.setattr(client, "_request_json", fake_request_json)
    models = client._discover_accession("P52569", [candidate()])

    assert [item.model_entity_id for item in models] == [
        "AF-P52569-F1",
        "AF-P52569-3-F1",
    ]
    assert any("uniprotkb" in call for call in calls)


def test_discovery_skips_accession_when_enumeration_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = AlphaFoldClient(pdb_dir=tmp_path / "pdb")
    monkeypatch.setattr(
        client,
        "_discover_accession",
        lambda accession, candidates: (_ for _ in ()).throw(RuntimeError("failed")),
    )

    selected = client.discover([candidate()])

    assert selected == {}


def test_foldseek_validation_writes_lowercase_tokens(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdb"
    pdb_dir.mkdir()
    selected_model = model()
    downloaded = {candidate().transcript: selected_model}

    def fake_run(
        command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_text(
            f"{selected_model.model_entity_id}.pdb_A ALPHAFOLD DEHYDROGENASE/REDUCTASE"
            "\tMABCDE\tQWERTY\tper-residue-metrics\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    recovered = recover_tokens(
        downloaded,
        pdb_dir=pdb_dir,
        descriptor_path=tmp_path / "foldseek.tsv",
        foldseek="foldseek",
        threads=8,
        run_command=fake_run,
    )

    assert recovered == {candidate().transcript: "qwerty"}


def test_download_accepts_an_empty_selection_without_creating_pdb_dir(
    tmp_path: Path,
) -> None:
    client = AlphaFoldClient(pdb_dir=tmp_path / "pdb")

    assert client.download({}) == {}
    assert not client.pdb_dir.exists()


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [
            "AF-P52569-3-F1_A\tMABCDE\tQWERTY",
            "AF-P52569-3-F1_B\tMABCDE\tQWERTY",
        ],
        ["AF-P52569-3-F1_A\tMABCDF\tQWERTY"],
        ["AF-P52569-3-F1_A\tMABCDE\tQWERT"],
    ],
)
def test_foldseek_rejects_invalid_descriptors(
    tmp_path: Path, rows: list[str]
) -> None:
    pdb_dir = tmp_path / "pdb"
    pdb_dir.mkdir()
    transcript = candidate().transcript

    def fake_run(
        command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_text("\n".join(rows), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    recovered = recover_tokens(
        {transcript: model()},
        pdb_dir=pdb_dir,
        descriptor_path=tmp_path / "foldseek.tsv",
        foldseek="foldseek",
        threads=1,
        run_command=fake_run,
    )

    assert recovered == {}


def test_merge_tokens_validates_lengths_and_refuses_replacement(tmp_path: Path) -> None:
    translations = tmp_path / "translations.fa"
    write_fasta(
        translations,
        {"ENST_EXISTING.1": "ABC", "ENST_RECOVERED.2": "DEFG"},
    )
    existing = tmp_path / "existing.fa.bz2"
    write_fasta(existing, {"ENST_EXISTING": "qwe"})
    output = tmp_path / "merged.fa.bz2"

    merged = merge_tokens(
        existing_path=existing,
        recovered={"ENST_RECOVERED": "rtyu"},
        translations_path=translations,
        output_path=output,
    )

    assert merged == {"ENST_EXISTING": "qwe", "ENST_RECOVERED": "rtyu"}
    assert {key: value.sequence for key, value in read_fasta(output).items()} == merged

    with pytest.raises(RecoveryError, match="replace existing"):
        merge_tokens(
            existing_path=existing,
            recovered={"ENST_EXISTING": "qwe"},
            translations_path=translations,
            output_path=output,
        )
