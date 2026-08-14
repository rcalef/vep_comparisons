from __future__ import annotations

import gzip
import subprocess
import sys
from pathlib import Path

import polars as pl

from vep_comparisons.cli import (
    downsample,
    filter_and_select_annotations,
    parse_args,
    read_vep,
)


VEP_COLUMNS = [
    "#Uploaded_variation",
    "Gene",
    "Feature",
    "Consequence",
    "cDNA_position",
    "CDS_position",
    "Protein_position",
    "Amino_acids",
    "SYMBOL",
    "BIOTYPE",
    "CANONICAL",
    "MANE",
    "TSL",
]


def _annotation(
    variant: str,
    gene: str,
    feature: str,
    *,
    consequence: str = "missense_variant",
    biotype: str = "protein_coding",
    canonical: str | None = None,
    mane: str | None = None,
    tsl: str | None = "1",
) -> dict[str, str | None]:
    return {
        "variant": variant,
        "gene": gene,
        "feature": feature,
        "consequence": consequence,
        "cdna_position": "10",
        "cds_position": "8",
        "protein_position": "3",
        "amino_acids": "A/T",
        "symbol": gene.lower(),
        "biotype": biotype,
        "canonical": canonical,
        "mane": mane,
        "tsl": tsl,
    }


def _write_vep(path: Path, rows: list[dict[str, str | None]]) -> None:
    with gzip.open(path, "wt") as output:
        output.write("## ENSEMBL VARIANT EFFECT PREDICTOR\n")
        output.write("## cache version 115\n")
        output.write("\t".join(VEP_COLUMNS) + "\n")
        for row in rows:
            lower_row = {key.lower(): value for key, value in row.items()}
            lower_row["#uploaded_variation"] = row["variant"]
            output.write(
                "\t".join(
                    "-"
                    if lower_row.get(column.lower()) is None
                    else str(lower_row[column.lower()])
                    for column in VEP_COLUMNS
                )
                + "\n"
            )


def test_read_vep_skips_metadata_and_reads_gzip(tmp_path: Path) -> None:
    path = tmp_path / "vep.tsv.gz"
    _write_vep(path, [_annotation("v1", "G1", "ENST1", mane=None)])

    result = read_vep(path)

    assert result.shape == (1, len(VEP_COLUMNS))
    assert result.columns[0] == "variant"
    assert result.item(0, "variant") == "v1"
    assert result.item(0, "mane") is None


def test_filtering_biotypes_multiple_genes_and_transcript_priority() -> None:
    rows = [
        _annotation(
            "removed-intron",
            "G0",
            "ENST0",
            consequence="missense_variant,intron_variant",
        ),
        _annotation(
            "removed-downstream",
            "G0",
            "ENST0",
            consequence="downstream_gene_variant",
        ),
        _annotation(
            "removed-upstream",
            "G0",
            "ENST0",
            consequence="upstream_gene_variant",
        ),
        _annotation(
            "removed-intergenic",
            "G0",
            "ENST0",
            consequence="intergenic_variant",
        ),
        _annotation("both", "G1", "ENST1", biotype="protein_coding"),
        _annotation("both", "G2", "ENST2", biotype="lncRNA"),
        _annotation("other", "G3", "ENST3", biotype="processed_transcript"),
        _annotation("multi", "G4", "ENST4"),
        _annotation("multi", "G5", "ENST5"),
        _annotation("priority", "MANE", "ENST_MANE", mane="MANE_SELECT", tsl="5"),
        _annotation(
            "priority", "MANE", "ENST_NOT_MANE", canonical="YES", mane=None
        ),
        _annotation("priority", "CANON", "ENST_CANON", canonical="YES", tsl="5"),
        _annotation("priority", "CANON", "ENST_NOT_CANON", canonical=None, tsl="1"),
        _annotation("priority", "TSL", "ENST_TSL_3", tsl="3"),
        _annotation("priority", "TSL", "ENST_TSL_2", tsl="2"),
        _annotation("priority", "FEATURE", "ENST000002", tsl="1"),
        _annotation("priority", "FEATURE", "ENST000001", tsl="1"),
    ]

    result = filter_and_select_annotations(pl.DataFrame(rows))

    result_variants = (
        result
        .get_column("variant")
    )
    multi_gene = (
        result
        .filter(pl.col("variant") == "multi")
    )
    assert set(result_variants) == {"multi", "priority"}
    assert multi_gene.height == 2
    selected = dict(
        result
        .filter(pl.col("variant") == "priority")
        .select("gene", "feature")
        .iter_rows()
    )
    assert selected == {
        "MANE": "ENST_MANE",
        "CANON": "ENST_CANON",
        "TSL": "ENST_TSL_2",
        "FEATURE": "ENST000001",
    }


def test_downsample_is_variant_level_stratified_and_reproducible() -> None:
    rows = []
    for biotype in ("protein_coding", "lncRNA"):
        for index in range(10):
            for gene in ("A", "B"):
                rows.append(
                    {
                        "variant": f"{biotype}-{index}",
                        "biotype": biotype,
                        "label": "Benign" if index % 2 == 0 else "Likely_benign",
                        "gene": gene,
                    }
                )
    rows.extend(
        [
            {
                "variant": label,
                "biotype": "protein_coding",
                "label": label,
                "gene": gene,
            }
            for label in ("Pathogenic", "Likely_pathogenic")
            for gene in ("A", "B")
        ]
    )
    full = pl.DataFrame(rows)

    negative_labels = ["Benign", "Likely_benign"]
    first = downsample(
        full, neg_labels=negative_labels, neg_fraction=0.3, seed=42
    )
    second = downsample(
        full, neg_labels=negative_labels, neg_fraction=0.3, seed=42
    )

    assert first.equals(second)
    negative = first.filter(pl.col("label").is_in(negative_labels))
    sampled_per_biotype = (
        negative
        .select("variant", "biotype")
        .unique()
        .group_by("biotype")
        .len()
        .get_column("len")
    )
    annotations_per_variant = (
        negative
        .group_by("variant")
        .len()
        .get_column("len")
    )
    retained_positive_labels = set(
        first.filter(~pl.col("label").is_in(negative_labels)).get_column("label")
    )
    assert (sampled_per_biotype == 3).all()
    assert (annotations_per_variant == 2).all()
    assert retained_positive_labels == {"Pathogenic", "Likely_pathogenic"}


def test_negative_label_cli_defaults_only_when_not_supplied() -> None:
    required = [
        "--vep-output",
        "vep.tsv.gz",
        "--selected-variants",
        "selected.tsv.gz",
        "--output-prefix",
        "final",
    ]

    assert parse_args(required).neg_labels == ["low_pip"]
    assert parse_args(
        required
        + [
            "--neg-labels",
            "Benign",
            "--neg-labels",
            "Likely_benign",
        ]
    ).neg_labels == ["Benign", "Likely_benign"]


def test_cli_end_to_end_preserves_metadata_and_writes_gzip(
    tmp_path: Path,
) -> None:
    selected_rows = []
    annotations = []
    for index in range(4):
        for biotype, offset in (("protein_coding", 0), ("lncRNA", 10)):
            variant = f"v{offset + index}"
            selected_rows.append(
                {
                    "variant": variant,
                    "chromosome": "chr1",
                    "start": offset + index,
                    "end": offset + index + 1,
                    "ref": "A",
                    "alt": "T",
                    "pip": 0.01,
                    "label": "Benign" if index % 2 == 0 else "Likely_benign",
                    "cohort": "synthetic",
                }
            )
            annotations.append(
                _annotation(variant, f"G{offset + index}", f"ENST{offset + index}", biotype=biotype)
            )
    selected_rows.append(
        {
            "variant": "pathogenic",
            "chromosome": "chr2",
            "start": 20,
            "end": 21,
            "ref": "C",
            "alt": "G",
            "pip": 0.99,
            "label": "Pathogenic",
            "cohort": "synthetic",
        }
    )
    annotations.extend(
        [
            _annotation("pathogenic", "HIGH1", "ENST_HIGH1"),
            _annotation("pathogenic", "HIGH2", "ENST_HIGH2"),
        ]
    )
    selected_rows.append(
        {
            "variant": "likely-pathogenic",
            "chromosome": "chr2",
            "start": 21,
            "end": 22,
            "ref": "C",
            "alt": "T",
            "pip": None,
            "label": "Likely_pathogenic",
            "cohort": "synthetic",
        }
    )
    annotations.append(
        _annotation("likely-pathogenic", "HIGH3", "ENST_HIGH3")
    )

    selected_path = tmp_path / "selected.tsv.gz"
    vep_path = tmp_path / "vep.txt.gz"
    prefix = tmp_path / "final_variants"
    selected = pl.DataFrame(selected_rows)
    (
        selected
        .write_csv(selected_path, separator="\t", compression="gzip")
    )
    _write_vep(vep_path, annotations)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vep_comparisons.cli",
            "--vep-output",
            str(vep_path),
            "--selected-variants",
            str(selected_path),
            "--output-prefix",
            str(prefix),
            "--neg-labels",
            "Benign",
            "--neg-labels",
            "Likely_benign",
            "--neg-fraction",
            "0.5",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    full_path = tmp_path / "final_variants.tsv.gz"
    downsampled_path = tmp_path / "final_variants.downsampled.tsv.gz"
    assert full_path.exists()
    assert downsampled_path.exists()
    assert full_path.read_bytes().startswith(b"\x1f\x8b")
    assert downsampled_path.read_bytes().startswith(b"\x1f\x8b")

    full = pl.read_csv(full_path, separator="\t")
    downsampled = pl.read_csv(downsampled_path, separator="\t")
    full_variant_count = (
        full
        .get_column("variant")
        .n_unique()
    )
    downsampled_variant_count = (
        downsampled
        .get_column("variant")
        .n_unique()
    )
    cohorts = (
        full
        .get_column("cohort")
        .unique()
        .to_list()
    )
    retained_positive = (
        downsampled
        .filter(pl.col("label").is_in(["Pathogenic", "Likely_pathogenic"]))
    )
    assert full.height == 11
    assert full_variant_count == 10
    assert downsampled.height == 7
    assert downsampled_variant_count == 6
    assert full.columns[:9] == list(selected_rows[0])
    assert cohorts == ["synthetic"]
    assert set(retained_positive.get_column("label")) == {
        "Pathogenic",
        "Likely_pathogenic",
    }
    assert retained_positive.height == 3
    assert "input: 11 rows, 10 variants" in completed.stdout
    assert "filtered for protein-coding and lncRNA: 11 rows, 10 variants" in completed.stdout
    assert "full: 11 rows, 10 variants" in completed.stdout
    assert "downsampled: 7 rows, 6 variants" in completed.stdout


def test_target_gene_join_drops_redundant_right_variant_key(
    tmp_path: Path,
) -> None:
    selected_path = tmp_path / "selected.tsv.gz"
    vep_path = tmp_path / "vep.txt.gz"
    prefix = tmp_path / "final_variants"
    pl.DataFrame(
        [
            {
                "variant": "v1",
                "chromosome": "chr1",
                "start": 0,
                "end": 1,
                "ref": "A",
                "alt": "T",
                "pip": 0.99,
                "label": "high_pip",
                "target_gene": "G1",
            }
        ]
    ).write_csv(selected_path, separator="\t", compression="gzip")
    _write_vep(
        vep_path,
        [
            _annotation("v1", "G1", "ENST1"),
            _annotation("v1", "G2", "ENST2"),
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "vep_comparisons.cli",
            "--vep-output",
            str(vep_path),
            "--selected-variants",
            str(selected_path),
            "--has-target-genes",
            "--output-prefix",
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = pl.read_csv(f"{prefix}.tsv.gz", separator="\t")
    assert result.get_column("gene").to_list() == ["G1"]
    assert result.get_column("target_gene").to_list() == ["G1"]
    assert "variant_right" not in result.columns
