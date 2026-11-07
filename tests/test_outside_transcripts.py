from pathlib import Path

import polars as pl
import pytest

from vep_comparisons.curation.workflow import (
    VEP_COLUMNS,
    curate_variants,
    downsample,
    select_outside_transcript_annotations,
)


def vep_frame() -> pl.DataFrame:
    rows = [
        ("outside", "upstream_gene_variant"),
        ("outside", "downstream_gene_variant"),
        ("intergenic", "intergenic_variant"),
        ("mixed", "upstream_gene_variant"),
        ("mixed", "non_coding_transcript_exon_variant"),
        ("intron", "intron_variant"),
        ("compound", "intron_variant,non_coding_transcript_variant"),
        ("missing", None),
    ]
    return pl.DataFrame(
        [{**dict.fromkeys(VEP_COLUMNS), "variant": variant, "consequence": consequence}
         for variant, consequence in rows],
        schema={column: pl.String for column in VEP_COLUMNS},
    )


def test_outside_requires_all_annotations() -> None:
    result = select_outside_transcript_annotations(vep_frame())
    assert set(result["variant"]) == {"outside", "intergenic"}
    assert result["gene"].null_count() == 2
    assert result.filter(pl.col("variant") == "outside")["consequence"].item() == (
        "downstream_gene_variant,upstream_gene_variant"
    )


@pytest.mark.parametrize("neg_fraction", [0.0, 1.0])
def test_eqtl_max_pip_without_vep_gene(tmp_path: Path, neg_fraction: float) -> None:
    vep = vep_frame().rename({"variant": "#Uploaded_variation"})
    vep.write_csv(tmp_path / "vep.tsv", separator="\t", null_value="-")
    selected = pl.DataFrame({
        "variant": ["intergenic", "intergenic"], "chromosome": ["chr1", "chr1"],
        "start": [0, 0], "end": [1, 1], "ref": ["A", "A"], "alt": ["G", "G"],
        "pip": [0.01, 0.9], "label": ["low_pip", "high_pip"],
        "tissue": ["Liver", "Brain"], "afc": [0.1, 0.8],
        "target_gene": ["gene1", "gene2"],
    })
    selected.write_csv(tmp_path / "selected.tsv", separator="\t")
    full, sampled = curate_variants(
        vep_output=tmp_path / "vep.tsv", selected_variants=tmp_path / "selected.tsv",
        output_prefix=tmp_path / "out", has_target_genes=True,
        region="outside-transcripts", neg_fraction=neg_fraction,
    )
    for path in (full, sampled):
        result = pl.read_csv(path, separator="\t", null_values="-")
        assert result.height == 1
        assert result["target_gene"].item() == "gene2"
        assert result["pip"].item() == 0.9
        assert result["label"].item() == "high_pip"
        assert result["tissue"].item() == "Brain"
        assert result["afc"].item() == 0.8


def test_sampling_null_biotypes_keeps_positive_associations() -> None:
    frame = pl.DataFrame({
        "variant": ["a", "a", "b", "c"],
        "label": ["high_pip", "low_pip", "low_pip", "low_pip"],
        "biotype": [None] * 4,
    })
    result = downsample(frame, neg_labels=["low_pip"], neg_fraction=0.0,
                        seed=42, outside_transcripts=True)
    assert result["variant"].to_list() == ["a", "a"]


def test_four_dataset_driver_deduplicates_eqtl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    sources = {
        "ukbb_finucane": "UKBB_94traits_release1.hg38.filtered.bed.gz",
        "multisusie": "filtered_variants.tsv.gz",
        "clinvar": "filtered_clinvar.tsv.gz",
        "cis_eqtl": "filtered_eqtls.tsv.gz",
    }
    for directory, filename in sources.items():
        source = tmp_path / directory
        (source / "vep").mkdir(parents=True)
        vep_frame().rename({"variant": "#Uploaded_variation"}).write_csv(
            source / "vep" / "variant_effect_output.txt.gz",
            separator="\t", compression="gzip", null_value="-",
        )
        eqtl = directory == "cis_eqtl"
        n = 2 if eqtl else 1
        selected = pl.DataFrame({
            "variant": ["intergenic"] * n, "chromosome": ["chr1"] * n,
            "start": [0] * n, "end": [1] * n, "ref": ["A"] * n, "alt": ["G"] * n,
            "pip": [0.9] * n, "label": ["high_pip"] * n,
        })
        if eqtl:
            selected = selected.with_columns(target_gene=pl.Series(["gene1", "gene2"]))
        selected.write_csv(source / filename, separator="\t", compression="gzip")
    out = tmp_path / "outside"
    monkeypatch.setattr(sys, "argv", [
        "curate_outside_transcripts.py", "--variants-dir", str(tmp_path),
        "--output-dir", str(out), "--collate",
    ])
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/curate_outside_transcripts.py"),
                   run_name="__main__")
    collated = pl.read_csv(out / "collated_variants.tsv.gz", separator="\t", null_values="-")
    assert collated.height == 1
    assert collated["eqtl_target_gene"].item() == "gene1"
    assert collated["variant"].n_unique() == 1
    assert pl.read_csv(out / "counts.tsv", separator="\t").height == 8
