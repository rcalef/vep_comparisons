import gzip
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from vep_comparisons.linear_probes.prepare import (
    DNA_RNA_MODELS,
    PROTEIN_MODELS,
    SCORE_FILES,
    load_population,
    prepare_manifest,
    restrict_source_population,
    sample_tail,
    select_cohorts,
)


def population() -> pl.DataFrame:
    rows = []
    for chromosome in range(1, 23):
        for position in range(40):
            rows.append({
                "variant": f"chr{chromosome}:{position + 1}:A:C",
                "gene": "g1", "feature": "t1", "chromosome": f"chr{chromosome}",
                "start": position, "end": position + 1, "ref": "A", "alt": "C",
                "biotype": "protein_coding" if position % 2 == 0 else "lncRNA",
                "consequence": "missense_variant", "cdna_position": position + 1,
                "cds_position": position + 1, "protein_position": position // 3 + 1,
                "amino_acids": "A/C", "phylop": float(position),
                "gnocchi_z": float(position // 3),
                **{f"{m}_score": float(position) for m in DNA_RNA_MODELS + PROTEIN_MODELS},
            })
    return pl.DataFrame(rows)


def test_source_filters_match_table2_eligibility() -> None:
    data = pl.DataFrame({
        "variant": ["a", "b", "c", "d", "e", "f", "g"],
        "alt": ["C"] * 5 + [".", "C"],
        "cdna_position": [1] * 6 + [None],
        "multisusie_label": [None, None, "low_pip", None, None, None, None],
        "eqtl_label": [None, None, None, "low_pip", None, "high_pip", "high_pip"],
        "clinvar_label": [None, None, None, None, "Pathogenic", None, None],
    })
    ukbb = pl.DataFrame({
        "variant": ["a", "a", "b", "c"],
        "trait": ["allowed", "excluded", "excluded", "allowed"],
        "pip": [0.05, 0.99, 0.9, 0.3],
    })
    result = restrict_source_population(data, ukbb, ["allowed"])
    assert sorted(result["variant"].to_list()) == ["a", "c", "d", "e"]


def test_site_deduplication_and_shared_context_priority() -> None:
    original = population().head(2)
    # The lexically first transcript lacks SaProt; prefer the shared context.
    missing = original.with_columns(feature=pl.lit("t0"), saprot_score=pl.lit(None))
    alternate = original.with_columns(alt=pl.lit("G"), variant=pl.col("variant") + "alt")
    data = pl.concat([original, missing, alternate, original], how="vertical_relaxed")
    cohorts = select_cohorts(data)
    assert cohorts.filter(pl.col("cohort") == "dna_rna").height == 2
    shared = cohorts.filter(pl.col("cohort") == "shared_missense")
    assert shared.height == 1
    assert shared["feature"].item() == "t1"
    broad = cohorts.filter(pl.col("site_id") == shared["site_id"].item())
    assert broad["context_id"].n_unique() == 1
    assert_frame_equal(cohorts, select_cohorts(data.reverse()))
    changed = select_cohorts(data.with_columns(
        phylop=-pl.col("phylop"), gnocchi_z=-pl.col("gnocchi_z"),
        **{f"{m}_score": -pl.col(f"{m}_score") for m in SCORE_FILES},
    ))
    assert_frame_equal(cohorts.select("cohort", "context_id"), changed.select("cohort", "context_id"))


def test_cohort_eligibility_requires_only_relevant_models() -> None:
    data = population().head(8).with_columns(
        chromosome=pl.when(pl.col("start") == 0).then(pl.lit("chrX")).otherwise(pl.col("chromosome")),
        nt_650_score=pl.when(pl.col("start") == 1).then(None).otherwise(pl.col("nt_650_score")),
        gnocchi_z=pl.when(pl.col("start") == 2).then(None).otherwise(pl.col("gnocchi_z")),
        saprot_score=pl.when(pl.col("start") == 4).then(None).otherwise(pl.col("saprot_score")),
        rinalmo_score=pl.when(pl.col("start") == 5).then(float("nan")).otherwise(pl.col("rinalmo_score")),
        evo_score=pl.lit(None),
    )
    result = select_cohorts(data)
    assert result.filter(pl.col("cohort") == "dna_rna")["start"].to_list() == [3, 4, 6, 7]
    assert result.filter(pl.col("cohort") == "shared_missense")["start"].to_list() == [6]


def test_thresholds_ties_sampling_and_chromosome_separation() -> None:
    data = population()
    manifest, thresholds, folds, counts = prepare_manifest(data, cap_per_tail=75)
    repeated = prepare_manifest(data.sample(fraction=1, shuffle=True, seed=31), cap_per_tail=75)
    for actual, expected in zip((manifest, thresholds, folds, counts), repeated, strict=True):
        assert_frame_equal(actual, expected)
    assert manifest.unique(["cohort", "task", "site_id"]).height == manifest.height
    assert folds.height == 22
    assert folds["outer_fold"].n_unique() == 5
    assert manifest.group_by("chromosome").agg(pl.col("outer_fold").n_unique())["outer_fold"].max() == 1
    for fold in range(5):
        train = set(manifest.filter(pl.col("outer_fold") != fold)["chromosome"])
        test = set(manifest.filter(pl.col("outer_fold") == fold)["chromosome"])
        assert train.isdisjoint(test)
        assert train and test
    cohorts = select_cohorts(data)
    for row in thresholds.iter_rows(named=True):
        values = cohorts.filter(pl.col("cohort") == row["cohort"])[row["task"]]
        assert row["q25"] == values.quantile(0.25, interpolation="nearest")
        assert row["q75"] == values.quantile(0.75, interpolation="nearest")
        assert row["tail_0_sites"] == (values <= row["q25"]).sum()
        assert row["tail_1_sites"] == ((values >= row["q75"]) & (values > row["q25"])).sum()
        assert row["selected_0_sites"] == row["selected_1_sites"] == 75
    assert counts["sites"].sum() == manifest.height
    full = prepare_manifest(data, cap_per_tail=10000)[1]
    assert_frame_equal(thresholds.select("cohort", "task", "q25", "q75"), full.select("cohort", "task", "q25", "q75"))
    tied = prepare_manifest(data.with_columns(gnocchi_z=pl.lit(1.0)))[1].filter(pl.col("task") == "gnocchi_z")
    # Match Table 2's lower-tail precedence if quartile boundaries coincide.
    assert tied["tail_1_sites"].sum() == 0


def test_sampling_quotas_and_small_tails() -> None:
    data = pl.DataFrame({
        "chromosome": ["chr1"] * 60 + ["chr2"] * 30 + ["chr3"] * 10,
        "site_id": [str(i) for i in range(100)],
    })
    sampled = sample_tail(data, 17, 42)
    assert sampled.height == sampled["site_id"].n_unique() == 17
    n = sampled.group_by("chromosome").len().sort("chromosome")["len"].to_numpy()
    assert np.all(np.abs(n - np.array([60, 30, 10]) * 17 / 100) < 1)
    assert_frame_equal(sampled, sample_tail(data.reverse(), 17, 42))
    assert_frame_equal(sample_tail(data, 200, 42), data.sort("site_id"))


def test_file_loading_retains_context_and_negates_scores(tmp_path: Path) -> None:
    data = population().head(2).with_columns(
        multisusie_pip=pl.lit(None, dtype=pl.Float64),
        eqtl_pip=pl.lit(None, dtype=pl.Float64),
        multisusie_label=pl.lit(None, dtype=pl.String),
        eqtl_label=pl.lit(None, dtype=pl.String),
        clinvar_label=pl.lit("Pathogenic"),
    )
    variants = tmp_path / "variants.tsv"
    data.drop([f"{model}_score" for model in SCORE_FILES]).write_csv(variants, separator="\t")
    for filename in SCORE_FILES.values():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        scores = data.select("variant", "gene", "feature").with_columns(score=pl.lit(-2.0))
        with gzip.open(path, "wt") as handle:
            handle.write(scores.write_csv(separator="\t"))
    (tmp_path / "UKBB_94traits_release1.traits.filtered").write_text("trait\nallowed\n")
    with gzip.open(tmp_path / "final_UKBB_94traits_release1.hg38.selected_transcript.txt.gz", "wt") as handle:
        handle.write("variant\ttrait\tpip\nunused\tallowed\t0.9\n")
    result = load_population(variants, tmp_path, tmp_path)
    assert result.height == 2
    for model in SCORE_FILES:
        assert result[f"{model}_score"].to_list() == [2.0, 2.0]
