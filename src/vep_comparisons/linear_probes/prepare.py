"""Prepare position-level cohorts and fixed labels for frozen linear probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

KEYS = ["variant", "gene", "feature"]
ANNOTATIONS = ["phylop", "gnocchi_z"]
DNA_RNA_MODELS = ["nt_100", "nt_650", "rinalmo", "orthrus"]
PROTEIN_MODELS = ["esmc", "saprot"]
SCORE_FILES = {
    "nt_100": "ntv3_100m_pre_8192_centered/all_shards.tsv.gz",
    "nt_650": "ntv3_650m_pre_8192_centered/all_shards.tsv.gz",
    "rinalmo": "rinalmo_giga.tsv.gz",
    "orthrus": "orthrus_mlm_6_track/all_shards.tsv.gz",
    "esmc": "collated_variants.esmc_600m.tsv.gz",
    "saprot": "collated_variants.saprot_650m.tsv.gz",
}
CONTEXT_COLUMNS = [
    *KEYS, "chromosome", "start", "end", "ref", "alt", "biotype",
    "consequence", "cdna_position", "cds_position", "protein_position", "amino_acids",
]
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


def restrict_source_population(
    data: pl.DataFrame, ukbb: pl.DataFrame, traits: list[str]
) -> pl.DataFrame:
    """Apply the Table 2 transcript and selected-UKBB-trait eligibility rules."""
    eligible = (
        ukbb
        .filter(pl.col("trait").is_in(traits))
        .group_by("variant")
        .agg(ukbb_pip=pl.col("pip").max())
        .filter((pl.col("ukbb_pip") <= 0.1) | (pl.col("ukbb_pip") >= 0.5))
        .select("variant")
        .with_columns(_ukbb_eligible=pl.lit(True))
    )
    return (
        data
        .join(eligible, on="variant", how="left")
        .filter(
            pl.col("alt") != ".",
            pl.col("cdna_position").is_not_null(),
            pl.any_horizontal(
                pl.col("_ukbb_eligible").fill_null(False),
                pl.col("multisusie_label", "eqtl_label", "clinvar_label").is_not_null(),
            ),
        )
        .drop("_ukbb_eligible")
    )


def load_population(variants: Path, scores_dir: Path, ukbb_dir: Path) -> pl.DataFrame:
    print(f"Reading annotations: {variants}", flush=True)
    data = pl.read_csv(
        variants, separator="\t", null_values="-", infer_schema_length=10000,
        schema_overrides={"multisusie_pip": pl.Float64, "eqtl_pip": pl.Float64},
    )
    print("Reading selected-trait UKBB evidence", flush=True)
    ukbb = pl.read_csv(
        ukbb_dir / "final_UKBB_94traits_release1.hg38.selected_transcript.txt.gz",
        separator="\t", columns=["variant", "trait", "pip"],
    )
    traits = pl.read_csv(
        ukbb_dir / "UKBB_94traits_release1.traits.filtered", separator="\t",
    )["trait"].to_list()
    data = restrict_source_population(data, ukbb, traits).select(
        *CONTEXT_COLUMNS, *ANNOTATIONS,
    )
    for model, filename in SCORE_FILES.items():
        print(f"Reading {model} scores", flush=True)
        scores = (
            pl.read_csv(scores_dir / filename, separator="\t", columns=[*KEYS, "score"])
            # Table 2 uses log(ref) - log(alt) for every included checkpoint.
            .rename({"score": f"{model}_score"})
            .with_columns(-pl.col(f"{model}_score"))
        )
        data = data.join(scores, on=KEYS, how="left", validate="m:1")
    return data


def select_cohorts(data: pl.DataFrame) -> pl.DataFrame:
    """Select a context by availability then lexical IDs, never score magnitude.

    Finite scores certify that the existing scoring workflows could construct
    the corresponding model inputs, including SaProt's structure tokens.
    """
    missense = (
        (pl.col("biotype") == "protein_coding")
        & pl.col("consequence").str.split(",").list.contains("missense_variant")
    )
    eligible = (
        data
        .filter(
            pl.col("chromosome").is_in(AUTOSOMES),
            missense | (pl.col("biotype") == "lncRNA"),
            pl.col("ref").is_in(list("ACGT")),
            pl.col("alt").is_in(list("ACGT")),
            pl.col("ref") != pl.col("alt"),
            pl.col("end") == pl.col("start") + 1,
            pl.col("cdna_position").is_not_null(),
            pl.all_horizontal(pl.col(
                *ANNOTATIONS, *[f"{model}_score" for model in DNA_RNA_MODELS],
            ).is_finite()),
        )
        .with_columns(
            site_id=pl.format("{}:{}", "chromosome", pl.col("start") + 1),
            _shared=(missense & pl.all_horizontal(
                pl.col(*[f"{model}_score" for model in PROTEIN_MODELS]).is_finite(),
            )).fill_null(False),
        )
        .sort(["_shared", *KEYS], descending=[True, False, False, False])
        .unique("site_id", keep="first", maintain_order=True)
        .with_columns(context_id=pl.concat_str(KEYS, separator="|"))
    )
    return (
        pl.concat([
            eligible.with_columns(cohort=pl.lit("dna_rna")),
            eligible.filter(pl.col("_shared")).with_columns(cohort=pl.lit("shared_missense")),
        ])
        .drop("_shared")
        .sort("cohort", "chromosome", "start")
    )


def assign_chromosome_folds(cohorts: pl.DataFrame, n_folds: int = 5) -> pl.DataFrame:
    """Greedily balance relative sizes of both cohorts with a single mapping."""
    names = sorted(cohorts["cohort"].unique().to_list())
    counts = {
        (cohort, chromosome): n
        for cohort, chromosome, n in cohorts.group_by("cohort", "chromosome").len().iter_rows()
    }
    sizes = np.array([
        [counts.get((name, chromosome), 0) for name in names]
        for chromosome in AUTOSOMES
    ])
    relative = sizes / np.maximum(sizes.sum(axis=0), 1)
    loads = np.zeros((n_folds, len(names)))
    assignments = np.zeros(22, dtype=int)
    order = sorted(range(22), key=lambda i: (-relative[i].sum(), i))
    for i in order:
        # Increment in the sum of squared fold loads, equally weighting cohorts.
        cost = ((loads + relative[i]) ** 2 - loads ** 2).sum(axis=1)
        fold = int(np.argmin(cost))
        assignments[i] = fold
        loads[fold] += relative[i]
    return pl.DataFrame({
        "chromosome": AUTOSOMES, "outer_fold": assignments,
        **{f"{name}_sites": sizes[:, i] for i, name in enumerate(names)},
    })


def sample_tail(data: pl.DataFrame, cap: int, seed: int) -> pl.DataFrame:
    """Largest-remainder chromosome quotas followed by seeded sampling."""
    if data.height <= cap:
        return data.sort("site_id")
    groups = data.partition_by("chromosome", as_dict=True)
    chromosomes = sorted(groups)
    quotas = np.array([groups[c].height for c in chromosomes]) * cap / data.height
    sizes = np.floor(quotas).astype(int)
    order = np.argsort(-(quotas - sizes), kind="stable")
    sizes[order[:cap - sizes.sum()]] += 1
    rng = np.random.default_rng(seed)
    sampled = []
    for chromosome, n in zip(chromosomes, sizes, strict=True):
        group = groups[chromosome].sort("site_id")
        indices = rng.choice(group.height, size=int(n), replace=False)
        sampled.append(group[indices.tolist()])
    return pl.concat(sampled).sort("site_id")


def prepare_manifest(
    data: pl.DataFrame, *, seed: int = 20261005, cap_per_tail: int = 10000,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    cohorts = select_cohorts(data)
    folds = assign_chromosome_folds(cohorts)
    selected = []
    thresholds = []
    for (cohort,), sites in cohorts.partition_by("cohort", as_dict=True).items():
        for annotation in ANNOTATIONS:
            values = sites[annotation]
            low = values.quantile(0.25, interpolation="nearest")
            high = values.quantile(0.75, interpolation="nearest")
            labeled = (
                sites
                .with_columns(
                    task=pl.lit(annotation),
                    label=pl.when(pl.col(annotation) <= low).then(0)
                    .when(pl.col(annotation) >= high).then(1).otherwise(None),
                )
                .filter(pl.col("label").is_not_null())
            )
            counts = {}
            for label in (0, 1):
                tail = labeled.filter(pl.col("label") == label)
                sample = sample_tail(tail, cap_per_tail, seed)
                selected.append(sample)
                counts[f"tail_{label}_sites"] = tail.height
                counts[f"selected_{label}_sites"] = sample.height
            thresholds.append({
                "cohort": cohort, "task": annotation, "population_sites": sites.height,
                "q25": low, "q75": high, "quantile_interpolation": "nearest",
                "seed": seed, "cap_per_tail": cap_per_tail, **counts,
            })
    manifest = (
        pl.concat(selected)
        .join(folds.select("chromosome", "outer_fold"), on="chromosome", how="left")
        .sort("cohort", "task", "outer_fold", "chromosome", "start")
    )
    counts = (
        manifest
        .group_by("cohort", "task", "outer_fold", "label")
        .len(name="sites")
        .sort("cohort", "task", "outer_fold", "label")
    )
    return manifest, pl.DataFrame(thresholds), folds, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--ukbb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--cap-per-tail", type=int, default=10000)
    args = parser.parse_args()
    assert args.cap_per_tail > 0
    data = load_population(args.variants, args.scores_dir, args.ukbb_dir)
    manifest, thresholds, folds, counts = prepare_manifest(
        data, seed=args.seed, cap_per_tail=args.cap_per_tail,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(args.output_dir / "manifest.parquet")
    thresholds.write_csv(args.output_dir / "thresholds.tsv", separator="\t")
    folds.write_csv(args.output_dir / "chromosome_folds.tsv", separator="\t")
    counts.write_csv(args.output_dir / "counts.tsv", separator="\t")
    sources = {
        "variants": args.variants,
        "ukbb_variants": args.ukbb_dir / "final_UKBB_94traits_release1.hg38.selected_transcript.txt.gz",
        "ukbb_traits": args.ukbb_dir / "UKBB_94traits_release1.traits.filtered",
        **{model: args.scores_dir / path for model, path in SCORE_FILES.items()},
    }
    pl.DataFrame({
        "input": list(sources), "path": [str(path.resolve()) for path in sources.values()],
    }).write_csv(args.output_dir / "sources.tsv", separator="\t")
    print(thresholds)
    print(f"Saved {manifest.height} memberships, {manifest['site_id'].n_unique()} unique sites")


if __name__ == "__main__":
    main()
