"""Command-line curation of selected variants annotated by Ensembl VEP."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


REQUIRED_SELECTED_COLUMNS = (
    "variant",
    "chromosome",
    "start",
    "end",
    "ref",
    "alt",
    "pip",
    "label",
)

VEP_COLUMNS = (
    "variant",
    "gene",
    "feature",
    "consequence",
    "cdna_position",
    "cds_position",
    "protein_position",
    "amino_acids",
    "symbol",
    "biotype",
    "canonical",
    "mane",
    "tsl",
)

OUTPUT_ANNOTATION_COLUMNS = (
    "gene",
    "feature",
    "consequence",
    "cdna_position",
    "cds_position",
    "protein_position",
    "amino_acids",
    "symbol",
    "biotype",
)

REMOVED_CONSEQUENCES = (
    "downstream_gene_variant",
    "upstream_gene_variant",
    "intron_variant",
    "intergenic_variant",
)

KEPT_BIOTYPES = ("protein_coding", "lncRNA")


def read_vep(path: Path) -> pl.DataFrame:
    return (
        pl.read_csv(
            path,
            separator="\t",
            has_header=True,
            comment_prefix="##",
            null_values="-",
            infer_schema=False,
        )
        .rename({"#Uploaded_variation": "variant"})
        .rename(lambda column: column.lower())
    )


def filter_and_select_annotations(vep: pl.DataFrame) -> pl.DataFrame:
    annotations = (
        vep
        .select(VEP_COLUMNS)
        .with_columns(consequence_terms=pl.col("consequence").str.split(","))
        # Remove variants whose predicted consequences are only those
        # in the removed categories (intergenic, intron, up/downstream of a gene)
        .filter(
            pl.col("consequence_terms")
            .list.set_intersection(REMOVED_CONSEQUENCES)
            .list.len()
            == 0
        )
        .drop("consequence_terms")
    )

    eligible_variants = (
        annotations
        .group_by("variant")
        .agg(
            has_protein_coding=pl.col("biotype").eq("protein_coding").any(),
            has_lncrna=pl.col("biotype").eq("lncRNA").any(),
        )
        # Select variants that are in a protein coding gene or a lncRNA, but
        # not both
        .filter(pl.col("has_protein_coding") != pl.col("has_lncrna"))
        .select("variant")
    )

    return (
        annotations
        .join(eligible_variants, on="variant", how="semi")
        .filter(pl.col("biotype").is_in(KEPT_BIOTYPES))
        .with_columns(
            mane_rank=pl.col("mane").is_null().cast(pl.UInt8),
            canonical_rank=(
                (pl.col("canonical") != "YES")
                .fill_null(True)
                .cast(pl.UInt8)
            ),
            tsl_rank=pl.col("tsl").str.extract(r"^(\d+)", 1).cast(
                pl.UInt8, strict=False
            ),
        )
        .sort(
            [
                "variant",
                "gene",
                "mane_rank",
                "canonical_rank",
                "tsl_rank",
                "feature",
            ],
            nulls_last=True,
        )
        .unique(["variant", "gene"], keep="first", maintain_order=True)
        .select("variant", *OUTPUT_ANNOTATION_COLUMNS)
    )


def sort_variants(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame
        .with_columns(
            chromosome_number=pl.col("chromosome")
            .str.extract(r"^chr(\d+)$", 1)
            .cast(pl.UInt8)
        )
        .sort(["chromosome_number", "start", "variant", "gene"])
        .drop("chromosome_number")
    )


def downsample(
    full: pl.DataFrame, *, low_pip_fraction: float, seed: int
) -> pl.DataFrame:
    low_pip_variants = (
        full
        .filter(pl.col("label") == "low_pip")
        .select("variant", "biotype")
        .unique(maintain_order=True)
        .sort(["biotype", "variant"])
    )

    sampled_ids: list[pl.DataFrame] = []
    biotype_strata = (
        low_pip_variants
        .partition_by("biotype", maintain_order=True)
    )
    for stratum in biotype_strata:
        sampled_ids.append(
            stratum
            .sample(
                fraction=low_pip_fraction,
                with_replacement=False,
                shuffle=True,
                seed=seed,
            )
            .select("variant")
        )

    sampled_low_pip = (
        full
        .filter(pl.col("label") == "low_pip")
        .join(
            pl.concat(sampled_ids)
            if sampled_ids
            else pl.DataFrame(schema={"variant": pl.String}),
            on="variant",
            how="semi",
        )
    )
    high_pip = (
        full
        .filter(pl.col("label") == "high_pip")
    )
    return pl.concat([high_pip, sampled_low_pip])


def report(
    stage: str,
    frame: pl.DataFrame,
    report_pip_by_biotype: bool = False,
) -> None:
    variant_count = (
        frame
        .get_column("variant")
        .n_unique()
    )
    print(
        f"{stage}: {frame.height:,} rows, "
        f"{variant_count:,} variants"
    )
    if report_pip_by_biotype:
        counts = (
            frame
            .pivot(
                on="biotype",
                index="label",
                values="variant",
                aggregate_function="len",
                sort_columns=True,
            )
            .fill_null(0)
            .sort("label")
        )
        print("counts per label:")
        print(counts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate selected variants from tabular Ensembl VEP output."
    )
    parser.add_argument("--vep-output", required=True, type=Path)
    parser.add_argument("--selected-variants", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--low-pip-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    selected = pl.read_csv(args.selected_variants, separator="\t", has_header=True)
    # Selecting these columns makes Polars report any missing canonical columns.
    _ = selected.select(REQUIRED_SELECTED_COLUMNS)

    vep = read_vep(args.vep_output)
    annotations = filter_and_select_annotations(vep)
    full = sort_variants(
        selected
        .join(
            annotations,
            on="variant",
            how="inner",
            validate="1:m",
            maintain_order="left",
        )
    )
    downsampled = sort_variants(
        downsample(full, low_pip_fraction=args.low_pip_fraction, seed=args.seed)
    )

    full_path = Path(f"{args.output_prefix}.tsv.gz")
    (
        full
        .write_csv(full_path, separator="\t", compression="gzip", null_value="-")
    )

    downsampled_path = Path(f"{args.output_prefix}.downsampled.tsv.gz")
    (
        downsampled
        .write_csv(downsampled_path, separator="\t", compression="gzip", null_value="-")
    )

    report("input", vep)
    report("filtered for protein-coding and lncRNA", annotations)
    report("full", full, report_pip_by_biotype=True)
    report("downsampled", downsampled, report_pip_by_biotype=True)


if __name__ == "__main__":
    main()
