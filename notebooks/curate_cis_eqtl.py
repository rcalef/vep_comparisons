import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import polars as pl
    import seaborn as sns

    return Path, pl, sns


@app.cell
def _(Path):
    data_dir = Path("/orcd/data/manoli/001/rcalef/data/vep_comparisons/variants/cis_eqtl")

    raw_data_dir = data_dir / "GTEx_Analysis_v11_eQTL"

    paths = sorted(raw_data_dir.iterdir())
    tissues = [x.name.split(".")[0] for x in paths]

    print(len(tissues))
    tissues[:5]
    return data_dir, paths


@app.cell
def _(paths, pl):
    keep_cols = [
        "phenotype_id",
        "variant_id",
        "pip",
        "cs_id",
        "afc",
    ]

    all_variants = (
        pl.concat(
            [
                pl.scan_parquet(path)
                .select(keep_cols)
                # Retain every credible-set member. In particular, variants without
                # an AFC must survive until the PIP thresholds are applied below so
                # that low-PIP members remain eligible as negatives.
                .with_columns(tissue=pl.lit(path.name.split(".")[0]))
                for path in paths
            ],
            how="vertical",
        )
        .group_by("phenotype_id", "variant_id")
        # Select the maximum-PIP observation for each gene/variant pair while also
        # retaining its best fine-mapped observation, if one exists. The latter
        # preserves the previous positive set; the former lets the negative cutoff
        # consider every credible-set membership. Aggregating directly avoids
        # materializing list columns for all input rows.
        .agg(
            pl.col("pip").max(),
            pl.col("cs_id").sort_by("pip", descending=True).first(),
            pl.col("afc").sort_by("pip", descending=True).first(),
            pl.col("tissue").sort_by("pip", descending=True).first(),
            pl.col("pip")
            .filter(pl.col("afc").is_not_null())
            .max()
            .alias("fine_mapped_pip"),
            pl.col("afc")
            .filter(pl.col("afc").is_not_null())
            .sort_by(
                pl.col("pip").filter(pl.col("afc").is_not_null()),
                descending=True,
            )
            .first()
            .alias("fine_mapped_afc"),
            pl.col("tissue")
            .filter(pl.col("afc").is_not_null())
            .sort_by(
                pl.col("pip").filter(pl.col("afc").is_not_null()),
                descending=True,
            )
            .first()
            .alias("fine_mapped_tissue"),
            pl.col("tissue").unique(maintain_order=True).alias("all_tissues"),
        )
        .with_columns(
            num_tissues=pl.col("all_tissues").list.len()
        )
        .collect()
    )
    all_variants.shape
    return (all_variants,)


@app.cell
def _(all_variants):
    all_variants.head()
    return


@app.cell
def _(all_variants):
    tissues_per_qtl = all_variants.get_column("num_tissues").value_counts().sort("num_tissues")
    tissues_per_qtl
    return (tissues_per_qtl,)


@app.cell
def _(sns, tissues_per_qtl):
    sns.barplot(x="num_tissues", y="count", data=tissues_per_qtl)
    return


@app.cell
def _(all_variants, pl):
    high_thresh = 0.5
    low_thresh = 0.1

    deduped_variants = (
        all_variants
        .with_columns(
            parts=pl.col("variant_id").str.split("_"),
        )
        .with_columns(
            chromosome=pl.col("parts").list.get(0),
            end=pl.col("parts").list.get(1).cast(pl.UInt32),
            ref=pl.col("parts").list.get(2),
            alt=pl.col("parts").list.get(3),
            all_tissues=pl.col("all_tissues").list.join(","),
            target_gene=pl.col("phenotype_id").str.split(".").list[0],
        )
        .with_columns(
            is_high_pip=pl.col("fine_mapped_pip") >= high_thresh,
            is_low_pip=pl.col("pip") <= low_thresh,
            variant=pl.col("chromosome") + ":" + pl.col("end").cast(pl.String) + ":" + pl.col("ref") + ":" + pl.col("alt"),
        )
        .filter(
            # Preserve the existing positive definition while allowing non-fine-
            # mapped credible-set members to enter the low-PIP negative pool.
            pl.col("is_high_pip") | pl.col("is_low_pip"),
            pl.col("ref").str.len_chars() == 1,
            pl.col("alt").str.len_chars() == 1,
            pl.col("chromosome").str.contains(r"chr([\d]+)")
        )
        .with_columns(
            start=pl.col("end")-1,
            pip=(
                pl.when(pl.col("is_high_pip"))
                .then(pl.col("fine_mapped_pip"))
                .otherwise(pl.col("pip"))
            ),
            afc=(
                pl.when(pl.col("is_high_pip"))
                .then(pl.col("fine_mapped_afc"))
                .otherwise(pl.col("afc"))
            ),
            tissue=(
                pl.when(pl.col("is_high_pip"))
                .then(pl.col("fine_mapped_tissue"))
                .otherwise(pl.col("tissue"))
            ),
            label=(
                pl.when(pl.col("is_high_pip"))
                .then(pl.lit("high_pip"))
                .otherwise(pl.lit("low_pip"))
            ),
            chromosome_number=pl.col("chromosome").str.extract(r"chr([\d]+)"),

        )
        .sort(["chromosome_number", "start"])
        .drop(
            "parts",
            "cs_id",
            "chromosome_number",
            "phenotype_id",
            "fine_mapped_pip",
            "fine_mapped_afc",
            "fine_mapped_tissue",
            "is_high_pip",
            "is_low_pip",
            "variant_id",
        )
        .select(
            "variant",
            "chromosome",
            "start",
            "end",
            "ref",
            "alt",
            "pip",
            "label",
            "target_gene",
            "afc",
            "tissue",
            "all_tissues",
        )
    )

    print(deduped_variants.shape)
    deduped_variants.head()
    return (deduped_variants,)


@app.cell
def _(deduped_variants):
    deduped_variants.get_column("label").value_counts()
    return


@app.cell
def _(data_dir, deduped_variants):
    (
        deduped_variants
        .write_csv(
            data_dir / "filtered_eqtls.tsv.gz",
            separator="\t",
            include_header=True,
            compression="gzip",
        )
    )
    return


@app.cell
def _(deduped_variants, pl):
    # VEP custom input is 1-based; selected_variants remains BED-style.
    vep_variants = (
        deduped_variants
        .with_columns(
            chromosome=pl.col("chromosome").str.extract(r"chr([\d]+)"),
            allele=pl.col("ref") + "/" + pl.col("alt"),
            strand=pl.lit("+"),
        )
        .unique("variant", maintain_order=True)
        .select(
            "chromosome",
            (pl.col("start") + 1).alias("start"),
            "end",
            "allele",
            "strand",
            "variant",
        )
    )
    print(vep_variants.shape)
    vep_variants.head()
    return (vep_variants,)


@app.cell
def _(data_dir, vep_variants):
    (
        vep_variants
        .write_csv(
            data_dir / "vep" / "filtered_eqtls.for_vep.tsv.gz",
            separator="\t",
            include_header=False,
            compression="gzip",
        )
    )
    return


@app.cell
def _(data_dir, pl):
    vep_filtered = (
        pl.read_csv(data_dir / "final_variants.tsv.gz", separator="\t", null_values="-")
    )
    vep_filtered.head()
    return (vep_filtered,)


@app.cell
def _(pl, vep_filtered):
    (
        vep_filtered
        .filter(pl.col("biotype") == "protein_coding")
        .get_column("consequence")
        .value_counts(sort=True)
    )
    return


@app.cell
def _(pl, vep_filtered):
    (
        vep_filtered
        .filter(
            pl.col("biotype") == "protein_coding",
            pl.col("protein_position").is_null()
        )
        .get_column("consequence")
        .value_counts(sort=True)
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
