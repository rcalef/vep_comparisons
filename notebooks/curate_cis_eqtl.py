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
def _(paths):
    paths[0]
    return


@app.cell
def _(paths, pl):
    keep_cols = [
        "phenotype_id",
        "variant_id",
        "pip",
        "cs_id",
        "afc",
    ]

    all_variants = []
    for path in paths:
        tissue_name = path.name.split(".")[0]
        all_variants.append(
            pl.read_parquet(path, columns=keep_cols)
            # Filter to the non-null allelic fold-change rows,
            # which corresponds to keeping the maximum PIP variant
            # per credible set (with some genes having multiple credible sets)
            .filter(pl.col("afc").is_not_null())
            .with_columns(tissue=pl.lit(tissue_name))
        )
    all_variants = (
        pl.concat(all_variants, how="vertical")
        .group_by("phenotype_id", "variant_id")
        .agg(pl.all().implode())
        .with_columns(
            num_tissues=pl.col("tissue").list.len()
        )
    )
    all_variants.shape
    return (all_variants,)


@app.cell
def _(all_variants):
    all_variants.head()
    return


@app.cell
def _(all_variants, pl):
    all_variants.filter(pl.col("num_tissues") > 1)
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
            selected_idx=pl.col("pip").list.arg_max(),
        )
        .with_columns(
            chromosome=pl.col("parts").list.get(0),
            end=pl.col("parts").list.get(1).cast(pl.UInt32),
            ref=pl.col("parts").list.get(2),
            alt=pl.col("parts").list.get(3),
            pip=pl.col("pip").list.get(pl.col("selected_idx")),
            afc=pl.col("afc").list.get(pl.col("selected_idx")),
            all_tissues=pl.col("tissue").list.join(","),
            tissue=pl.col("tissue").list.get(pl.col("selected_idx")),

        )
        .filter(
            (pl.col("pip") >= high_thresh) | (pl.col("pip") <= low_thresh),
            pl.col("ref").str.len_chars() == 1,
            pl.col("alt").str.len_chars() == 1,
            pl.col("chromosome").str.contains(r"chr([\d]+)")
        )
        .with_columns(
            start=pl.col("end")-1,
            label=(
                pl.when(pl.col("pip") >= high_thresh)
                .then(pl.lit("high_pip"))
                .otherwise(pl.lit("low_pip"))
            ),
            chromosome_number=pl.col("chromosome").str.extract(r"chr([\d]+)"),
        )
        .sort(["chromosome_number", "start"])
        .drop("selected_idx", "parts", "cs_id", "chromosome_number")
        .rename({
            "variant_id": "variant",
            "phenotype_id": "target_gene",
        })
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
        .select(
            "chromosome",
            (pl.col("start") + 1).alias("start"),
            "end",
            "allele",
            "strand",
            "variant",
        )
    )
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
def _():
    return


if __name__ == "__main__":
    app.run()
