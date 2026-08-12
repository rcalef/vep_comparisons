import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell
def _(Path):
    data_dir = Path("/orcd/data/manoli/001/rcalef/data/vep_comparisons/")
    dataset_dir = data_dir / "variants" / "ukbb_finucane"
    return (dataset_dir,)


@app.cell
def _(dataset_dir, pl):
    col_defs = pl.read_csv(
        dataset_dir / "UKBB_94traits_release1.cols",
        separator="\t",
        has_header=False,
        new_columns=["name", "desc"],
    )
    column_names = (
        col_defs
        .get_column("name")
        .to_list()
    )
    variants = (
        pl.read_csv(
            dataset_dir / "UKBB_94traits_release1.bed.gz",
            separator="\t",
            has_header=False,
            new_columns=column_names,
            infer_schema_length=10000,
            schema_overrides={"column_2": pl.Float32, "column_3": pl.Float32},
        )
        .cast({"start": pl.Int32, "end": pl.Int32})
    )
    variants.shape
    return (variants,)


@app.cell
def _(pl, variants):
    # Keep SNVs passing the author-provided filters and retain the maximum-PIP
    # row when a variant was flagged by multiple methods or for multiple traits.
    pass_variants = (
        variants
        .filter(
            (pl.col("allele1").str.len_chars() == 1),
            (pl.col("allele2").str.len_chars() == 1),
            ~pl.col("LD_SV"),
            ~pl.col("LD_HWE"),
        )
        .group_by("variant")
        .agg(pl.all().get(pl.col("pip").arg_max()))
    )
    pass_variants.shape
    return (pass_variants,)


@app.cell
def _(dataset_dir, pass_variants, pl):
    # Only rsIDs can be remapped with the downloaded UCSC hg38 table.
    (
        pass_variants
        .filter(pl.col("rsid").str.starts_with("rs"))
        .write_csv(
            dataset_dir / "UKBB_94traits_release1.filtered.bed.gz",
            separator="\t",
            include_header=False,
            compression="gzip",
        )
    )
    return


@app.cell
def _(dataset_dir, pl):
    hg38_rsids = pl.read_csv(
        dataset_dir / "hg38_rsids.tsv.gz",
        separator="\t",
        has_header=False,
        columns=[0, 1, 2, 3, 4],
        new_columns=["chromosome", "start", "end", "rsid", "hg38_ref"],
    )
    return (hg38_rsids,)


@app.cell
def _(hg38_rsids, pass_variants, pl):
    len_before = len(pass_variants)
    mapped_variants = (
        pass_variants
        .drop(["chromosome", "start", "end"])
        .join(hg38_rsids, on="rsid", how="inner")
        .filter(pl.col("hg38_ref") == pl.col("allele1"))
        # hg38_ref was only needed for the reference check. Drop it before
        # assigning the canonical selected-variant allele names.
        .drop("hg38_ref")
        .rename({"allele1": "ref", "allele2": "alt"})
        .with_columns(
            variant=(
                pl.col("chromosome")
                + ":"
                + pl.col("end").cast(pl.String)
                + ":"
                + pl.col("ref")
                + ":"
                + pl.col("alt")
            ),
            chromosome_number=pl.col("chromosome").str.extract(r"chr([\d]+)"),
        )
        .sort(["chromosome_number", "start"])
        .drop("chromosome_number")
    )
    print(f"{len_before} -> {len(mapped_variants)}")
    return (mapped_variants,)


@app.cell
def _(mapped_variants, pl):
    high_thresh = 0.5
    low_thresh = 0.1
    selected_variants = (
        mapped_variants
        .filter(
            pl.col("chromosome").str.contains(r"^chr[\d]+$"),
            (pl.col("pip") >= high_thresh) | (pl.col("pip") <= low_thresh),
        )
        .with_columns(
            label=pl.when(pl.col("pip") >= high_thresh)
            .then(pl.lit("high_pip"))
            .otherwise(pl.lit("low_pip"))
        )
    )
    (
        selected_variants
        .select(
            "variant", "chromosome", "start", "end", "ref", "alt", "pip", "label"
        )
        .head()
    )
    return (selected_variants,)


@app.cell
def _(dataset_dir, selected_variants):
    (
        selected_variants
        .write_csv(
            dataset_dir / "UKBB_94traits_release1.hg38.filtered.bed.gz",
            separator="\t",
            include_header=True,
            compression="gzip",
        )
    )
    return


@app.cell
def _(pl, selected_variants):
    # VEP custom input is 1-based; selected_variants remains BED-style.
    vep_variants = (
        selected_variants
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
def _(dataset_dir, vep_variants):
    (
        vep_variants
        .write_csv(
            dataset_dir / "vep" / "UKBB_94traits_release1.hg38.filtered.txt.gz",
            separator="\t",
            include_header=False,
            compression="gzip",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run Ensembl VEP on the prepared custom input, then use the shared curation
    command for transcript selection and low-PIP downsampling:

    ```bash
    dataset_dir="/storage/data/vep_comparisons/variants/ukbb_finucane"

    vep \
      -i "${dataset_dir}/vep/UKBB_94traits_release1.hg38.filtered.txt.gz" \
      --check_ref \
      --tab \
      --mane \
      --canonical \
      --af_gnomade \
      --af_gnomadg \
      --max_af \
      --biotype \
      --tsl \
      --symbol \
      --compress_output gzip \
      --cache \
      --dir_cache /storage/data/assemblies/hg38/vep \
      --force_overwrite \
      --dir /storage/vep/ \
      --fork 8 \
      --verbose \
      -o "${dataset_dir}/vep/variant_effect_output.txt.gz" \
      2>&1 | tee "${dataset_dir}/vep/run.log"

    curate-vep-variants \
      --vep-output "${dataset_dir}/vep/variant_effect_output.txt.gz" \
      --selected-variants "${dataset_dir}/UKBB_94traits_release1.hg38.filtered.bed.gz" \
      --output-prefix "${dataset_dir}/final_UKBB_94traits_release1.hg38"
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
