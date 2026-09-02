import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import polars as pl

    return Path, pl


@app.cell
def _(Path):
    data_dir = Path("/path/to/vep_comparisons/")
    dataset_dir = data_dir / "variants" / "multisusie"
    return (dataset_dir,)


@app.cell
def _(dataset_dir, pl):
    # The uploaded identifier is the chr:pos:ref:alt variant identifier, not an rsID.
    variants = (
        pl.read_csv(dataset_dir / "pips.tsv", separator="\t", has_header=True)
        .rename({"rsid": "variant", "bp": "pos", "PIP": "pip"})
        .with_columns(variant_parts=pl.col("variant").str.split(":"))
        .with_columns(
            chromosome=pl.col("variant_parts").list[0],
            ref=pl.col("variant_parts").list[2],
            alt=pl.col("variant_parts").list[3],
            start=pl.col("pos") - 1,
            end=pl.col("pos"),
        )
        .drop("variant_parts")
    )
    variants.shape
    return (variants,)


@app.cell
def _(pl, variants):
    high_thresh = 0.5
    low_thresh = 0.1

    # Keep SNVs, take maximum PIP across populations/traits, and omit the
    # intermediate-PIP variants from this comparison.
    selected_variants = (
        variants
        .filter(
            pl.col("ref").str.len_chars() == 1,
            pl.col("alt").str.len_chars() == 1,
        )
        .group_by("variant")
        .agg(pl.all().get(pl.col("pip").arg_max()))
        .filter(
            pl.col("chromosome").str.contains(r"^chr[\d]+$"),
            (pl.col("pip") >= high_thresh) | (pl.col("pip") <= low_thresh),
        )
        .with_columns(
            label=pl.when(pl.col("pip") >= high_thresh)
            .then(pl.lit("high_pip"))
            .otherwise(pl.lit("low_pip")),
            chromosome_number=pl.col("chromosome").str.extract(r"chr([\d]+)"),
        )
        .sort(["chromosome_number", "start"])
        .drop("chromosome_number")
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
            dataset_dir / "filtered_variants.tsv.gz",
            separator="\t",
            include_header=True,
            compression="gzip",
        )
    )


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
            dataset_dir / "vep" / "filtered_variants.for_vep.tsv.gz",
            separator="\t",
            include_header=False,
            compression="gzip",
        )
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run Ensembl VEP on the prepared custom input, then use the shared curation
    command for transcript selection and low-PIP downsampling:

    ```bash
    dataset_dir="/path/to/vep_comparisons/variants/multisusie"

    vep \
      -i "${dataset_dir}/vep/filtered_variants.for_vep.tsv.gz" \
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
      --dir_cache /path/to/assemblies/hg38/vep \
      --force_overwrite \
      --dir /path/to/vep/ \
      --fork 8 \
      --verbose \
      -o "${dataset_dir}/vep/variant_effect_output.txt.gz" \
      2>&1 | tee "${dataset_dir}/vep/run.log"

    curate-vep-variants \
      --vep-output "${dataset_dir}/vep/variant_effect_output.txt.gz" \
      --selected-variants "${dataset_dir}/filtered_variants.tsv.gz" \
      --output-prefix "${dataset_dir}/final_variants"
    ```
    """)


if __name__ == "__main__":
    app.run()
