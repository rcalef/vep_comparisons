import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns

    from pysam import FastaFile

    return Path, np, pl


@app.cell
def _(Path):
    data_dir = Path("/orcd/data/manoli/001/rcalef/data/vep_comparisons/")
    return (data_dir,)


@app.cell
def _(data_dir, pl):
    col_defs = pl.read_csv(
        data_dir / "variants" / "release1.1" / "UKBB_94traits_release1.cols",
        separator="\t",
        has_header=False,
        new_columns=["name", "desc"],
    )
    col_defs
    return (col_defs,)


@app.cell
def _(col_defs, data_dir, pl):
    variants = (
        pl.read_csv(
            data_dir / "variants" / "release1.1" / "UKBB_94traits_release1.bed.gz",
            separator="\t",
            has_header=False,
            new_columns=col_defs.get_column("name").to_list(),
            infer_schema_length=10000,
            schema_overrides={
                "column_2": pl.Float32,
                "column_3": pl.Float32,
            },
        )
        .cast({"start": pl.Int32, "end": pl.Int32})
    )

    variants.shape
    return (variants,)


@app.cell
def _(variants):
    variants.get_column("LD_SV").value_counts()
    return


@app.cell
def _(variants):
    variants.get_column("LD_HWE").value_counts()
    return


@app.cell
def _(variants):
    variants.get_column("method").value_counts()
    return


@app.cell
def _(variants):
    variants.get_column("cohort").value_counts()
    return


@app.cell
def _(variants):
    variants.get_column("model_marginal").value_counts()
    return


@app.cell
def _(variants):
    variants.get_column("variant").is_duplicated().value_counts()
    return


@app.cell
def _(pl, variants):
    variants.filter(pl.col("variant").is_duplicated()).head()
    return


@app.cell
def _(pl, variants):
    # Filtering criteria:
    # - SNVs (i.e. ref and alt allele both length 1)
    # - Pass the two author included filters (not (LD_SV or LD_HWE))
    # - Drop rows referring to the same varian (e.g. flagged by both methods)
    pass_variants = (
        variants
        .filter(
            (pl.col("allele1").str.len_chars() == 1) & (pl.col("allele2").str.len_chars() == 1) &
            ~pl.col("LD_SV") &
            ~pl.col("LD_HWE")
        )
        .group_by(pl.col("variant"))
        .agg(
            pl.all().get(pl.col("pip").arg_max())
        )
    )
    pass_variants.shape
    return (pass_variants,)


@app.cell
def _(pass_variants):
    (pass_variants.get_column("pip") >= 0.9).value_counts()
    return


@app.cell
def _(pass_variants):
    pass_variants.get_column("rsid").str.starts_with("rs").value_counts()
    return


@app.cell
def _(pass_variants, pl):
    (
        pass_variants
        .filter(~pl.col("rsid").str.starts_with("rs"))
        .get_column("pip") >= 0.9
    ).value_counts()
    return


@app.cell
def _(data_dir, pass_variants, pl):
    # Just keep the ones with rsIDs for mapping to hg38
    # See recommended process here:
    #  https://genome.ucsc.edu/FAQ/FAQreleases.html#snpConversion
    # edit: bigBedToNamedItems seemed to just hang, so instead just
    # ended up querying the tableBrowser for each autosome separately
    # and concatenating
    (
        pass_variants
        .filter(pl.col("rsid").str.starts_with("rs"))
        .write_csv(
            data_dir / "variants" / "release1.1" / "UKBB_94traits_release1.filtered.bed.gz",
            separator="\t",
            include_header=False,
            compression="gzip",
        )
    )
    return


@app.cell
def _(data_dir, pl):
    hg38_rsids = pl.read_csv(
        data_dir / "variants" / "release1.1" / "hg38_rsids.tsv.gz",
        separator="\t",
        has_header=False,
        columns=[0, 1, 2, 3, 4],
        new_columns=[
            "chromosome",
            "start",
            "end",
            "rsid",
            "ref"
        ]
    )
    return (hg38_rsids,)


@app.cell
def _(hg38_rsids):
    hg38_rsids.head()
    return


@app.cell
def _(hg38_rsids, pass_variants, pl):
    len_before = len(pass_variants)
    mapped_variants = (
        pass_variants
        .drop(["chromosome", "start", "end"])
        .join(
            hg38_rsids,
            on="rsid",
            how="inner",
            #validate="1:1",
        )
        .with_columns(
            variant=pl.col("chromosome") + ":" + pl.col("end").cast(pl.String) + ":" + pl.col("allele1") + ":" + pl.col("allele2"),
            chr_num=pl.col("chromosome").str.extract(r"chr([\d]+)"),
        )
        .sort(by=["chr_num", "start"])
        .drop("chr_num")
    )
    len_after = len(mapped_variants)
    print(f"{len_before} -> {len_after}")
    return (mapped_variants,)


@app.cell
def _(mapped_variants):
    mapped_variants.head()
    return


@app.cell
def _(mapped_variants, pl):
    mapped_variants.with_columns(length=pl.col("end") - pl.col("start")).get_column("length").value_counts()
    return


@app.cell
def _(mapped_variants, pl):
    mapped_variants.with_columns(match=pl.col("ref") == pl.col("allele1")).get_column("match").value_counts()
    return


@app.cell
def _(mapped_variants, pl):
    mapped_variants_filtered = mapped_variants.filter(pl.col("ref") == pl.col("allele1"))
    return (mapped_variants_filtered,)


@app.cell
def _(mapped_variants_filtered):
    pip_vals = mapped_variants_filtered.get_column("pip")
    return (pip_vals,)


@app.cell
def _(pip_vals):
    pip_vals.describe()
    return


@app.cell
def _(np):
    bins = np.logspace(start=-4, stop=0, num=5)
    bins
    return (bins,)


@app.cell
def _(bins, pip_vals):
    pip_vals.cut(breaks=bins).value_counts()
    return


@app.cell
def _():
    high_thresh = 0.5
    low_thresh = 0.1
    return high_thresh, low_thresh


@app.cell
def _(high_thresh, mapped_variants_filtered, pl):
    # High PIP variants are only on autosomes, so we'll filter to those for simplicity
    (
        mapped_variants_filtered
        .filter(pl.col("pip") >= high_thresh)
        .get_column("chromosome")
        .value_counts()
    )
    return


@app.cell
def _(high_thresh, low_thresh, mapped_variants_filtered, pl):
    selected_variants = (
        mapped_variants_filtered
        .filter(
            (pl.col("pip") >= high_thresh) | (pl.col("pip") <= low_thresh) &
            (pl.col("chromosome").str.contains(r"chr[\d]+$"))
        )
        .with_columns(
            label=pl.when(pl.col("pip") >= high_thresh).then(pl.lit("high_pip")).otherwise(pl.lit("low_pip"))
        )
    )
    print(selected_variants.shape)
    selected_variants.head()
    return (selected_variants,)


@app.cell
def _(selected_variants):
    selected_variants.get_column("label").value_counts()
    return


@app.cell
def _(data_dir, selected_variants):
    selected_variants.write_csv(
        data_dir / "variants" / "release1.1" / "UKBB_94traits_release1.hg38.filtered.bed.gz",
        separator="\t",
        include_header=True,
        compression="gzip",
    )
    return


@app.cell
def _(pl, selected_variants):
    # Reformatting variants for inputting to Ensembl VEP
    vep_variants = (
        selected_variants
        .with_columns(
            chromosome=pl.col("chromosome").str.extract(r"chr([\d]+)"),
            # Seems like VEP expects 1-based half-open cintervals?
            start=pl.col("start") + 1,
            allele=pl.col("allele1") + "/" + pl.col("allele2"),
            strand=pl.lit("+"),
        )
        .select(["chromosome", "start", "end", "allele", "strand", "variant"])
    )
    vep_variants.head()
    return (vep_variants,)


@app.cell
def _(data_dir, vep_variants):
    vep_variants.write_csv(
        data_dir / "variants" / "release1.1" / "vep" / "UKBB_94traits_release1.hg38.filtered.txt.gz",
        separator="\t",
        include_header=False,
        compression="gzip",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Ensembl VEP was run with the above input as follows:
    ```bash
    #! /bin/bash
    set -eux


    wd="/storage/data/vep_comparisons/variants/release1.1/vep"
    vep \
            -i "${wd}/UKBB_94traits_release1.hg38.filtered.txt.gz" \
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
            -o "${wd}/variant_effect_output.txt.gz" \
            2>&1 | tee "${wd}/run.log"
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since Ensembl defines many different categories of transcript types (see [here](https://useast.ensembl.org/info/genome/genebuild/biotypes.html)), we simplify things by defining protein coding as the `protein_coding` biotype (i.e. excluding things like nonsense-mediated decay transcripts) and define non-coding as the `lncRNA` biotype (i.e. excluding the many other types of non-coding RNA such as tRNA, rRNA, miRNA, etc).

    Criteria for variants we want to retain:
    - Present in an exon of a `protein_coding` or `lncRNA` gene
    - Not overlapping both a protein-coding exon and a non-coding exon (for simplicity)

    For each variant we also need to select representative transcripts, we priortiize as follows:
    - MANE select
    - Ensembl canonical
    - TSL level (1 to 5, lower is higher quality)

    Note that this process does retain multiple genes for some variants.
    """)
    return


@app.cell
def _(data_dir, pl):
    vep_output = (
        pl.read_csv(
            data_dir / "variants" / "release1.1" / "vep" / "variant_effect_output.txt.gz",
            separator="\t",
            has_header=True,
            schema_overrides={
                "#CHROM": pl.String,
                "PHENO": pl.String,
            },
            comment_prefix="##",
            null_values="-",
        )
        .rename(mapping={"#Uploaded_variation": "variant"})
        .rename(mapping=lambda x: x.lower())
    )
    print(vep_output.shape)
    vep_output.head()
    return (vep_output,)


@app.cell
def _(vep_output):
    vep_output.head()
    return


@app.cell
def _(pl, vep_output):
    # NOTE: A ton of these variants are in introns (like 1.1M)
    remove_consequences = [
        "downstream_gene_variant",
        "upstream_gene_variant",
        "intron_variant",
        "intergenic_variant",
    ]

    keep_cols = [
        "variant",
        "location",
        "allele",
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
    ]

    vep_output_filtered = (
        vep_output
        .select(keep_cols)
        .with_columns(
            conseq_split=pl.col("consequence").str.split(","),
        )
        .filter(
            pl.col("conseq_split").list.set_intersection(remove_consequences).list.len() == 0,
        )
    )
    print(vep_output_filtered.shape)
    vep_output_filtered.head()
    return (vep_output_filtered,)


@app.cell
def _(vep_output_filtered):
    vep_output_filtered.get_column("variant").unique().len()
    return


@app.cell
def _(pl, vep_output_filtered):
    variants_by_exon_type = (
        vep_output_filtered
        .group_by("variant")
        .agg(
            all_biotypes=pl.implode("biotype").list.unique()
        )
        .with_columns(
            pc=pl.col("all_biotypes").list.contains("protein_coding"),
            nc=pl.col("all_biotypes").list.contains("lncRNA"),
        )
        .with_columns(
            pc_and_nc=pl.col("pc") & pl.col("nc"),
        )
    )

    (
        variants_by_exon_type
        .select("pc", "nc", "pc_and_nc")
        .describe()
    )
    return (variants_by_exon_type,)


@app.cell
def _(pl, variants_by_exon_type):
    filtered_variants_by_exon_type = (
        variants_by_exon_type
        .filter(
            ~pl.col("pc_and_nc"),
            pl.col("pc") | pl.col("nc")
        )
    )
    filtered_variants_by_exon_type.shape
    return (filtered_variants_by_exon_type,)


@app.cell
def _(filtered_variants_by_exon_type):
    filtered_variants_by_exon_type.get_column("variant").unique().len()
    return


@app.cell
def _(filtered_variants_by_exon_type, pl, vep_output_filtered):
    transcript_per_variant = (
        vep_output_filtered
        .filter(
            pl.col("variant").is_in(filtered_variants_by_exon_type.get_column("variant").to_list()),
            pl.col("biotype").is_in(["protein_coding", "lncRNA"]),
        )
        .group_by(["variant", "gene"])
        .agg(
            feature=pl.col("feature").sort_by(["mane", "canonical", "tsl"], descending=False, nulls_last=True).first()
        )
    )
    print(transcript_per_variant.shape)
    transcript_per_variant.head()
    return (transcript_per_variant,)


@app.cell
def _(selected_variants):
    selected_variants.head()
    return


@app.cell
def _(selected_variants, transcript_per_variant, vep_output_filtered):
    final_variants = (
        vep_output_filtered
        .join(transcript_per_variant, on=["variant", "gene", "feature"])
        .drop(["location", "allele", "conseq_split"])
        .join(
            selected_variants,
            on="variant",
        )
        .select([
            "chromosome", "start", "end",
            "allele1", "allele2", "variant", "rsid",
            "gene", "feature", "consequence",
            "cdna_position", "cds_position", "protein_position", "amino_acids",
            "symbol", "biotype",
            "maf", "pip", "label"
        ])
        .rename({"allele1": "ref", "allele2": "alt"})
    )
    print(final_variants.shape)
    final_variants.head()
    return (final_variants,)


@app.cell
def _(final_variants):
    final_variants.get_column("biotype").value_counts()
    return


@app.cell
def _(final_variants):
    final_variants.pivot(
        on="biotype",
        index="label",values="label",
        aggregate_function='len',
        sort_columns=True,
    )
    return


@app.cell
def _(final_variants):
    final_variants.unique("variant").pivot(
        on="biotype",
        index="label",values="label",
        aggregate_function='len',
        sort_columns=True,
    )
    return


@app.cell
def _(final_variants):
    final_variants.get_column("variant").value_counts(name="num_genes").get_column("num_genes").value_counts(sort=True)
    return


@app.cell
def _(data_dir, final_variants):
    final_variants.write_csv(
        data_dir / "variants" / "release1.1" / "final_UKBB_94traits_release1.hg38.selected_transcript.txt.gz",
        separator="\t",
        include_header=True,
        compression="gzip",
    )
    return


@app.cell
def _(final_variants, pl):
    downsampled_low_pip_variants = (
        final_variants
        .filter(
            pl.col("label") == "low_pip"
        )
        .group_by("biotype")
        .map_groups(lambda df: df.sample(fraction=0.1, with_replacement=False, seed=42))
    )
    downsampled_low_pip_variants.shape
    return (downsampled_low_pip_variants,)


@app.cell
def _(downsampled_low_pip_variants, final_variants, pl):
    downsampled_variants = (
        pl.concat((
            downsampled_low_pip_variants,
            final_variants.filter(pl.col("label") == "high_pip")
        ))
        .with_columns(
            chrom_num=pl.col("chromosome").str.extract(r"chr([\d]+)")
        )
        .sort(by=["chrom_num", "start"])
        .drop("chrom_num")
    )
    downsampled_variants.head()
    return (downsampled_variants,)


@app.cell
def _(downsampled_variants):
    downsampled_variants.pivot(
        on="biotype",
        index="label",values="label",
        aggregate_function='len',
        sort_columns=True,
    )
    return


@app.cell
def _(data_dir, downsampled_variants):
    downsampled_variants.write_csv(
        data_dir / "variants" / "release1.1" / "final_UKBB_94traits_release1.hg38.selected_transcript.downsampled.txt.gz",
        separator="\t",
        include_header=True,
        compression="gzip",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
