import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():

    from pathlib import Path

    import pandas as pd
    import polars as pl
    import seaborn as sns

    return Path, pd, pl, sns


@app.cell
def _(Path):
    data_dir = Path("/path/to/vep_comparisons/")

    dataset_dir = data_dir / "variants" / "clinvar"
    vcf_path = dataset_dir / "clinvar_20260208.vcf.gz"
    return dataset_dir, vcf_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prepare ClinVar variants
    """)


@app.cell
def _(pl, vcf_path):
    raw_clinvar = (
        pl.read_csv(
            vcf_path,
            separator="\t",
            has_header=True,
            schema_overrides={'#CHROM': pl.String},
            comment_prefix='##'
        )
        .with_row_index("idx")
    )

    # Split INFO field into separate columns
    wide_info = (
        raw_clinvar
        .select("idx", "INFO")
        .filter(
            pl.col("INFO").is_not_null(),
            (pl.col("INFO") != "."),
        )
        # "A=1;B=foo;FLAG" -> ["A=1", "B=foo", "FLAG"]
        .with_columns(
            pl.col("INFO").str.split(";").alias("_item")
        )
        .explode("_item")

        # "A=1" -> {"field_0": "A", "field_1": "1"}
        # "FLAG" -> {"field_0": "FLAG", "field_1": null}
        .with_columns(
            pl.col("_item")
            .str.split_exact("=", 1)
            .alias("_kv")
        )
        .unnest("_kv")
        .rename({
            "field_0": "_key",
            "field_1": "_value",
        })

        # Match your previous handling of flag-style INFO entries.
        # All INFO-derived columns will be strings in this version.
        .with_columns(
            pl.col("_value").fill_null("true")
        )

        # One column per INFO key.
        .pivot(
            index="idx",
            on="_key",
            values="_value",
            aggregate_function="last",
            maintain_order=True,
        )
    )

    wide_info.head()
    return raw_clinvar, wide_info


@app.cell
def _(raw_clinvar, wide_info):
    parsed_clinvar = (
        raw_clinvar
        .join(wide_info, on="idx", how="left")
        .drop("idx", "INFO")
    )
    parsed_clinvar.head()
    return (parsed_clinvar,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Convert clinical review status to stars
        - On the ClinVar webpage, the various clinical review statuses are presented as a confidence level ranging from 0-4 stars. Perform that conversion here.
        - Then filter to two stars or greater.
    - Select variants with "normal" significance status
        - Select for variants that are at least likely benign or likely pathogenic, and VUS (i.e. exclude the weird categories).
    """)


@app.cell
def _(parsed_clinvar):
    parsed_clinvar.get_column("CLNREVSTAT").value_counts(sort=True)


@app.cell
def _(parsed_clinvar, pl):
    status_to_stars = {
        'practice_guideline': 4,
        'reviewed_by_expert_panel': 3,
        'criteria_provided,_multiple_submitters,_no_conflicts': 2,
        'criteria_provided,_conflicting_classifications': 1,
        'criteria_provided,_single_submitter': 1,
        'no_classification_provided': 0,
        'no_classification_for_the_single_variant': 0,
        'no_assertion_criteria_provided': 0,
        'no_classifications_from_unflagged_records': 0,
    }
    want_types = ['Benign', 'Likely_benign', 'Pathogenic', 'Likely_pathogenic']
    autosomes = [f"chr{chromosome}" for chromosome in range(1, 23)]

    output_cols= [
        "variant", "chromosome", "start", "end", "ref", "alt", "pip", "label", "alleleid", "stars",
    ]


    filtered_clinvar = (
        parsed_clinvar
        .rename({
            "#CHROM": "chromosome",
            "POS": "end",
            "CLNSIG": "label",
        })
        .with_columns(
            stars=pl.col("CLNREVSTAT").replace_strict(status_to_stars, default=0, return_dtype=pl.UInt8),
            chromosome="chr"+pl.col("chromosome").cast(pl.String)
        )
        .filter(
            pl.col("chromosome").is_in(autosomes),
            pl.col("stars") >= 2,
            pl.col("label").is_in(want_types),
            pl.col("REF").str.len_chars() == 1,
            pl.col("ALT").str.len_chars() == 1,
        )
        # Make column format fit other datasets
        .rename(lambda x: x.lower())
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
            start=pl.col("end") - 1,
            pip=None,
        )
        .select(output_cols)
    )
    filtered_clinvar.head()
    return (filtered_clinvar,)


@app.cell
def _(filtered_clinvar):
    (
        filtered_clinvar
        .pivot(on="stars", index="label", values="stars", aggregate_function="len")
        .fill_null(0)
        .select(["label"]+ list(map(str, range(2, 5))))
    )


@app.cell
def _(dataset_dir, filtered_clinvar):
    (
        filtered_clinvar
        .write_csv(
            dataset_dir / "filtered_clinvar.tsv.gz",
            separator="\t",
            include_header=True,
            compression="gzip",
        )
    )


@app.cell
def _(filtered_clinvar, pl):
    # VEP custom input is 1-based; selected_variants remains BED-style.
    vep_variants = (
        filtered_clinvar
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
            dataset_dir / "vep" / "filtered_clinvar.for_vep.tsv.gz",
            separator="\t",
            include_header=False,
            compression="gzip",
        )
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parse VEP results
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given the variants of interest above, their consequence on transcripts was predicted using Ensembl's VEP tool. The tool was run as follows:
    ```bash
    vep \
            -i /path/to/clinvar/20260208/clinvar_20260208.filtered.vcf.gz \
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
            --force_overwrite \
            --dir /path/to/vep/ \
            --fork 8 \
            --verbose \
            -o /path/to/rna_localization/variants/clinvar/variant_effect_output.txt.gz \
            2>&1 | tee run.log
    ```
    annotations were sourced using the version 115 of the GRCh38 cache (i.e. GENCODE 49).
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial parsing
    Parsing out the "Extra" string which contains key-value pairs of additional outputs, filtering to only hits to canonical transcripts, and removing `upstream_gene_variant` and `downstream_gene_variant` hits.
    """)


@app.cell
def _(parsed_vep_path):
    if parsed_vep_path.exists():
        print(
            "Initial parsing file already exists, are you sure you want to rerun? "
            "Parsing from original file requires ~90GB of memory due to large size of the file (~12M rows)."
        )


@app.cell
def _(pd, pl, vep_path):
    def _parse_info_string(info_str):
        if pd.isna(info_str) or info_str == '.':
            return {}
        info_dict = {}
        for part in info_str.split(';'):
            if '=' in part:
                key, value = part.split('=', 1)
                info_dict[key] = value
        return info_dict
    df_4 = pl.read_csv(vep_path, separator='\t', has_header=True, schema_overrides={'#CHROM': str}, comment_prefix='##')
    _parsed_info = []
    for _row in df_4.iter_rows(named=True):
        _parsed_info.append(_parse_info_string(_row['Extra']))
    _info_df = pl.DataFrame(_parsed_info)
    df_4 = df_4.hstack(_info_df)
    print(df_4.shape)
    df_4.head()
    return (df_4,)


@app.cell
def _(df_4):
    len(df_4['Gene'].unique())


@app.cell
def _(df_4, pl):
    len(df_4.filter(pl.col('CANONICAL') == 'YES')['Feature'].unique())


@app.cell
def _(df_4, pl):
    df_4.filter(pl.col('BIOTYPE') == 'lncRNA')


@app.cell
def _(df_4, pl):
    df_4.filter(pl.col('BIOTYPE') == 'lncRNA').select(pl.col('cDNA_position').value_counts(sort=True))


@app.cell
def _(df_4, pl):
    df_4.select(pl.col('BIOTYPE').value_counts(sort=True)).unnest('BIOTYPE')


@app.cell
def _(df_4, pl):
    df_4.filter(pl.col('CANONICAL') == 'YES').select(pl.col('BIOTYPE').value_counts(sort=True)).unnest('BIOTYPE')


@app.cell
def _(df_4):
    df_4['#Uploaded_variation'].unique()


@app.cell
def _(df_4, pl):
    df_4.select(pl.col('CANONICAL').value_counts(sort=True)).unnest('CANONICAL')


@app.cell
def _(df_4, pl):
    canonical = df_4.filter(pl.col('CANONICAL') == 'YES')
    canonical.shape
    return (canonical,)


@app.cell
def _(canonical, pl):
    canonical.select(pl.col("Consequence").value_counts(sort=True)).unnest("Consequence")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For now, we'll remove `downstream_gene_variant` and `upstream_gene_variant`, since these are just defined as variants that are within 5kb of the 5' or 3' end of a gene, and it seems like many variants are annotated with these in additon to transcripts that they actually fall into. In the future, we can try to be a bit more stringent and only remove these entries for variants that do actually fall into another transcript (i.e. keep rows where the upstream/downstream is the only prediction for that variant).
    """)


@app.cell
def _(canonical, pl):
    canonical_1 = canonical.filter(~pl.col('Consequence').is_in(['downstream_gene_variant', 'upstream_gene_variant']))
    canonical_1.shape
    return (canonical_1,)


@app.cell
def _(canonical_1):
    canonical_1.head()


@app.cell
def _(canonical_1, pl):
    canonical_1.filter(pl.col('BIOTYPE') == 'lncRNA').select(pl.col('cDNA_position').value_counts(sort=True))


@app.cell
def _(canonical_1, pl):
    canonical_1.filter(pl.col('BIOTYPE') == 'lncRNA')


@app.cell
def _(canonical_1, pl):
    _want_cols = ['#Uploaded_variation', 'Gene', 'Feature', 'Consequence', 'SYMBOL', 'BIOTYPE', 'MAX_AF', 'MAX_AF_POPS', 'MANE_SELECT']
    initial_vars = canonical_1.select(pl.col(_want_cols)).rename(lambda c: c.strip('#').lower())
    initial_vars.head()
    return (initial_vars,)


@app.cell
def _(initial_vars, parsed_vep_path):
    initial_vars.write_csv(parsed_vep_path, separator="\t", compression="gzip")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore selected variants
    """)


@app.cell
def _(parsed_vep_path, pl):
    initial_vars_1 = pl.read_csv(parsed_vep_path, separator='\t').rename({'uploaded_variation': 'id'})
    print(initial_vars_1.shape)
    initial_vars_1.head()
    return (initial_vars_1,)


@app.cell
def _(filt_tsv_path, pl):
    clinvar_annot = pl.read_csv(
        filt_tsv_path,
        schema_overrides={"chrom": str},
        separator="\t",
    )
    print(clinvar_annot.shape)
    clinvar_annot.head()
    return (clinvar_annot,)


@app.cell
def _(clinvar_annot):
    assert len(clinvar_annot) == len(clinvar_annot["id"].unique())


@app.cell
def _(clinvar_annot, initial_vars_1):
    merged = initial_vars_1.join(clinvar_annot, on='id', how='inner')
    assert len(merged) == len(initial_vars_1)
    return (merged,)


@app.cell
def _(merged, pl):
    len(merged.filter(pl.col("clnsig") != "Uncertain_significance"))


@app.cell
def _(merged):
    merged.head()


@app.cell
def _(merged):
    (
        merged
        .pivot(on="clnsig", index="consequence", values="clnsig", aggregate_function="len")
        .fill_null(0)
        .head(10)
    )


@app.cell
def _(merged, pl):
    merged.select(pl.col("consequence").str.split(",").explode().value_counts(sort=True)).unnest("consequence")


@app.cell
def _(merged, pl):
    merged.select(pl.col("clnsig").value_counts(sort=True)).unnest("clnsig")


@app.cell
def _(pd, pl, sns):
    coding_effect = [
        "missense_variant",
        "frameshift_variant",
        "stop_gained",
        "splice_donor_variant",
        "inframe_deletion",
        "splice_acceptor_variant",
        "start_lost",
        "stop_lost",
        "coding_sequence_variant",
        "protein_altering_variant",
        "transcript_ablation",
    ]

    name_map = {
        "Likely_pathogenic": "pathogenic",
        "Pathogenic": "pathogenic",
        "Likely_benign": "benign",
        "Benign": "benign",
    }

    def make_consequence_plot(
        df: pd.DataFrame,
        silent_only: bool = True,
    ):
        long = (
            df
            .select(pl.col(["id", "gene", "feature", "biotype", "clnsig", "consequence"]))
            .with_columns(
                consequence=pl.col("consequence").str.split(",")
            )
            .explode(columns="consequence")
        )

        plot_df = (
            long
            .filter(pl.col("clnsig") != "Uncertain_significance")
            .with_columns(
                clnsig=pl.col("clnsig").replace_strict(name_map, return_dtype=str)
            )
            .pivot(on="clnsig", index="consequence", values="clnsig", aggregate_function="len")
            .fill_null(0)
            .unpivot(index="consequence", variable_name="clnsig", value_name="count")
        )
        if silent_only:
            plot_df = plot_df.filter(~pl.col("consequence").is_in(coding_effect))

        order = plot_df.select(pl.col("consequence").value_counts(sort=True)).unnest("consequence")["consequence"].to_list()

        ax = sns.barplot(
            x="count",
            y="consequence",
            hue="clnsig",
            order=order,
            orient="h",
            data=plot_df,
        )
        ax.set_xscale("log")
        _ = ax.set_yticklabels([t.get_text().replace('_', ' ') for t in ax.get_yticklabels()])
        ax.legend(title="Clinical significance")
        ax.set_title("Predicted effects of ClinVar variants (2+ stars) on canonical GRCh38 transcripts")

    return coding_effect, make_consequence_plot


@app.cell
def _(make_consequence_plot, merged):
    make_consequence_plot(merged, silent_only=False)


@app.cell
def _(make_consequence_plot, merged):
    make_consequence_plot(merged)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To be very conservative, we'll remove any variant whose consequence could be one of the "coding" type that we've defined above.
    """)


@app.cell
def _(conseq_long, pl):
    num_consequences = conseq_long.select(pl.col("id").value_counts(sort=True)).unnest("id")
    num_consequences
    return (num_consequences,)


@app.cell
def _(coding_effect, conseq_long, pl):
    num_silent_consequences = (
        conseq_long
        .filter(~pl.col("consequence").is_in(coding_effect))
        .select(pl.col("id").value_counts(sort=True)).unnest("id")
    )
    num_silent_consequences
    return (num_silent_consequences,)


@app.cell
def _(num_consequences, num_silent_consequences, pl):
    want_1 = num_consequences.join(num_silent_consequences, on='id', suffix='_silent', validate='1:1').with_columns(want=pl.col('count') == pl.col('count_silent')).filter(pl.col('want'))
    want_1
    return (want_1,)


@app.cell
def _(merged, pl, want_1):
    silent_vars = merged.filter(pl.col('id').is_in(want_1['id'].to_list()))
    print(len(silent_vars))
    silent_vars.head()
    return (silent_vars,)


@app.cell
def _(silent_vars):
    len(silent_vars["id"].unique())


@app.cell
def _(pl, silent_vars):
    silent_vars.filter(pl.col("id").is_duplicated())


@app.cell
def _(pl, silent_vars):
    silent_vars.select(pl.col("clnsig").value_counts(sort=True)).unnest("clnsig")


@app.cell
def _(make_consequence_plot, silent_vars):
    make_consequence_plot(silent_vars)


@app.cell
def _(pl, silent_vars):
    (
        silent_vars
        .filter(pl.col("clnsig").str.contains("athogenic"))
        .select(pl.col("consequence").str.split(",").explode().value_counts(sort=True)).unnest("consequence")
        .head(n=10)
    )


@app.cell
def _(pl, silent_vars):
    (
        silent_vars
        .filter(pl.col("clnsig").str.contains("athogenic") & pl.col("consequence").str.contains("synonymous"))
        .head(n=10)
    )


@app.cell
def _(silent_vars, silent_vars_path):
    silent_vars.write_csv(silent_vars_path, separator="\t", compression="gzip")


@app.cell
def _(pd):
    parsed_variants = pd.read_parquet("/path/to/rna_localization/variants/clinvar/clinvar_variants_dataset.parquet")
    parsed_variants.head()
    return (parsed_variants,)


@app.cell
def _(parsed_variants):
    parsed_variants.clnsig.value_counts()


@app.cell
def _(parsed_variants):
    len(parsed_variants)


@app.cell
def _(pd, silent_vars_path):
    silent_vars_1 = pd.read_table(silent_vars_path)
    return (silent_vars_1,)


@app.cell
def _(silent_vars_1):
    silent_vars_1.clnsig.value_counts()


@app.cell
def _(silent_vars_1):
    ben_vars = silent_vars_1.loc[lambda x: x.clnsig.str.contains('enign')]
    ben_vars
    return (ben_vars,)


@app.cell
def _(ben_vars):
    ben_vars.consequence.value_counts()


@app.cell
def _():
    benign_downsample_categories = [
        "synonymous_variant",
        "intron_variant,non_coding_transcript_variant",
        "splice_polypyrimidine_tract_variant,intron_variant ",
        "splice_region_variant,splice_polypyrimidine_tract_variant,intron_variant ",
    ]
    return (benign_downsample_categories,)


@app.cell
def _(ben_vars, benign_downsample_categories, pd):
    ben_vars_downsampled = pd.concat((
        ben_vars.loc[lambda x: ~x.consequence.isin(benign_downsample_categories)],
        ben_vars.loc[lambda x: x.consequence.isin(benign_downsample_categories)].groupby("consequence").sample(n=2000, random_state=42),
    ))
    ben_vars_downsampled.shape
    return (ben_vars_downsampled,)


@app.cell
def _(ben_vars_downsampled):
    ben_vars_downsampled.consequence.value_counts()


@app.cell
def _(silent_vars_1):
    path_vars = silent_vars_1.loc[lambda x: x.clnsig.str.contains('athogenic')]
    path_vars
    return (path_vars,)


@app.cell
def _(ben_vars_downsampled, path_vars, pd):
    downsampled_vars = pd.concat((path_vars, ben_vars_downsampled)).sort_values(["chrom", "pos"])
    len(downsampled_vars)
    return (downsampled_vars,)


@app.cell
def _(clinvar_variants_dir, downsampled_vars):
    downsampled_vars.to_csv(clinvar_variants_dir / "silent_variants_effects.downsampled.txt.gz", sep="\t", index=False)


@app.cell
def _(path_vars):
    path_vars.consequence.value_counts()


@app.cell
def _(path_vars):
    path_vars.consequence.value_counts()


@app.cell
def _(path_vars):
    path_vars.query("consequence == 'non_coding_transcript_exon_variant'")


@app.cell
def _(path_vars):
    path_vars.query("consequence == 'non_coding_transcript_exon_variant'").biotype.value_counts()


@app.cell
def _(path_vars):
    path_vars.query("consequence == 'non_coding_transcript_exon_variant' and biotype == 'lncRNA'")


if __name__ == "__main__":
    app.run()
