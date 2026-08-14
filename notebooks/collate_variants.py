import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import seaborn as sns

    return Path, cs, mo, pl


@app.cell
def _(Path):
    data_dir = Path("/orcd/data/manoli/001/rcalef/data/vep_comparisons/variants")

    data_paths = {
        "ukbb": data_dir
        / "ukbb_finucane"
        / "final_UKBB_94traits_release1.hg38.selected_transcript.txt.gz",
        "multisusie": data_dir / "multisusie" / "final_variants.tsv.gz",
        "eqtl": data_dir / "cis_eqtl" / "final_variants.tsv.gz",
        "clinvar": data_dir / "clinvar" / "final_variants.tsv.gz",
    }
    positive_labels = {
        "ukbb": ["high_pip"],
        "multisusie": ["high_pip"],
        "eqtl": ["high_pip"],
        "clinvar": ["Pathogenic", "Likely_pathogenic"],
    }
    return data_dir, data_paths, positive_labels


@app.cell
def _(Path, pl):
    id_cols = [
        "variant",
        "gene",
    ]

    shared_cols_str = [
        "chromosome",
        "ref",
        "alt",
        "feature",
        "consequence",
        "amino_acids",
        "symbol",
        "biotype",
    ]

    shared_cols_int = [
        "start",
        "end",
        "cdna_position",
        "cds_position",
        "protein_position",
    ]
    shared_cols = shared_cols_str + shared_cols_int

    separate_cols = [
        "pip",
        "label",
    ]


    def read_filtered_vep(path: Path) -> pl.DataFrame:
        return pl.read_csv(
            path,
            separator="\t",
            null_values="-",
            schema_overrides={
                **{name: pl.String for name in shared_cols_str},
                **{name: pl.UInt32 for name in shared_cols_int},
            },
        )


    return id_cols, read_filtered_vep, separate_cols, shared_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Want to perform the collation as follows:
    - For the variants shared between datasets:
        - Are the sets of overlapped genes the same?
        - For each gene, do all of the other annotations match (i.e. transcript, position, etc)?
        - Do the other data match (e.g. chromomosome, pos, ref, alt)?
    - For variants not shared between datasets, just want to concat
    """)
    return


@app.cell
def _(
    cs,
    data_paths,
    id_cols,
    pl,
    read_filtered_vep,
    separate_cols,
    shared_cols,
):
    # These datasets contain every retained VEP gene for each variant. The eQTL
    # data are intentionally restricted to the gene tested by the association.
    complete_gene_coverage = {"ukbb", "multisusie", "clinvar"}
    autosomes = [f"chr{chromosome}" for chromosome in range(1, 23)]

    datasets = {}
    frames_to_collate = []
    for name, path in data_paths.items():
        variants = read_filtered_vep(path)
        if name == "eqtl" and "variant_right" in variants.columns:
            mismatched_join_keys = variants.filter(
                ~pl.col("variant").eq_missing(pl.col("variant_right"))
            )
            if not mismatched_join_keys.is_empty():
                raise ValueError(
                    "eQTL variant_right contains values that differ from variant: "
                    f"{mismatched_join_keys.head()}"
                )
            variants = variants.drop("variant_right")
        if name == "clinvar":
            excluded = variants.filter(~pl.col("chromosome").is_in(autosomes))
            variants = variants.filter(pl.col("chromosome").is_in(autosomes))
            print(
                "Excluded "
                f"{excluded.get_column('variant').n_unique():,} non-autosomal "
                f"ClinVar variants / {excluded.height:,} gene-level rows"
            )
        if variants.select(pl.struct(id_cols).is_duplicated().any()).item():
            raise ValueError(f"{name!r} has duplicate (variant, gene) rows")

        datasets[name] = variants
        extra_cols = variants.select(
            cs.exclude(id_cols + shared_cols + separate_cols)
        ).columns
        rename_cols = {col: f"{name}_{col}" for col in separate_cols + extra_cols}
        frames_to_collate.append(variants.rename(rename_cols))

    validation_rows = []
    dataset_names = list(datasets)
    for i, left_name in enumerate(dataset_names):
        for right_name in dataset_names[i + 1 :]:
            left = datasets[left_name]
            right = datasets[right_name]

            shared_variants = (
                left
                .select("variant")
                .unique()
                .join(
                    right.select("variant").unique(),
                    on="variant",
                    how="inner",
                )
            )
            left = left.join(shared_variants, on="variant", how="semi")
            right = right.join(shared_variants, on="variant", how="semi")

            left_only = (
                left
                .select(id_cols)
                .join(
                    right.select(id_cols),
                    on=id_cols,
                    how="anti",
                )
            )
            right_only = (
                right
                .select(id_cols)
                .join(
                    left.select(id_cols),
                    on=id_cols,
                    how="anti",
                )
            )

            left_complete = left_name in complete_gene_coverage
            right_complete = right_name in complete_gene_coverage
            if left_complete and right_complete:
                invalid_coverage = pl.concat([left_only, right_only])
            elif left_complete:
                invalid_coverage = right_only
            elif right_complete:
                invalid_coverage = left_only
            else:
                invalid_coverage = pl.DataFrame(schema=left_only.schema)

            if not invalid_coverage.is_empty():
                raise ValueError(
                    f"Invalid gene coverage between {left_name!r} and "
                    f"{right_name!r}:\n{invalid_coverage.head()}"
                )

            common = left.join(
                right, on=id_cols, how="inner", suffix="_right", validate="1:1"
            )
            annotation_mismatches = common.filter(
                pl.any_horizontal(
                    ~pl.col(col).eq_missing(pl.col(f"{col}_right"))
                    for col in shared_cols
                )
            )
            if not annotation_mismatches.is_empty():
                raise ValueError(
                    f"VEP annotations differ between {left_name!r} and "
                    f"{right_name!r}:\n{annotation_mismatches.head()}"
                )

            validation_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "shared_variants": len(shared_variants),
                    "left_only_genes": len(left_only),
                    "right_only_genes": len(right_only),
                }
            )

    # Stack first to retain the union of (variant, gene) rows, then coalesce
    # columns. Validation above guarantees that shared VEP values agree.
    curr_variants = (
        pl.concat(frames_to_collate, how="diagonal_relaxed")
        .group_by(id_cols, maintain_order=True)
        .agg(pl.exclude(id_cols).drop_nulls().first())
        .select(
            *id_cols,
            *shared_cols,
            *[cs.starts_with(name) for name in data_paths],
        )
    )
    invalid_chromosomes = curr_variants.filter(
        ~pl.col("chromosome").is_in(autosomes)
    )
    if not invalid_chromosomes.is_empty():
        raise ValueError(
            "Collated output contains non-autosomal variants: "
            f"{invalid_chromosomes.get_column('chromosome').unique().to_list()}"
        )
    validation = pl.DataFrame(validation_rows)
    return curr_variants, validation


@app.cell
def _(validation):
    validation
    return


@app.cell
def _(cs, curr_variants, data_paths, pl):
    overlap = (
        curr_variants
        .with_columns(
            **{
                f"{name}_present": pl.col(f"{name}_label").is_not_null()
                for name in data_paths
            }
        )
        .group_by(cs.ends_with("_present"))
        .agg(pl.len().alias("n_variants"))
        .sort("n_variants", descending=True)
    )
    overlap
    return


@app.cell
def _(curr_variants):
    curr_variants.head()
    return


@app.cell
def _(curr_variants, data_dir):
    collated_path = data_dir / "collated_variants.tsv.gz"
    curr_variants.write_csv(
        collated_path,
        separator="\t",
        compression="gzip",
        null_value="-",
    )
    collated_path
    return (collated_path,)


@app.cell
def _(cs, curr_variants, data_paths, pl, positive_labels):
    overlap_pos_only = (
        curr_variants
        .with_columns(
            **{
                f"{name}_present": pl.col(f"{name}_label").is_in(
                    positive_labels[name]
                ).fill_null(False)
                for name in data_paths
            }
        )
        .group_by(cs.ends_with("_present"))
        .agg(pl.len().alias("n_variants"))
        .sort("n_variants", descending=True)
    )
    overlap_pos_only
    return


@app.cell
def _(curr_variants, pl):
    (
        curr_variants
        .filter(
            pl.col("ukbb_label") == "high_pip",
        )
        .get_column("eqtl_label")
        .value_counts()
    )
    return
if __name__ == "__main__":
    app.run()
