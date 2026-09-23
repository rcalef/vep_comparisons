import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import os
    import re
    import time
    from itertools import groupby
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import pyBigWig
    import pysam
    from tqdm import tqdm

    return Path, groupby, math, mo, os, pl, pyBigWig, pysam, re, time, tqdm


@app.cell
def _(Path, os):
    data_dir = Path(
        os.environ.get(
            "VEP_COMPARISONS_DATA_DIR",
            "/orcd/data/manoli/001/rcalef/data/vep_comparisons",
        )
    )
    variants_dir = data_dir / "variants"
    collated_path = Path(
        os.environ.get(
            "VEP_COMPARISONS_COLLATED_PATH",
            variants_dir / "collated_variants.tsv.gz",
        )
    )
    annotated_path = Path(
        os.environ.get(
            "VEP_COMPARISONS_ANNOTATED_PATH",
            variants_dir / "collated_variants.annotated.tsv.gz",
        )
    )

    gpn_dir = Path(
        os.environ.get(
            "VEP_COMPARISONS_GPN_STAR_LLR_DIR",
            data_dir.parent / "gpn_star" / "gpn-star-scores" / "data",
        )
    )
    gpn_score_names = {
        "gpn_star_vertebrate": "gpn_star_v_score",
        "gpn_star_mammal": "gpn_star_m_score",
        "gpn_star_primate": "gpn_star_p_score",
    }

    annotation_paths = {
        "phylop": data_dir / "conservation" / "phylop" / "hg38.phyloP447way.bw",
        "gnocchi": data_dir
        / "constraint"
        / "gnocchi_constraint_z_genome_1kb.qc.download.txt.gz",
        "roulette": data_dir / "constraint" / "roulette",
        "gpn_star_vertebrate": gpn_dir / "gpn-star-hg38-v100-200m" / "llr",
        "gpn_star_mammal": gpn_dir / "gpn-star-hg38-m447-200m" / "llr",
        "gpn_star_primate": gpn_dir / "gpn-star-hg38-p243-200m" / "llr",
        "alphagenome_atlas": Path(
            os.environ.get(
                "VEP_COMPARISONS_ALPHAGENOME_ATLAS_PATH",
                data_dir.parent
                / "alphagenome_atlas"
                / "alphagenome_variant_impact_score_snvs.tsv.gz",
            )
        ),
    }
    return annotated_path, annotation_paths, collated_path, gpn_score_names


@app.cell
def annotation_functions(Path, groupby, math, pl, pyBigWig, pysam, re, time, tqdm):
    REQUIRED_COLUMNS = ("variant", "chromosome", "start", "end", "ref", "alt")
    ANNOTATION_COLUMNS = (
        "phylop",
        "gnocchi_z",
        "roulette_mr",
        "roulette_ar",
        "roulette_filter",
        "gpn_star_v_score",
        "gpn_star_m_score",
        "gpn_star_p_score",
        "alphagenome_atlas_raw_score",
    )
    AUTOSOME_RE = re.compile(r"chr(?:[1-9]|1[0-9]|2[0-2])\Z")
    BASE_RE = re.compile(r"[ACGT]\Z")
    ALT_RE = re.compile(r"(?:[ACGT]|\.)\Z")


    def validate_variants(frame: pl.DataFrame) -> pl.DataFrame:
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        null_counts = frame.select(
            pl.col(column).null_count().alias(column) for column in REQUIRED_COLUMNS
        ).row(0, named=True)
        columns_with_nulls = {
            column: count for column, count in null_counts.items() if count
        }
        if columns_with_nulls:
            raise ValueError(f"Required columns contain nulls: {columns_with_nulls}")

        invalid_rows = []
        for row in frame.select(REQUIRED_COLUMNS).iter_rows(named=True):
            chromosome = row["chromosome"]
            start = row["start"]
            end = row["end"]
            ref = row["ref"]
            alt = row["alt"]
            expected_variant = f"{chromosome}:{end}:{ref}:{alt}"
            if (
                not isinstance(chromosome, str)
                or AUTOSOME_RE.fullmatch(chromosome) is None
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end != start + 1
                or not isinstance(ref, str)
                or BASE_RE.fullmatch(ref) is None
                or not isinstance(alt, str)
                or ALT_RE.fullmatch(alt) is None
                or (alt != "." and ref == alt)
                or row["variant"] != expected_variant
            ):
                invalid_rows.append(row)
                if len(invalid_rows) == 5:
                    break
        if invalid_rows:
            raise ValueError(
                "Variants must be GRCh38 autosomal single-position records with "
                "zero-based half-open coordinates, A/C/G/T reference alleles, "
                "A/C/G/T or '.' alternate alleles, and identifiers formatted as "
                "chrN:POS:REF:ALT. "
                f"Examples of invalid rows: {invalid_rows}"
            )

        coordinate_counts = (
            frame.select(REQUIRED_COLUMNS)
            .unique()
            .group_by("variant")
            .len(name="coordinate_count")
            .filter(pl.col("coordinate_count") != 1)
        )
        if not coordinate_counts.is_empty():
            examples = coordinate_counts.head(5).to_dicts()
            raise ValueError(
                f"Conflicting coordinates or alleles for the same variant: {examples}"
            )

        return frame.select(REQUIRED_COLUMNS).unique(
            subset=["variant"], maintain_order=True
        )


    def annotate_phylop(variants: pl.DataFrame, path: Path) -> tuple[pl.DataFrame, float]:
        started = time.perf_counter()
        rows = []
        with pyBigWig.open(str(path)) as bigwig:
            for row in variants.select("variant", "chromosome", "start", "end").iter_rows(
                named=True
            ):
                values = bigwig.values(row["chromosome"], row["start"], row["end"])
                value = values[0] if values else None
                if value is not None and not math.isfinite(value):
                    value = None
                rows.append({"variant": row["variant"], "phylop": value})
        result = pl.DataFrame(rows, schema={"variant": pl.String, "phylop": pl.Float64})
        return result, time.perf_counter() - started


    def annotate_gnocchi(variants: pl.DataFrame, path: Path) -> tuple[pl.DataFrame, float]:
        started = time.perf_counter()
        intervals = (
            pl.read_csv(
                path,
                separator="\t",
                schema_overrides={
                    "chrom": pl.String,
                    "start": pl.UInt32,
                    "end": pl.UInt32,
                    "z": pl.Float64,
                },
            )
            .rename(
                {
                    "chrom": "chromosome",
                    "start": "gnocchi_start",
                    "end": "gnocchi_end",
                    "z": "gnocchi_z",
                }
            )
            .sort("chromosome", "gnocchi_start")
        )
        result = (
            variants.select("variant", "chromosome", "start")
            .sort("chromosome", "start")
            .join_asof(
                intervals,
                left_on="start",
                right_on="gnocchi_start",
                by="chromosome",
                strategy="backward",
                check_sortedness=False,
            )
            .with_columns(
                pl.when(pl.col("start") < pl.col("gnocchi_end"))
                .then(pl.col("gnocchi_z"))
                .otherwise(None)
                .alias("gnocchi_z")
            )
            .select("variant", "gnocchi_z")
        )
        return result, time.perf_counter() - started


    def annotate_roulette(
        variants: pl.DataFrame,
        roulette_dir: Path,
    ) -> tuple[pl.DataFrame, float, pl.DataFrame]:
        started = time.perf_counter()
        results = []
        timing_rows = []
        for (chromosome,), chromosome_variants in tqdm(
            variants.partition_by("chromosome", as_dict=True, maintain_order=True).items()
        ):
            chromosome_started = time.perf_counter()
            chromosome_id = chromosome.removeprefix("chr")
            positions = chromosome_variants.get_column("end").unique()
            path = roulette_dir / f"{chromosome_id}_rate_v5.2_TFBS_correction_all.vcf.bgz"
            scores = (
                pl.scan_csv(
                    path,
                    comment_prefix="##",
                    separator="\t",
                    schema_overrides={"POS": pl.UInt32},
                )
                .select("POS", "REF", "ALT", "FILTER", "INFO")
                .filter(pl.col("POS").is_in(positions.implode()))
                .select(
                    pl.col("POS").alias("end"),
                    pl.col("REF").alias("ref"),
                    pl.col("ALT").alias("alt"),
                    pl.col("INFO")
                    .str.extract(r"(?:^|;)MR=([^;]+)", 1)
                    .cast(pl.Float64, strict=False)
                    .alias("roulette_mr"),
                    pl.col("INFO")
                    .str.extract(r"(?:^|;)AR=([^;]+)", 1)
                    .cast(pl.Float64, strict=False)
                    .alias("roulette_ar"),
                    pl.col("FILTER").alias("roulette_filter"),
                )
                .with_columns(
                    pl.when(pl.col(column).is_finite())
                    .then(pl.col(column))
                    .otherwise(None)
                    .alias(column)
                    for column in ("roulette_mr", "roulette_ar")
                )
                .collect(engine="streaming")
            )
            result = chromosome_variants.join(
                scores,
                on=["end", "ref", "alt"],
                how="left",
                validate="m:1",
            ).select(
                "variant",
                "roulette_mr",
                "roulette_ar",
                "roulette_filter",
            )
            results.append(result)
            timing_rows.append(
                {
                    "source": "roulette",
                    "chromosome": chromosome,
                    "elapsed_seconds": time.perf_counter() - chromosome_started,
                }
            )
            del scores

        chromosome_timings = pl.DataFrame(timing_rows).sort(
            pl.col("chromosome").str.strip_prefix("chr").cast(pl.UInt8)
        )
        return pl.concat(results), time.perf_counter() - started, chromosome_timings


    def annotate_gpn_star(
        variants: pl.DataFrame,
        llr_dir: Path,
        score_name: str,
    ) -> tuple[pl.DataFrame, float, pl.DataFrame]:
        started = time.perf_counter()
        results = []
        timing_rows = []
        for (chromosome,), chromosome_variants in tqdm(
            variants.partition_by("chromosome", as_dict=True, maintain_order=True).items()
        ):
            chromosome_started = time.perf_counter()
            chromosome_id = chromosome.removeprefix("chr")
            positions = chromosome_variants.get_column("end").unique()
            scores = (
                pl.scan_parquet(llr_dir / f"llr_chr{chromosome_id}.parquet")
                .select("chrom", "pos", "ref", "alt", "llr_calibrated")
                .filter(pl.col("pos").is_in(positions.implode()))
                .rename({"llr_calibrated": score_name})
                .collect(engine="streaming")
            )
            result = (
                chromosome_variants.with_columns(
                    chrom=pl.col("chromosome").str.strip_prefix("chr"),
                    pos=pl.col("end").cast(pl.Int64),
                )
                .join(
                    scores,
                    on=["chrom", "pos", "ref", "alt"],
                    how="left",
                    validate="m:1",
                )
                .select("variant", score_name)
            )
            results.append(result)
            timing_rows.append(
                {
                    "source": "gpn_star",
                    "chromosome": chromosome,
                    "elapsed_seconds": time.perf_counter() - chromosome_started,
                }
            )

        chromosome_timings = pl.DataFrame(timing_rows).sort(
            pl.col("chromosome").str.strip_prefix("chr").cast(pl.UInt8)
        )
        return pl.concat(results), time.perf_counter() - started, chromosome_timings


    def annotate_alphagenome_atlas(
        variants: pl.DataFrame, path: Path
    ) -> tuple[pl.DataFrame, float, pl.DataFrame]:
        """Copy Atlas Variant Impact raw scores unchanged; omit PHRED."""
        started = time.perf_counter()
        rows = []
        timing_rows = []
        ordered = (
            variants
            .select(REQUIRED_COLUMNS)
            .unique(subset="variant")
            .sort(
                pl.col("chromosome").str.strip_prefix("chr").cast(pl.UInt8),
                "start",
            )
        )
        with pysam.TabixFile(str(path)) as tabix:
            contigs = set(tabix.contigs)
            for (chromosome,), chromosome_variants in tqdm(
                ordered.partition_by("chromosome", as_dict=True, maintain_order=True).items()
            ):
                chromosome_started = time.perf_counter()
                for (_, start, end), position_variants in groupby(
                    chromosome_variants.iter_rows(), key=lambda row: row[1:4]
                ):
                    position_variants = list(position_variants)
                    scores = {}
                    if chromosome in contigs and any(
                        row[5] != "." for row in position_variants
                    ):
                        for record in tabix.fetch(chromosome, start, end):
                            chrom, pos, ref, alt, raw_score, _phred = record.split("\t")
                            if chrom == chromosome and int(pos) == end:
                                scores[ref, alt] = float(raw_score)
                    for variant, _, _, _, ref, alt in position_variants:
                        rows.append(
                            (variant, scores.get((ref, alt)) if alt != "." else None)
                        )
                timing_rows.append(
                    {
                        "source": "alphagenome_atlas",
                        "chromosome": chromosome,
                        "elapsed_seconds": time.perf_counter() - chromosome_started,
                    }
                )
        result = pl.DataFrame(
            rows,
            schema={"variant": pl.String, "alphagenome_atlas_raw_score": pl.Float64},
            orient="row",
        )
        return result, time.perf_counter() - started, pl.DataFrame(timing_rows)


    def coverage_by_chromosome(
        variants: pl.DataFrame,
        annotations: pl.DataFrame,
        value_column: str,
        source: str,
    ) -> pl.DataFrame:
        return (
            variants.select("variant", "chromosome")
            .join(
                annotations.select("variant", value_column),
                on="variant",
                how="left",
                validate="1:1",
            )
            .group_by("chromosome")
            .agg(
                pl.len().alias("n_unique_variants"),
                pl.col(value_column).is_not_null().sum().alias("covered"),
            )
            .with_columns(
                source=pl.lit(source),
                missing=pl.col("n_unique_variants") - pl.col("covered"),
            )
            .select(
                "source",
                "chromosome",
                "n_unique_variants",
                "covered",
                "missing",
            )
            .sort(pl.col("chromosome").str.strip_prefix("chr").cast(pl.UInt8))
        )


    def validate_annotated_output(
        original: pl.DataFrame,
        unique_variants: pl.DataFrame,
        annotation_records: pl.DataFrame,
        annotated: pl.DataFrame,
    ) -> None:
        if annotation_records.height != unique_variants.height:
            raise ValueError("Annotation record count differs from unique variant count")
        if annotation_records.get_column("variant").n_unique() != unique_variants.height:
            raise ValueError("Expected exactly one annotation record per unique variant")
        if annotated.height != original.height:
            raise ValueError("Annotation changed the input row count")
        if not annotated.select(original.columns).equals(original):
            raise ValueError("Annotation changed input values or row order")

        if annotated.schema["alphagenome_atlas_raw_score"] != pl.Float64:
            raise ValueError("AlphaGenome Atlas raw scores must be Float64")
        if annotated.filter(
            (pl.col("alt") == ".") & pl.col("alphagenome_atlas_raw_score").is_not_null()
        ).height:
            raise ValueError("Reference-only records must have null Atlas scores")

        invalid_gpn_star = annotated.filter(
            pl.any_horizontal(
                ((pl.col("alt") == ".") & pl.col(column).is_not_null())
                | ((pl.col("alt") != ".") & ~pl.col(column).is_finite().fill_null(False))
                for column in ANNOTATION_COLUMNS
                if column.startswith("gpn_star_")
            )
        )
        if not invalid_gpn_star.is_empty():
            print(
                f"GPN-Star scores: found {len(invalid_gpn_star)} invalid rows. "
                "Rows must be finite for alternate alleles and null for "
                f"reference-only records: {invalid_gpn_star.head(5).to_dicts()}"
            )

        conflicts = (
            annotated.group_by("variant")
            .agg(
                pl.max_horizontal(
                    pl.col(column).n_unique() for column in ANNOTATION_COLUMNS
                ).alias("annotation_versions")
            )
            .filter(pl.col("annotation_versions") != 1)
        )
        if not conflicts.is_empty():
            raise ValueError(
                "Duplicate gene-level rows received inconsistent annotations: "
                f"{conflicts.head(5).to_dicts()}"
            )

    return (
        ANNOTATION_COLUMNS,
        annotate_alphagenome_atlas,
        annotate_gnocchi,
        annotate_gpn_star,
        annotate_phylop,
        annotate_roulette,
        coverage_by_chromosome,
        validate_annotated_output,
        validate_variants,
    )


@app.cell
def _(collated_path, pl, validate_variants):
    collated = pl.read_csv(
        collated_path,
        separator="\t",
        null_values="-",
        schema_overrides={
            "variant": pl.String,
            "chromosome": pl.String,
            "start": pl.UInt32,
            "end": pl.UInt32,
            "ref": pl.String,
            "alt": pl.String,
        },
    )
    unique_variants = validate_variants(collated)
    print(
        f"Loaded {collated.height:,} rows containing "
        f"{unique_variants.height:,} unique variants"
    )
    return collated, unique_variants


@app.cell
def _(annotate_phylop, annotation_paths, unique_variants):
    phylop_annotations, phylop_elapsed = annotate_phylop(
        unique_variants, annotation_paths["phylop"]
    )
    print(f"PhyloP completed in {phylop_elapsed:.2f} seconds")
    return phylop_annotations, phylop_elapsed


@app.cell
def _(annotate_gnocchi, annotation_paths, unique_variants):
    gnocchi_annotations, gnocchi_elapsed = annotate_gnocchi(
        unique_variants, annotation_paths["gnocchi"]
    )
    print(f"Gnocchi completed in {gnocchi_elapsed:.2f} seconds")
    return gnocchi_annotations, gnocchi_elapsed


@app.cell
def _(annotate_roulette, annotation_paths, unique_variants):
    roulette_annotations, roulette_elapsed, roulette_chromosome_timings = annotate_roulette(
        unique_variants,
        annotation_paths["roulette"],
    )
    print(f"Roulette completed in {roulette_elapsed:.2f} seconds")
    return roulette_annotations, roulette_chromosome_timings, roulette_elapsed


@app.cell
def _(annotate_gpn_star, annotation_paths, gpn_score_names, unique_variants):
    gpn_data = {}

    for name, score_name in gpn_score_names.items():
        annots, elapsed, chrom_timings = annotate_gpn_star(
            unique_variants, annotation_paths[name], score_name=score_name
        )
        gpn_data[name] = (score_name, annots, elapsed, chrom_timings)
        print(f"GPN-Star ({name}) completed in {elapsed:.2f} seconds")
    return (gpn_data,)


@app.cell
def _(annotate_alphagenome_atlas, annotation_paths, unique_variants):
    atlas_annotations, atlas_elapsed, atlas_chromosome_timings = annotate_alphagenome_atlas(
        unique_variants, annotation_paths["alphagenome_atlas"]
    )
    print(f"AlphaGenome Atlas completed in {atlas_elapsed:.2f} seconds")
    return atlas_annotations, atlas_chromosome_timings, atlas_elapsed


@app.cell
def _(
    atlas_annotations,
    coverage_by_chromosome,
    gnocchi_annotations,
    gpn_data,
    phylop_annotations,
    pl,
    roulette_annotations,
    unique_variants,
):
    coverage_summary = pl.concat(
        [
            coverage_by_chromosome(
                unique_variants,
                atlas_annotations,
                "alphagenome_atlas_raw_score",
                "alphagenome_atlas",
            ),
            coverage_by_chromosome(unique_variants, phylop_annotations, "phylop", "phylop"),
            coverage_by_chromosome(
                unique_variants, gnocchi_annotations, "gnocchi_z", "gnocchi"
            ),
            coverage_by_chromosome(
                unique_variants,
                roulette_annotations,
                "roulette_filter",
                "roulette",
            ),
            *[
                coverage_by_chromosome(
                    unique_variants,
                    annots,
                    score_name,
                    gpn_name,
                )
                for (gpn_name, (score_name, annots, _, _)) in gpn_data.items()
            ],
        ]
    )
    coverage_summary
    return


@app.cell
def _(
    atlas_chromosome_timings,
    atlas_elapsed,
    gnocchi_elapsed,
    gpn_data,
    phylop_elapsed,
    pl,
    roulette_chromosome_timings,
    roulette_elapsed,
):
    source_timings = pl.DataFrame(
        {
            "source": ["phylop", "gnocchi", "roulette", "alphagenome_atlas"]
            + list(gpn_data.keys()),
            "elapsed_seconds": [
                phylop_elapsed,
                gnocchi_elapsed,
                roulette_elapsed,
                atlas_elapsed,
                *[elapsed for (_, _, elapsed, _) in gpn_data.values()],
            ],
        }
    )
    (
        source_timings,
        roulette_chromosome_timings,
        atlas_chromosome_timings,
        pl.concat([chrom_timings for (_, _, _, chrom_timings) in gpn_data.values()]),
    )
    return


@app.cell
def join_annotations(
    atlas_annotations,
    collated,
    gnocchi_annotations,
    gpn_data,
    phylop_annotations,
    roulette_annotations,
    validate_annotated_output,
):
    annotation_records = (
        phylop_annotations
        .join(gnocchi_annotations, on="variant", how="inner", validate="1:1")
        .join(roulette_annotations, on="variant", how="inner", validate="1:1")
        .join(atlas_annotations, on="variant", how="inner", validate="1:1")
    )
    for _, _annots, _, _ in gpn_data.values():
        annotation_records = annotation_records.join(
            _annots, on="variant", how="inner", validate="1:1"
        )

    annotated = (
        collated.with_row_index("_input_row")
        .join(annotation_records, on="variant", how="left", validate="m:1")
        .sort("_input_row")
        .drop("_input_row")
    )
    unique_variants_for_validation = collated.select(
        "variant", "chromosome", "start", "end", "ref", "alt"
    ).unique(subset=["variant"], maintain_order=True)
    validate_annotated_output(
        collated, unique_variants_for_validation, annotation_records, annotated
    )
    return (annotated,)


@app.cell
def write_annotations(annotated, annotated_path, collated, pl):
    annotated.write_csv(
        annotated_path,
        separator="\t",
        compression="gzip",
        null_value="-",
    )
    reread = pl.read_csv(
        annotated_path,
        separator="\t",
        null_values="-",
        schema_overrides={
            "variant": pl.String,
            "chromosome": pl.String,
            "start": pl.UInt32,
            "end": pl.UInt32,
            "ref": pl.String,
            "alt": pl.String,
            "gpn_star_v_score": pl.Float32,
            "gpn_star_m_score": pl.Float32,
            "gpn_star_p_score": pl.Float32,
            "alphagenome_atlas_raw_score": pl.Float64,
        },
    )
    if reread.schema != annotated.schema:
        raise ValueError("Reread output schema differs from the written schema")
    if reread.height != annotated.height:
        raise ValueError("Reread output row count differs from the written output")
    if reread.get_column("variant").to_list() != collated.get_column("variant").to_list():
        raise ValueError("Reread output row order differs from the input")
    if not reread.get_column("alphagenome_atlas_raw_score").equals(
        annotated.get_column("alphagenome_atlas_raw_score")
    ):
        raise ValueError("Reread AlphaGenome Atlas raw scores differ from written values")
    print(f"Wrote and verified {annotated_path} ({reread.height:,} rows)")
    annotated_path
    return


@app.cell
def _(ANNOTATION_COLUMNS, annotated, mo, pl):
    mo.vstack(
        [
            mo.md("### Output preview"),
            annotated.select(
                "variant", "chromosome", "start", "end", *ANNOTATION_COLUMNS
            ).head(),
            mo.md("### Overall coverage"),
            annotated.select(
                pl.len().alias("rows"),
                pl.col("variant").n_unique().alias("unique_variants"),
                *(
                    pl.col(column).is_not_null().sum().alias(f"{column}_covered")
                    for column in ANNOTATION_COLUMNS
                ),
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
