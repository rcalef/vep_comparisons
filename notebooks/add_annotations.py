import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import os
    import re
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import pyBigWig
    import pysam

    return (
        Path,
        ThreadPoolExecutor,
        as_completed,
        math,
        mo,
        os,
        pl,
        pyBigWig,
        pysam,
        re,
        time,
    )


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

    annotation_paths = {
        "phylop": data_dir / "conservation" / "phylop" / "hg38.phyloP447way.bw",
        "gnocchi": data_dir
        / "constraint"
        / "gnocchi_constraint_z_genome_1kb.qc.download.txt.gz",
        "roulette": data_dir / "constraint" / "roulette",
    }
    return annotated_path, annotation_paths, collated_path


@app.cell
def _(
    Path,
    ThreadPoolExecutor,
    as_completed,
    math,
    pl,
    pyBigWig,
    pysam,
    re,
    time,
):
    REQUIRED_COLUMNS = ("variant", "chromosome", "start", "end", "ref", "alt")
    ANNOTATION_COLUMNS = (
        "phylop",
        "gnocchi_z",
        "roulette_mr",
        "roulette_ar",
        "roulette_filter",
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
                "Conflicting coordinates or alleles for the same variant: "
                f"{examples}"
            )

        return frame.select(REQUIRED_COLUMNS).unique(
            subset=["variant"], maintain_order=True
        )

    def annotate_phylop(
        variants: pl.DataFrame, path: Path
    ) -> tuple[pl.DataFrame, float]:
        started = time.perf_counter()
        rows = []
        with pyBigWig.open(str(path)) as bigwig:
            for row in variants.select("variant", "chromosome", "start", "end").iter_rows(
                named=True
            ):
                values = bigwig.values(
                    row["chromosome"], row["start"], row["end"]
                )
                value = values[0] if values else None
                if value is not None and not math.isfinite(value):
                    value = None
                rows.append({"variant": row["variant"], "phylop": value})
        result = pl.DataFrame(
            rows, schema={"variant": pl.String, "phylop": pl.Float64}
        )
        return result, time.perf_counter() - started

    def annotate_gnocchi(
        variants: pl.DataFrame, path: Path
    ) -> tuple[pl.DataFrame, float]:
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
            .rename({
                "chrom": "chromosome",
                "start": "gnocchi_start",
                "end": "gnocchi_end",
                "z": "gnocchi_z",
            })
            .sort("chromosome", "gnocchi_start")
        )
        result = (
            variants
            .select("variant", "chromosome", "start")
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

    def parse_vcf_info(info: str) -> dict[str, str | None]:
        parsed = {}
        if info == ".":
            return parsed
        for item in info.split(";"):
            key, separator, value = item.partition("=")
            if key in parsed:
                raise ValueError(f"Duplicate INFO key {key!r}: {info!r}")
            parsed[key] = value if separator else None
        return parsed

    def optional_float(value: str | None) -> float | None:
        if value in {None, "", ".", "-"}:
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    def annotate_roulette_chromosome(
        chromosome: str,
        variants: pl.DataFrame,
        roulette_dir: Path,
    ) -> tuple[pl.DataFrame, float]:
        started = time.perf_counter()
        chromosome_id = chromosome.removeprefix("chr")
        path = roulette_dir / f"{chromosome_id}_rate_v5.2_TFBS_correction_all.vcf.bgz"
        index_path = Path(f"{path}.csi")
        if not path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"Missing Roulette VCF or CSI index for {chromosome}")

        rows = []
        with pysam.TabixFile(str(path), index=str(index_path)) as tabix:
            for variant in variants.select(REQUIRED_COLUMNS).iter_rows(named=True):
                output_row = {
                    "variant": variant["variant"],
                    "roulette_mr": None,
                    "roulette_ar": None,
                    "roulette_filter": None,
                }
                # ClinVar includes reference-only records with ALT=".". They
                # can receive coordinate-based annotations but have no allele
                # to match against Roulette.
                if variant["alt"] == ".":
                    rows.append(output_row)
                    continue

                exact_matches = []
                for record in tabix.fetch(
                    chromosome_id, variant["start"], variant["end"]
                ):
                    fields = record.split("\t")
                    if len(fields) < 8:
                        raise ValueError(f"Malformed Roulette VCF record: {record!r}")
                    if (
                        int(fields[1]) == variant["end"]
                        and fields[3] == variant["ref"]
                        and fields[4] == variant["alt"]
                    ):
                        exact_matches.append(fields)

                if len(exact_matches) > 1:
                    raise ValueError(
                        "Ambiguous Roulette allele match for "
                        f"{variant['variant']}: {len(exact_matches)} records"
                    )

                if exact_matches:
                    fields = exact_matches[0]
                    info = parse_vcf_info(fields[7])
                    output_row.update(
                        roulette_mr=optional_float(info.get("MR")),
                        roulette_ar=optional_float(info.get("AR")),
                        roulette_filter=fields[6],
                    )
                rows.append(output_row)

        result = pl.DataFrame(
            rows,
            schema={
                "variant": pl.String,
                "roulette_mr": pl.Float64,
                "roulette_ar": pl.Float64,
                "roulette_filter": pl.String,
            },
        )
        return result, time.perf_counter() - started

    def annotate_roulette(
        variants: pl.DataFrame,
        roulette_dir: Path,
        max_workers: int = 4,
    ) -> tuple[pl.DataFrame, float, pl.DataFrame]:
        if max_workers < 1 or max_workers > 4:
            raise ValueError("Roulette max_workers must be between 1 and 4")
        started = time.perf_counter()
        groups = {
            row[0]: rows
            for row, rows in variants.partition_by(
                "chromosome", as_dict=True, maintain_order=True
            ).items()
        }
        results = []
        timing_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chromosome = {
                executor.submit(
                    annotate_roulette_chromosome,
                    chromosome,
                    chromosome_variants,
                    roulette_dir,
                ): chromosome
                for chromosome, chromosome_variants in groups.items()
            }
            for future in as_completed(future_to_chromosome):
                chromosome = future_to_chromosome[future]
                result, elapsed = future.result()
                results.append(result)
                timing_rows.append(
                    {
                        "source": "roulette",
                        "chromosome": chromosome,
                        "elapsed_seconds": elapsed,
                    }
                )

        result = pl.concat(results).sort(
            pl.col("variant").str.extract(r"^chr(\d+):", 1).cast(pl.UInt8),
            pl.col("variant").str.extract(r"^chr\d+:(\d+):", 1).cast(pl.UInt32),
        )
        chromosome_timings = pl.DataFrame(timing_rows).sort(
            pl.col("chromosome").str.strip_prefix("chr").cast(pl.UInt8)
        )
        return result, time.perf_counter() - started, chromosome_timings

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
        annotate_gnocchi,
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
    roulette_annotations, roulette_elapsed, roulette_chromosome_timings = (
        annotate_roulette(
            unique_variants,
            annotation_paths["roulette"],
            max_workers=4,
        )
    )
    print(f"Roulette completed in {roulette_elapsed:.2f} seconds with four workers")
    return roulette_annotations, roulette_chromosome_timings, roulette_elapsed


@app.cell
def _(
    coverage_by_chromosome,
    gnocchi_annotations,
    phylop_annotations,
    pl,
    roulette_annotations,
    unique_variants,
):
    coverage_summary = pl.concat(
        [
            coverage_by_chromosome(
                unique_variants, phylop_annotations, "phylop", "phylop"
            ),
            coverage_by_chromosome(
                unique_variants, gnocchi_annotations, "gnocchi_z", "gnocchi"
            ),
            coverage_by_chromosome(
                unique_variants,
                roulette_annotations,
                "roulette_filter",
                "roulette",
            ),
        ]
    )
    coverage_summary


@app.cell
def _(
    gnocchi_elapsed,
    phylop_elapsed,
    pl,
    roulette_chromosome_timings,
    roulette_elapsed,
):
    source_timings = pl.DataFrame(
        {
            "source": ["phylop", "gnocchi", "roulette"],
            "elapsed_seconds": [phylop_elapsed, gnocchi_elapsed, roulette_elapsed],
        }
    )
    source_timings, roulette_chromosome_timings


@app.cell
def _(
    collated,
    gnocchi_annotations,
    phylop_annotations,
    roulette_annotations,
    validate_annotated_output,
):
    annotation_records = (
        phylop_annotations.join(
            gnocchi_annotations, on="variant", how="inner", validate="1:1"
        )
        .join(roulette_annotations, on="variant", how="inner", validate="1:1")
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
def _(annotated, annotated_path, collated, pl):
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
        },
    )
    if reread.schema != annotated.schema:
        raise ValueError("Reread output schema differs from the written schema")
    if reread.height != annotated.height:
        raise ValueError("Reread output row count differs from the written output")
    if reread.get_column("variant").to_list() != collated.get_column("variant").to_list():
        raise ValueError("Reread output row order differs from the input")
    print(f"Wrote and verified {annotated_path} ({reread.height:,} rows)")
    annotated_path


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


if __name__ == "__main__":
    app.run()
