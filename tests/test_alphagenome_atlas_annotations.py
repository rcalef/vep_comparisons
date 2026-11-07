import gzip
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pysam

from notebooks.add_annotations import (
    annotation_functions,
    join_annotations,
    write_annotations,
)


def test_atlas_matching_and_gene_row_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "atlas.tsv"
    source.write_text(
        "#CHROM\tPOS\tREF\tALT\traw_score\tPHRED\n"
        "chr1\t1\tA\tC\t-0.123456789012345\t9\n"
        "chr1\t1\tA\tG\t0.25\t8\n"
        "chr1\t2\tC\tT\t2.5\t7\n"
        "chr1\t3\tC\tT\t3.5\t6\n"
        "chr2\t1\tA\tC\t4.5\t5\n"
    )
    indexed = pysam.tabix_index(
        str(source), seq_col=0, start_col=1, end_col=1, zerobased=False
    )
    original = (
        pl.DataFrame(
            {
                "variant": [
                    "chr2:1:A:C", "chr1:2:C:T", "chr1:1:A:G", "chr1:1:A:C",
                    "chr1:1:T:C", "chr1:1:A:T", "chr1:4:C:T", "chr1:1:A:.",
                    "chr1:1:A:C", "chr1:3:C:T", "chr1:5:A:.", "chr3:1:A:C",
                ],
                "gene": [f"gene{i}" for i in range(12)],
                "original_value": list(range(12)),
            }
        )
        .with_columns(parts=pl.col("variant").str.split(":"))
        .with_columns(
            chromosome=pl.col("parts").list.get(0),
            end=pl.col("parts").list.get(1).cast(pl.UInt32),
            ref=pl.col("parts").list.get(2),
            alt=pl.col("parts").list.get(3),
        )
        .with_columns(start=pl.col("end") - 1)
        .drop("parts")
    )
    with pysam.TabixFile(indexed) as tabix:
        handle = MagicMock(wraps=tabix)
        handle.contigs = tabix.contigs
        factory = MagicMock()
        factory.return_value.__enter__.return_value = handle
        _, functions = annotation_functions.run(pysam=SimpleNamespace(TabixFile=factory))
        scores, _, timings = functions["annotate_alphagenome_atlas"](original, Path(indexed))
        factory.assert_called_once_with(indexed)
        assert [call.args for call in handle.fetch.call_args_list] == [
            ("chr1", 0, 1), ("chr1", 1, 2), ("chr1", 2, 3),
            ("chr1", 3, 4), ("chr2", 0, 1),
        ]

    assert scores.height == original["variant"].n_unique()
    assert scores.schema["alphagenome_atlas_raw_score"] == pl.Float64
    assert timings["chromosome"].to_list() == ["chr1", "chr2", "chr3"]
    unique = functions["validate_variants"](original)
    other_scores = unique.select("variant").with_columns(
        pl.lit("PASS" if name == "roulette_filter" else 0.5).alias(name)
        for name in functions["ANNOTATION_COLUMNS"]
        if name != "alphagenome_atlas_raw_score"
    )
    _, joined = join_annotations.run(
        atlas_annotations=scores,
        collated=original,
        gnocchi_annotations=other_scores.select("variant", "gnocchi_z"),
        gpn_data={
            name: (name, other_scores.select("variant", name).cast({name: pl.Float32}), 0, None)
            for name in ("gpn_star_v_score", "gpn_star_m_score", "gpn_star_p_score")
        },
        phylop_annotations=other_scores.select("variant", "phylop"),
        roulette_annotations=other_scores.select("variant", "roulette_mr", "roulette_ar", "roulette_filter"),
        validate_annotated_output=functions["validate_annotated_output"],
    )
    annotated = joined["annotated"]
    assert annotated.select(original.columns).equals(original)
    assert annotated["alphagenome_atlas_raw_score"].to_list() == [
        4.5, 2.5, 0.25, -0.123456789012345, None, None, None, None,
        -0.123456789012345, 3.5, None, None,
    ]
    destination = tmp_path / "annotated.tsv.gz"
    write_annotations.run(
        annotated=annotated, annotated_path=destination, collated=original
    )
    reread = pl.read_csv(destination, separator="\t", null_values="-", schema_overrides=annotated.schema)
    assert reread.equals(annotated)
    with gzip.open(destination, "rt") as output:
        lines = output.read().splitlines()
    score_index = lines[0].split("\t").index("alphagenome_atlas_raw_score")
    assert lines[5].split("\t")[score_index] == "-"
    assert "PHRED" not in annotated.columns
