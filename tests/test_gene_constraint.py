import gzip
import os
from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig
import pytest

from notebooks.compare_gene_constraint import summary_functions
from scripts.collect_gene_constraint import DATA, OUTPUT, collect, gnocchi_scores, validate_scores, write_scores


def write_fixture(directory: Path) -> tuple[Path, Path, Path]:
    gff3 = directory / "genes.gff3.gz"
    rows = []
    genes = [
        ("A.1", "protein_coding", "chr1", 1, 30, [(1, 5), (4, 10), (21, 25)]),
        ("B.2", "protein_coding", "chr1", 5, 9, [(5, 9)]),
        ("C.1", "lncRNA", "chr1", 8, 22, [(8, 12), (18, 22)]),
        ("D.1", "lncRNA", "chr1", 41, 50, [(41, 50)]),
        ("E.1", "protein_coding", "chr2", 1, 5, [(1, 5)]),
        ("X.1", "lncRNA", "chrX", 1, 5, [(1, 5)]),
        ("other.1", "pseudogene", "chr1", 1, 5, [(1, 5)]),
    ]
    for gene, biotype, chrom, start, end, exons in genes:
        attrs = f"gene_id={gene};gene_type={biotype};gene_name={gene.split('.')[0]}"
        rows.append(f"{chrom}\tfixture\tgene\t{start}\t{end}\t.\t+\t.\tID={gene};{attrs}")
        for index, (left, right) in enumerate(exons):
            # Each exon belongs to a different transcript; selection is gene-level.
            tx = f"{gene}.tx{index}"
            rows.append(f"{chrom}\tfixture\ttranscript\t{start}\t{end}\t.\t+\t.\tID={tx};Parent={gene};{attrs}")
            rows.append(f"{chrom}\tfixture\texon\t{left}\t{right}\t.\t+\t.\tParent={tx};gene_id={gene}")
    with gzip.open(gff3, "wt") as handle:
        handle.write("##gff-version 3\n" + "\n".join(rows) + "\n")
    bigwig = directory / "scores.bw"
    with pyBigWig.open(str(bigwig), "w") as handle:
        handle.addHeader([("chr1", 100), ("chr2", 100)])
        handle.addEntries(["chr1"] * 3 + ["chr2"], [0, 4, 20, 0],
                          ends=[4, 12, 24, 2], values=[2.0, -1.0, 3.0, 0.0])
    gnocchi = directory / "gnocchi.tsv.gz"
    with gzip.open(gnocchi, "wt") as handle:
        handle.write("chrom\tstart\tend\tz\nchr1\t2\t8\t2\nchr1\t8\t18\t4\n"
                     "chr1\t18\t24\tNA\nchr1\t24\t28\t-2\n")
    return gff3, bigwig, gnocchi


def test_collection_coordinates_unions_weighting_and_overlap(tmp_path: Path) -> None:
    gff3, bigwig, gnocchi = write_fixture(tmp_path)
    result = collect(gff3, bigwig, gnocchi, tmp_path / "work")
    output = tmp_path / "scores.tsv.gz"
    write_scores(result, output)
    assert pl.read_csv(output, separator="\t", null_values="-").equals(result)
    rows = {(row["gene_id"], row["region"]): row for row in result.to_dicts()}
    assert len(rows) == result.height == 10
    body, exons = rows["A.1", "gene_body"], rows["A.1", "exons"]
    assert (body["gene_start"], body["gene_end"], body["region_length"]) == (0, 30, 30)
    assert exons["region_length"] == 15
    assert rows["B.2", "exons"]["region_length"] == 5
    assert body["phylop_sum"] == 12
    assert body["phylop_covered_bases"] == 16
    assert body["phylop_mean"] == pytest.approx(0.75)
    assert body["phylop_coverage_fraction"] == pytest.approx(16 / 30)
    assert exons["phylop_sum"] == 14
    assert exons["phylop_covered_bases"] == 14
    assert exons["phylop_mean"] == 1
    assert body["gnocchi_sum"] == 44
    assert body["gnocchi_covered_bases"] == 20
    assert body["gnocchi_mean"] == pytest.approx(2.2)
    assert exons["gnocchi_sum"] == 18
    assert exons["gnocchi_covered_bases"] == 9
    assert exons["gnocchi_mean"] == 2
    assert body["opposite_biotype_overlap_bases"] == 15
    assert exons["opposite_biotype_overlap_bases"] == 5
    assert exons["opposite_biotype_overlap_fraction"] == pytest.approx(1 / 3)
    # A and B overlap each other: C must count their shared bases just once.
    assert rows["C.1", "gene_body"]["opposite_biotype_overlap_bases"] == 15
    assert rows["C.1", "exons"]["opposite_biotype_overlap_bases"] == 10
    for region in ("gene_body", "exons"):
        missing = rows["D.1", region]
        for score in ("phylop", "gnocchi"):
            assert missing[f"{score}_mean"] is None
            assert missing[f"{score}_sum"] == missing[f"{score}_covered_bases"] == 0
            assert missing[f"{score}_coverage_fraction"] == 0
        assert missing["opposite_biotype_overlap_bases"] == 0
        assert rows["E.1", region]["phylop_mean"] == 0
        assert rows["E.1", region]["gnocchi_mean"] is None


def test_no_gnocchi_overlaps(tmp_path: Path) -> None:
    (tmp_path / "regions.bed").write_text("chr1\t0\t5\tinterval_0\n")
    source = tmp_path / "gnocchi.tsv"
    source.write_text("chrom\tstart\tend\tz\nchr1\t10\t20\t1\n")
    intervals = pl.DataFrame({"interval_id": ["interval_0"], "gene_id": ["A"], "region": ["exons"]})
    assert gnocchi_scores(source, intervals, tmp_path).is_empty()


def test_summaries_weight_genes_equally_and_use_scores_independently() -> None:
    _, functions = summary_functions.run()
    data = pl.DataFrame({
        "subset": ["all_genes"] * 6, "region": ["exons"] * 6,
        "score": ["phylop"] * 3 + ["gnocchi"] * 3,
        "biotype": ["protein_coding", "protein_coding", "lncRNA"] * 2,
        "value": [2.0, 8.0, 1.0, None, 4.0, 2.0],
        "coverage": [0.1, 1.0, 0.5, 0.0, 0.8, 0.2],
        "covered_bases": [1, 1000, 50, 0, 800, 20],
    })
    summary, differences = functions["summarize"](data)
    pc = summary.filter(pl.col("biotype") == "protein_coding")
    phy = pc.filter(pl.col("score") == "phylop").row(0, named=True)
    gno = pc.filter(pl.col("score") == "gnocchi").row(0, named=True)
    assert phy["mean"] == phy["median"] == 5
    assert phy["q25"] == 3.5 and phy["q75"] == 6.5
    assert gno["total_genes"] == 2 and gno["genes_with_score"] == 1
    assert gno["mean"] == 4 and gno["coverage_mean"] == 0.4
    assert differences.filter(pl.col("score") == "phylop")["mean_difference"][0] == 4


@pytest.mark.skipif(os.environ.get("GENE_CONSTRAINT_REAL_CHECK") != "1",
                    reason="Opt-in check requires the collected table and source tracks")
def test_real_genes_against_direct_base_values() -> None:
    scores = pl.read_csv(OUTPUT, separator="\t", null_values="-")
    validate_scores(scores, 53005)
    bodies = scores.filter(pl.col("region") == "gene_body")
    assert bodies.filter(pl.col("biotype") == "protein_coding").height == 19166
    assert bodies.filter(pl.col("biotype") == "lncRNA").height == 33839
    # Small spans allow an independent per-base check without genome-wide arrays.
    selected = pl.concat([
        bodies.filter(
            pl.col("chromosome") == chrom, pl.col("biotype") == biotype,
            pl.col("region_length").is_between(2000, 15000),
            pl.col("gnocchi_coverage_fraction").is_between(0.1, 0.95),
            pl.col("phylop_coverage_fraction") > 0,
        ).head(1)
        for chrom in ("chr1", "chr2", "chr22")
        for biotype in ("protein_coding", "lncRNA")
    ])
    assert selected.height == 6
    exon_coordinates: dict[str, list[tuple[int, int]]] = {
        gene: [] for gene in selected["gene_id"]
    }
    with gzip.open(DATA / "assemblies/hg38/v50_gencode/gencode.v50.annotation.gff3.gz", "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if fields[2] != "exon":
                continue
            gene = fields[8].split("gene_id=", 1)[1].split(";", 1)[0]
            if gene in exon_coordinates:
                exon_coordinates[gene].append((int(fields[3]) - 1, int(fields[4])))
    windows = pl.read_csv(
        DATA / "vep_comparisons/constraint/gnocchi_constraint_z_genome_1kb.qc.download.txt.gz",
        separator="\t", null_values=["NA", "nan", ".", "-"],
    ).filter(pl.col("z").is_finite())
    with pyBigWig.open(str(DATA / "vep_comparisons/conservation/phylop/hg38.phyloP447way.bw")) as track:
        for gene in selected.iter_rows(named=True):
            start, end, chrom = gene["gene_start"], gene["gene_end"], gene["chromosome"]
            phy = np.asarray(track.values(chrom, start, end), dtype=float)
            gno = np.full(end - start, np.nan)
            for window in windows.filter(pl.col("chrom") == chrom, pl.col("start") < end,
                                         pl.col("end") > start).iter_rows(named=True):
                gno[max(start, window["start"]) - start:min(end, window["end"]) - start] = window["z"]
            exon_mask = np.zeros(end - start, dtype=bool)
            for left, right in exon_coordinates[gene["gene_id"]]:
                exon_mask[left - start:right - start] = True
            for region in ("gene_body", "exons"):
                mask = np.ones(end - start, dtype=bool) if region == "gene_body" else exon_mask
                saved = scores.filter(pl.col("gene_id") == gene["gene_id"],
                                      pl.col("region") == region).row(0, named=True)
                assert saved["region_length"] == mask.sum()
                for name, values in (("phylop", phy), ("gnocchi", gno)):
                    covered = values[mask & np.isfinite(values)]
                    assert saved[f"{name}_covered_bases"] == len(covered)
                    if len(covered):
                        # UCSC writes score sums with limited decimal precision.
                        assert saved[f"{name}_mean"] == pytest.approx(covered.mean(), rel=1e-5, abs=1e-5)
                    else:
                        assert saved[f"{name}_mean"] is None
            print(f"Direct body/exon checks passed: {gene['gene_id']} ({gene['gene_name']})")
