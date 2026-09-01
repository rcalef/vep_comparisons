from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pysam
import pytest

from vep_comparisons.dna_variant_scoring import (
    OUTPUT_COLUMNS,
    CandidatePlan,
    DnaCandidate,
    GeneSpan,
    InputValidationError,
    OrientationScores,
    ScoringContext,
    ShardCompatibilityError,
    assemble_rows,
    centered_window,
    collate_dna_scores,
    complemented,
    deduplicate_contexts,
    extract_window,
    filter_candidates_by_alt,
    plan_candidates,
    plan_gene_window,
    read_dna_candidates,
    read_gene_spans,
    reverse_complement,
    score_dna_variants,
    shard_variant_ids,
    sha256_file,
    validate_and_resolve_candidates,
    validate_window_policy,
)
from vep_comparisons.ntv3_model import EXPECTED_VOCABULARY, validate_model_package


class FakeReference:
    def __init__(self, sequences: dict[str, str]):
        self.sequences = sequences
        self.references = tuple(sequences)

    def get_reference_length(self, chromosome: str) -> int:
        return len(self.sequences[chromosome])

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        return self.sequences[chromosome][start:end]


def candidate(
    position: int,
    *,
    variant: str = "v1",
    gene: str = "ENSG1",
    chromosome: str = "chr1",
    ref: str = "A",
    alt: str = "C",
    input_index: int = 0,
) -> DnaCandidate:
    return DnaCandidate(
        variant, gene, "ENST1", chromosome, position, position + 1, ref, alt, input_index
    )


def write_variants(path: Path, rows: list[DnaCandidate]) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "gene", "feature", "chromosome", "start", "end", "ref", "alt"), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in writer.fieldnames})


def write_gtf(path: Path, spans: list[GeneSpan]) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write("##gtf-version 2.2\n")
        for span in spans:
            handle.write(
                f'{span.chromosome}\ttest\tgene\t{span.start + 1}\t{span.end}\t.\t{span.strand}\t.\tgene_id "{span.gene}.7";\n'
            )


def write_reference(path: Path, sequence: str) -> None:
    path.write_text(f">chr1\n{sequence}\n")
    pysam.faidx(str(path))


def make_model_package(tmp_path: Path) -> tuple[Path, Path, Path]:
    weights = tmp_path / "weights"
    code = tmp_path / "code"
    weights.mkdir()
    code.mkdir()
    config = {
        "architectures": ["NTv3PreTrained"],
        "alphabet_size": 11,
        "num_downsamples": 7,
        "mask_token_id": 2,
        "pad_token_id": 1,
    }
    (weights / "config.json").write_text(json.dumps(config))
    (weights / "model.safetensors").write_bytes(b"fake")
    (weights / "tokenizer_config.json").write_text("{}")
    (weights / "vocab.json").write_text(json.dumps(EXPECTED_VOCABULARY))
    (code / "__init__.py").write_text("")
    for name in (
        "configuration_ntv3_pretrained.py",
        "modeling_ntv3_pretrained.py",
        "tokenization_ntv3.py",
    ):
        (code / name).write_text(f"# {name}\n")
    manifest = {
        "model": "ntv3-100m-pre",
        "weights": {
            "repository": "weights",
            "revision": "1" * 40,
            "sha256": {name: sha256_file(weights / name) for name in ("config.json", "model.safetensors", "tokenizer_config.json", "vocab.json")},
        },
        "code": {
            "repository": "code",
            "revision": "2" * 40,
            "sha256": {name: sha256_file(code / name) for name in ("configuration_ntv3_pretrained.py", "modeling_ntv3_pretrained.py", "tokenization_ntv3.py")},
        },
        "expected": {
            "alphabet_size": 11,
            "num_downsamples": 7,
            "token_ids": EXPECTED_VOCABULARY,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return weights, code, manifest_path


class FakeModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.contexts: list[ScoringContext] = []

    def score_contexts(self, contexts, *, reference, batch_size):
        self.contexts.extend(contexts)
        if self.fail:
            raise RuntimeError("simulated inference failure")
        return {
            context.key: OrientationScores(
                float(context.window_start), float(context.window_start + 2)
            )
            for context in contexts
        }


def test_center_convention_and_boundary_padding() -> None:
    reference = FakeReference({"chr1": "A" * 200})
    first = candidate(5)
    first_window = centered_window(first, 200, 128)
    assert (first_window.start, first_window.end, first_window.target_index) == (-59, 69, 64)
    assert (first_window.left_pad, first_window.right_pad) == (59, 0)
    context = ScoringContext("chr1", -59, 128, 64, "A", "C")
    sequence = extract_window(reference, context)
    assert sequence == "N" * 59 + "A" * 69
    assert sequence[64] == "A"

    last_window = centered_window(candidate(198), 200, 128)
    assert (last_window.start, last_window.end, last_window.right_pad) == (134, 262, 62)
    assert extract_window(reference, ScoringContext("chr1", 134, 128, 64, "A", "C")) == "A" * 66 + "N" * 62


def test_reverse_complement_alleles_and_target_index() -> None:
    assert reverse_complement("ATCGN") == "NCGAT"
    assert {base: complemented(base) for base in "ACGT"} == {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
    }
    context = ScoringContext("chr1", 10, 128, 31, "A", "G")
    assert context.window_length - 1 - context.target_index == 96


@pytest.mark.parametrize(
    ("window", "margin"), [(0, 0), (127, 0), (128, -1), (128, 64)]
)
def test_window_policy_rejects_invalid_values(window: int, margin: int) -> None:
    with pytest.raises(InputValidationError):
        validate_window_policy(window, margin)


def test_gene_context_statuses_and_nearest_shift_ignore_strand() -> None:
    centered = centered_window(candidate(70), 500, 128)
    assert centered.start == 6
    for strand in ("+", "-"):
        status, shifted = plan_gene_window(
            candidate(70),
            centered,
            GeneSpan("ENSG1", "chr1", 0, 100, strand),
            contig_length=500,
            window_length=128,
            min_variant_margin=10,
        )
        assert status == "shifted_full_gene"
        assert shifted is not None
        assert (shifted.start, shifted.end, shifted.target_index) == (0, 128, 70)

    status, same = plan_gene_window(
        candidate(70), centered, GeneSpan("ENSG1", "chr1", 20, 100, "+"),
        contig_length=500, window_length=128, min_variant_margin=10
    )
    assert status == "centered_full_gene" and same == centered
    status, window = plan_gene_window(
        candidate(70), centered, GeneSpan("ENSG1", "chr1", 0, 200, "+"),
        contig_length=500, window_length=128, min_variant_margin=10
    )
    assert (status, window) == ("gene_too_long", None)
    assert plan_gene_window(
        candidate(70), centered, None, contig_length=500, window_length=128, min_variant_margin=10
    ) == ("missing_gene_span", None)
    with pytest.raises(InputValidationError, match="gene_contig_mismatch"):
        plan_gene_window(
            candidate(70), centered, GeneSpan("ENSG1", "chr2", 0, 10, "+"),
            contig_length=500, window_length=128, min_variant_margin=10
        )


def test_multi_gene_plans_reuse_centered_and_deduplicate_shifted_contexts() -> None:
    reference = FakeReference({"chr1": "A" * 500})
    rows = [candidate(70, gene="G1", input_index=0), candidate(70, gene="G2", input_index=1)]
    spans = {gene: GeneSpan(gene, "chr1", 0, 100, "+") for gene in ("G1", "G2")}
    plans = plan_candidates(rows, spans, reference, window_length=128, min_variant_margin=10)
    assert plans[0].centered is plans[1].centered
    contexts = deduplicate_contexts(plans)
    assert len(contexts) == 2  # one centered and one identical shifted request


def test_candidate_and_gtf_validation_aggregates_failures(tmp_path: Path) -> None:
    variants = tmp_path / "variants.tsv.gz"
    write_variants(
        variants,
        [
            candidate(2, variant="bad-ref", ref="C"),
            candidate(3, variant="bad-allele", alt="N", input_index=1),
            candidate(4, variant="same", alt="A", input_index=2),
        ],
    )
    reference = FakeReference({"chr1": "A" * 10})
    with pytest.raises(InputValidationError) as caught:
        validate_and_resolve_candidates(read_dna_candidates(variants), reference)
    categories = {issue.category for issue in caught.value.issues}
    assert categories == {"reference_mismatch", "invalid_allele", "identical_alleles"}

    gtf = tmp_path / "genes.gtf.gz"
    write_gtf(
        gtf,
        [GeneSpan("ENSG1", "chr1", 0, 10, "+"), GeneSpan("ENSG1", "chr1", 1, 10, "+")],
    )
    with pytest.raises(InputValidationError, match="conflicting_gene_span"):
        read_gene_spans(gtf, reference.references)


def test_invalid_alt_rows_are_filtered_before_candidate_validation(tmp_path: Path) -> None:
    variants = tmp_path / "variants.tsv.gz"
    write_variants(
        variants,
        [
            candidate(1, variant="valid"),
            candidate(2, variant="dot", alt=".", input_index=1),
            candidate(3, variant="ambiguous", alt="N", input_index=2),
            candidate(4, variant="empty", alt="", input_index=3),
        ],
    )
    retained, filtered = filter_candidates_by_alt(read_dna_candidates(variants))
    assert [item.variant for item in retained] == ["valid"]
    assert [item.variant for item in filtered] == ["dot", "ambiguous", "empty"]


def test_shards_are_contiguous_balanced_and_variant_level() -> None:
    rows = []
    for index in range(7):
        rows.append(candidate(index, variant=f"v{index}", gene="G1", input_index=2 * index))
        if index == 3:
            rows.append(candidate(index, variant=f"v{index}", gene="G2", input_index=2 * index + 1))
    shards = [shard_variant_ids(rows, ("chr1",), num_shards=3, shard_index=index) for index in range(3)]
    assert shards == [["v0", "v1"], ["v2", "v3"], ["v4", "v5", "v6"]]
    assert sum(shards, []) == [f"v{index}" for index in range(7)]


def test_fake_score_arithmetic_and_centered_gene_copy() -> None:
    item = candidate(64)
    window = centered_window(item, 300, 128)
    plans = [CandidatePlan(item, window, "centered_full_gene", window)]
    context = ScoringContext("chr1", 0, 128, 64, "A", "C")
    rows = assemble_rows(plans, {context.key: OrientationScores(2.0, 4.0)}, window_length=128)
    assert rows[0]["score"] == 3.0
    assert rows[0]["gene_score_forward"] == 2.0
    assert rows[0]["gene_score_reverse"] == 4.0
    assert rows[0]["gene_score"] == 3.0


def test_manifest_validation_checks_hashes(tmp_path: Path) -> None:
    weights, code, manifest = make_model_package(tmp_path)
    result = validate_model_package(weights, code, manifest)
    assert result["weights"]["revision"] == "1" * 40
    (code / "tokenization_ntv3.py").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_model_package(weights, code, manifest)


def test_multishard_scoring_resume_collation_and_incompatibility(tmp_path: Path) -> None:
    reference = tmp_path / "reference.fa"
    variants = tmp_path / "variants.tsv.gz"
    genes = tmp_path / "genes.gtf.gz"
    write_reference(reference, "A" * 500)
    rows = [
        candidate(50, variant="v1", gene="G1", input_index=0),
        candidate(50, variant="v1", gene="G2", input_index=1),
        candidate(150, variant="v2", gene="G3", input_index=2),
        candidate(250, variant="v3", gene="MISSING", input_index=3),
        candidate(350, variant="filtered", gene="G1", alt=".", input_index=4),
    ]
    write_variants(variants, rows)
    write_gtf(
        genes,
        [
            GeneSpan("G1", "chr1", 20, 80, "+"),
            GeneSpan("G2", "chr1", 0, 100, "-"),
            GeneSpan("G3", "chr1", 100, 300, "+"),
        ],
    )
    weights, code, manifest = make_model_package(tmp_path)
    output_dir = tmp_path / "shards"
    models: list[FakeModel] = []

    def factory(*args):
        model = FakeModel()
        models.append(model)
        return model

    summaries = []
    for shard_index in range(2):
        summaries.append(
            score_dna_variants(
                variants_path=variants,
                reference_path=reference,
                genes_path=genes,
                model_dir=weights,
                model_code_dir=code,
                output_dir=output_dir,
                manifest_path=manifest,
                window_length=128,
                min_variant_margin=10,
                batch_size=2,
                device="cpu",
                num_shards=2,
                shard_index=shard_index,
                model_factory=factory,
            )
        )
    assert sum(summary.candidates for summary in summaries) == 4
    assert sum(summary.unique_variants for summary in summaries) == 3
    reused = score_dna_variants(
        variants_path=variants,
        reference_path=reference,
        genes_path=genes,
        model_dir=weights,
        model_code_dir=code,
        output_dir=output_dir,
        manifest_path=manifest,
        window_length=128,
        min_variant_margin=10,
        batch_size=99,  # batch size is operational and does not invalidate a completed shard
        device="cpu",
        num_shards=2,
        shard_index=0,
        model_factory=factory,
    )
    assert reused.reused and len(models) == 2

    final = tmp_path / "final.tsv.gz"
    output, metadata = collate_dna_scores(
        variants_path=variants, output_dir=output_dir, output_path=final, num_shards=2
    )
    assert output == final and metadata.is_file()
    with gzip.open(final, "rt") as handle:
        result = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["variant"], row["gene"]) for row in result] == [
        row.key for row in rows if row.alt in {"A", "C", "G", "T"}
    ]
    assert tuple(result[0]) == OUTPUT_COLUMNS
    assert result[3]["gene_context_status"] == "missing_gene_span"
    assert result[3]["gene_score"] == ""
    first_metadata = json.loads(summaries[0].metadata_path.read_text())
    assert first_metadata["counts"]["filtered_invalid_alt_candidates"] == 1
    assert first_metadata["counts"]["filtered_invalid_alt_counts"] == {".": 1}

    shard_meta = summaries[1].metadata_path
    payload = json.loads(shard_meta.read_text())
    payload["run_identity"]["window_length"] = 256
    shard_meta.write_text(json.dumps(payload))
    with pytest.raises(ShardCompatibilityError, match="incompatible"):
        collate_dna_scores(
            variants_path=variants,
            output_dir=output_dir,
            output_path=tmp_path / "bad.tsv.gz",
            num_shards=2,
        )


def test_inference_failure_leaves_no_shard_files(tmp_path: Path) -> None:
    reference = tmp_path / "reference.fa"
    variants = tmp_path / "variants.tsv.gz"
    genes = tmp_path / "genes.gtf.gz"
    write_reference(reference, "A" * 200)
    write_variants(variants, [candidate(50)])
    write_gtf(genes, [GeneSpan("ENSG1", "chr1", 20, 80, "+")])
    weights, code, manifest = make_model_package(tmp_path)
    output_dir = tmp_path / "output"
    with pytest.raises(RuntimeError, match="simulated"):
        score_dna_variants(
            variants_path=variants,
            reference_path=reference,
            genes_path=genes,
            model_dir=weights,
            model_code_dir=code,
            output_dir=output_dir,
            manifest_path=manifest,
            window_length=128,
            min_variant_margin=10,
            device="cpu",
            model_factory=lambda *args: FakeModel(fail=True),
        )
    assert not output_dir.exists()


def test_centered_only_scoring_needs_no_gtf_or_gene_margin(tmp_path: Path) -> None:
    reference = tmp_path / "reference.fa"
    variants = tmp_path / "variants.tsv.gz"
    write_reference(reference, "A" * 200)
    write_variants(
        variants,
        [
            candidate(50, variant="kept"),
            candidate(60, variant="filtered", alt=".", input_index=1),
        ],
    )
    weights, code, manifest = make_model_package(tmp_path)
    summary = score_dna_variants(
        variants_path=variants,
        reference_path=reference,
        genes_path=None,
        model_dir=weights,
        model_code_dir=code,
        output_dir=tmp_path / "output",
        manifest_path=manifest,
        window_length=128,
        min_variant_margin=1024,  # ignored without gene-aware planning
        device="cpu",
        model_factory=lambda *args: FakeModel(),
    )
    with gzip.open(summary.output_path, "rt") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["variant"] == "kept"
    assert rows[0]["gene_context_status"] == "not_requested"
    assert rows[0]["window_length"] == "128"
    assert rows[0]["gene_score"] == ""
    metadata = json.loads(summary.metadata_path.read_text())
    assert metadata["run_identity"]["genes"] is None
    assert metadata["run_identity"]["gene_context_mode"] == "centered_only"
    assert metadata["counts"]["filtered_invalid_alt_candidates"] == 1
