from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pysam
import pytest
import torch

from vep_comparisons.cli import parse_args
from vep_comparisons.dna import evo2
from vep_comparisons.dna.evo2 import Evo2Model, allele_sequences, pool_bounds, sequence_log_likelihoods
from vep_comparisons.dna.windows import ScoringContext
from vep_comparisons.dna.workflow import collate_dna_scores, score_dna_variants
from vep_comparisons.linear_probes.add_evo2_scores import add_evo2_scores
from vep_comparisons.linear_probes.collate import collate_model
from vep_comparisons.linear_probes.extract_evo2 import extract_embeddings


class Reference:
    sequence = "ACGTACGT"

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        return self.sequence[start:end]

    def get_reference_length(self, chromosome: str) -> int:
        return len(self.sequence)


class FakeEvo:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(tokenize=lambda seq: list(seq.encode()))
        self.inputs: list[torch.Tensor] = []

    def __call__(
        self, inputs: torch.Tensor, *, return_embeddings: bool, layer_names: list[str] | None,
    ) -> tuple[tuple[torch.Tensor, None], dict[str, torch.Tensor]]:
        assert not torch.is_grad_enabled()
        assert layer_names == ([evo2.EMBEDDING_LAYER] if return_embeddings else None)
        self.inputs.append(inputs.clone())
        positions = torch.arange(inputs.shape[1]).expand_as(inputs)
        prefix = inputs.cumsum(dim=-1)
        # Next-token distributions depend on every earlier input, detecting focal-only scoring.
        logits = (torch.arange(128)[None, None, :] - (prefix % 128).unsqueeze(-1)).abs().neg().bfloat16()
        hidden = torch.stack([inputs, positions], dim=-1).float()
        return (logits, None), {evo2.EMBEDDING_LAYER: hidden}


def fake_adapter() -> Evo2Model:
    model = object.__new__(Evo2Model)
    model.evo = FakeEvo()
    model.device = torch.device("cpu")
    return model


def test_causal_full_vocabulary_likelihood_and_precision() -> None:
    inputs = torch.tensor([[0, 1, 2, 0]])
    logits = torch.tensor([[[1., 0., -1.], [0., 2., 1.], [3., 0., 1.], [9., 9., 9.]]]).bfloat16()
    expected = np.mean([
        float(logits[0, i, inputs[0, i + 1]])
        - np.log(np.exp(logits[0, i].float().numpy().astype(np.float64)).sum())
        for i in range(3)
    ])
    values = sequence_log_likelihoods(logits, inputs)
    assert values.dtype == np.float64
    assert values[0] == pytest.approx(expected, abs=1e-7)


def test_four_alleles_padding_and_mirrored_pooling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evo2, "POOL_LENGTH", 4)
    context = ScoringContext("chr1", -3, 8, 4, "C", "T")
    assert allele_sequences(Reference(), context) == ["NNNACGTA", "TACGTNNN", "NNNATGTA", "TACATNNN"]
    assert pool_bounds(context) == [(2, 6)] * 4
    model = fake_adapter()
    scores, vectors = model.score_and_embed_contexts([context], reference=Reference(), batch_size=1)
    sequences = allele_sequences(Reference(), context)
    expected = np.concatenate([
        [np.mean(list(seq.encode())[2:6]), 3.5] for seq in sequences
    ])[None, :]
    np.testing.assert_array_equal(vectors, expected)
    assert vectors.dtype == np.float32
    assert len(model.evo.inputs) == 4
    assert all(ids.shape == (1, 8) for ids in model.evo.inputs)
    batched_scores, batched_vectors = model.score_and_embed_contexts([context], reference=Reference(), batch_size=4)
    np.testing.assert_array_equal(batched_vectors, vectors)
    assert batched_scores == scores
    likelihoods, _ = model._predict(sequences, None)
    assert scores[context.key].forward == pytest.approx(likelihoods[2] - likelihoods[0])
    assert scores[context.key].reverse == pytest.approx(likelihoods[3] - likelihoods[1])
    assert model.score_contexts([context], reference=Reference(), batch_size=2) == scores
    np.testing.assert_array_equal(model.embed_contexts([context], reference=Reference(), batch_size=3), vectors)


def test_pool_bounds_for_off_center_context() -> None:
    assert pool_bounds(ScoringContext("chr1", 0, 8192, 5000, "A", "T")) == [
        (4936, 5064), (3128, 3256), (4936, 5064), (3128, 3256),
    ]


def test_evo2_cli_and_model_specific_defaults() -> None:
    base = ["dna", "score", "--variants", "v.tsv", "--reference", "r.fa", "--output-dir", "out"]
    args = parse_args([*base, "--model", "evo2-7b", "--checkpoint", "model.pt"])
    assert args.dtype == "bfloat16" and args.window_length == 8192
    assert args.model_dir is None and args.model_code_dir is None
    for extra in ([], ["--dtype", "float32"], ["--model-dir", "unused"]):
        with pytest.raises(SystemExit):
            parse_args([*base, "--model", "evo2-7b", *([] if not extra else ["--checkpoint", "model.pt", *extra])])
    with pytest.raises(SystemExit):
        parse_args(base)
    nt = parse_args([*base, "--model-dir", "weights", "--model-code-dir", "code"])
    assert nt.dtype == "float32"


def test_evo2_loading_uses_local_weights_and_preserves_native_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    calls = []
    backbone = torch.nn.Linear(1, 1)
    def load(name: str, **kwargs: object) -> SimpleNamespace:
        calls.append((name, kwargs))
        return SimpleNamespace(model=backbone)
    monkeypatch.setitem(sys.modules, "evo2", SimpleNamespace(Evo2=load))
    adapter = Evo2Model(Path("local.pt"))
    assert calls == [("evo2_7b", {"local_path": "local.pt", "use_kernels": False})]
    assert not backbone.training and backbone.weight.dtype == torch.float32
    assert adapter.device == torch.device("cuda:0")
    with pytest.raises(ValueError):
        Evo2Model(Path("local.pt"), dtype="float32")
    with pytest.raises(ValueError):
        Evo2Model(Path("local.pt"), device="cuda:1")


def site_rows() -> pl.DataFrame:
    return pl.DataFrame({
        "site_id": ["chr1:2", "chr1:5"], "context_id": ["v1|g1|t1", "v2|g2|t2"],
        "variant": ["v1", "v2"], "gene": ["g1", "g2"], "feature": ["t1", "t2"],
        "chromosome": ["chr1"] * 2, "start": [1, 4], "end": [2, 5],
        "ref": ["C", "A"], "alt": ["T", "C"],
    })


def test_scoring_extraction_and_fixed_benchmark_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = tmp_path / "reference.fa"
    reference.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(reference))
    sites = site_rows()
    variants = tmp_path / "variants.tsv"
    sites.write_csv(variants, separator="\t")
    monkeypatch.setattr(evo2, "Evo2Model", lambda *args, **kwargs: fake_adapter())
    scoring = tmp_path / "scoring"
    for shard in range(2):
        score_dna_variants(
            variants_path=variants, reference_path=reference, checkpoint=Path("fake.pt"),
            output_dir=scoring, model_name="evo2-7b", num_shards=2, shard_index=shard,
        )
    score_file = collate_dna_scores(
        variants_path=variants, output_dir=scoring, output_path=tmp_path / "scores.tsv.gz", num_shards=2,
    )
    scores = pl.read_csv(score_file, separator="\t")
    assert scores["model"].to_list() == ["evo2-7b"] * 2
    membership = sites.with_columns(cohort=pl.lit("dna_rna"), task=pl.lit("phylop"), outer_fold=pl.lit(0), label=pl.lit(1))
    manifest = pl.concat([membership, membership.with_columns(task=pl.lit("gnocchi_z"))])
    extended = add_evo2_scores(manifest, scores)
    assert extended.drop("evo2_score").equals(manifest)
    np.testing.assert_array_equal(extended["evo2_score"].to_numpy(), np.tile(-scores["score"].to_numpy(), 2))
    with pytest.raises(AssertionError):
        add_evo2_scores(manifest, scores.head(1))
    with pytest.raises(AssertionError):
        add_evo2_scores(manifest, scores.with_columns(model=pl.lit("ntv3-100m-pre")))
    embeddings = tmp_path / "embeddings"
    embedding_file, extracted_scores = extract_embeddings(
        sites, reference, fake_adapter(), embeddings, batch_size=2, num_shards=1, shard_index=0,
    )
    extracted = pl.read_csv(extracted_scores, separator="\t")
    assert extracted.equals(scores)
    with np.load(embedding_file, allow_pickle=False) as saved:
        assert saved["site_id"].tolist() == sites["site_id"].to_list()
        assert saved["context_id"].tolist() == sites["context_id"].to_list()
        assert saved["embeddings"].shape == (2, 8)
    # Synthetic hidden states have two channels instead of the checkpoint's 4096.
    from vep_comparisons.linear_probes.constants import DIMENSIONS
    monkeypatch.setitem(DIMENSIONS, "evo2", 8)
    summary = collate_model(manifest, "evo2", embeddings, tmp_path / "evo2.npz", 1)
    assert summary["sites"] == 2


def test_collate_cli_supports_sixteen_evo2_shards_and_four_existing_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from vep_comparisons.linear_probes import collate
    from vep_comparisons.linear_probes.constants import DIMENSIONS

    manifest = pl.DataFrame({
        "site_id": [f"s{i:02d}" for i in range(16)],
        "context_id": [f"c{i:02d}" for i in range(16)], "cohort": ["dna_rna"] * 16,
    })
    manifest_file = tmp_path / "manifest.parquet"
    manifest.write_parquet(manifest_file)
    for model, shards in (("orthrus", 4), ("evo2", 16)):
        directory = tmp_path / "embeddings" / model
        directory.mkdir(parents=True)
        for shard in range(shards):
            rows = manifest.gather_every(shards, offset=shard)
            np.savez(
                directory / f"shard-{shard:05d}-of-{shards:05d}.npz",
                site_id=np.array(rows["site_id"].to_list()), context_id=np.array(rows["context_id"].to_list()),
                embeddings=np.ones((rows.height, DIMENSIONS[model]), dtype=np.float32),
            )
    monkeypatch.setattr(sys, "argv", [
        "collate", "--manifest", str(manifest_file), "--input-dir", str(tmp_path / "embeddings"),
        "--output-dir", str(tmp_path / "collated"), "--models", "orthrus", "evo2",
        "--num-shards", "4", "--evo2-num-shards", "16",
    ])
    collate.main()
    summary = pl.read_csv(tmp_path / "collated/summary.tsv", separator="\t")
    assert summary["shards"].to_list() == [4, 16]
    assert summary["sites"].to_list() == [16, 16]
