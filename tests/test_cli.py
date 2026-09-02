from pathlib import Path

import pytest

from vep_comparisons.cli import parse_args


@pytest.mark.parametrize(
    ("argv", "domain", "leaf", "handler"),
    [
        (["curate", "--vep-output", "vep.tsv", "--selected-variants", "selected.tsv", "--output-prefix", "out"], "curate", None, "_run_curate"),
        (["protein", "score", "--variants", "v.tsv", "--sequences", "s.fa", "--model", "esmc-300m", "--model-dir", "models", "--output", "scores.tsv.gz"], "protein", "score", "_run_protein"),
        (["protein", "recover-saprot", "--mismatches", "m.tsv", "--translations", "t.fa", "--existing-tokens", "x.fa", "--output-dir", "out"], "protein", "recover-saprot", "_run_recovery"),
        (["dna", "score", "--variants", "v.tsv", "--reference", "g.fa", "--model-dir", "weights", "--model-code-dir", "code", "--output-dir", "out"], "dna", "score", "_run_dna_score"),
        (["dna", "collate", "--variants", "v.tsv", "--input-dir", "in", "--num-shards", "2", "--output", "out.tsv.gz"], "dna", "collate", "_run_dna_collate"),
        (["rna", "orthrus", "score", "--variants", "v.tsv", "--transcripts", "t.fa", "--gff3", "a.gff3", "--checkpoint", "model", "--output-dir", "out"], "rna", "score", "_run_orthrus_score"),
        (["rna", "orthrus", "collate", "--variants", "v.tsv", "--transcripts", "t.fa", "--gff3", "a.gff3", "--checkpoint", "model", "--input-dir", "in", "--num-shards", "2", "--output", "out.tsv.gz"], "rna", "collate", "_run_orthrus_collate"),
        (["rna", "rinalmo", "score", "--variants", "v.tsv", "--transcripts", "t.fa", "--gff3", "a.gff3", "--weights", "model.pt", "--output-dir", "out"], "rna", "score", "_run_rinalmo_score"),
        (["rna", "rinalmo", "collate", "--variants", "v.tsv", "--transcripts", "t.fa", "--gff3", "a.gff3", "--weights", "model.pt", "--input-dir", "in", "--num-shards", "2", "--output", "out.tsv.gz"], "rna", "collate", "_run_rinalmo_collate"),
    ],
)
def test_unified_cli_dispatch(
    argv: list[str], domain: str, leaf: str | None, handler: str
) -> None:
    args = parse_args(argv)
    assert args.domain == domain
    if leaf is not None:
        command = getattr(args, f"{domain}_command", None) or args.rna_command
        assert command == leaf
    assert args.handler.__name__ == handler


@pytest.mark.parametrize(
    "flag",
    ["--manifest", "--model-manifest", "--checkpoint-manifest"],
)
def test_manifest_flags_are_removed(flag: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([
            "dna", "score", "--variants", "v.tsv", "--reference", "g.fa",
            "--model-dir", "weights", "--model-code-dir", "code",
            "--output-dir", "out", flag, str(Path("manifest.json")),
        ])
