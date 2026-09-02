from __future__ import annotations

from pathlib import Path

import torch

from vep_comparisons.cli import parse_args
from vep_comparisons.dna.ntv3 import EXPECTED_VOCABULARY, NTv3Model, _load_code_package


class FakeTokenizer:
    def __init__(self) -> None:
        self.last_input_ids = None

    def __call__(self, sequences, *, add_special_tokens, padding, return_tensors):
        assert not add_special_tokens
        assert not padding
        ids = torch.tensor(
            [[EXPECTED_VOCABULARY[base] for base in sequence] for sequence in sequences]
        )
        self.last_input_ids = ids
        return {"input_ids": ids}


class FakeMaskedLM:
    def __init__(self) -> None:
        self.masked = None

    def __call__(self, **inputs):
        self.masked = inputs["input_ids"].clone()
        batch, length = self.masked.shape
        logits = torch.arange(11, dtype=torch.float32).expand(batch, length, 11).clone()
        return type("Output", (), {"logits": logits})()


def test_mask_position_length_and_exact_alt_minus_ref_arithmetic() -> None:
    adapter = object.__new__(NTv3Model)
    adapter.tokenizer = FakeTokenizer()
    adapter.device = torch.device("cpu")
    adapter.model = FakeMaskedLM()
    adapter.token_ids = {base: EXPECTED_VOCABULARY[base] for base in "ATCGN"}
    adapter.mask_token_id = EXPECTED_VOCABULARY["<mask>"]

    values = adapter._score_batch(
        ["ACGT" * 32, "TGCA" * 32],
        [31, 96],
        ["C", "C"],
        ["T", "G"],
    )

    assert values == [-1.0, 1.0]
    assert adapter.model.masked.shape == (2, 128)
    assert adapter.model.masked[0, 31].item() == EXPECTED_VOCABULARY["<mask>"]
    assert adapter.model.masked[1, 96].item() == EXPECTED_VOCABULARY["<mask>"]


def test_local_code_package_does_not_execute_unpinned_initializer(tmp_path: Path) -> None:
    code = tmp_path / "snapshot"
    code.mkdir()
    (code / "__init__.py").write_text("raise RuntimeError('must not execute')\n")
    (code / "helper.py").write_text("VALUE = 17\n")
    package = _load_code_package(code)
    assert package.__path__ == [str(code)]

    import importlib

    helper = importlib.import_module("_vep_comparisons_pinned_ntv3.helper")
    assert helper.VALUE == 17


def test_unified_cli_defaults_to_8192_centered_only() -> None:
    args = parse_args(
        [
            "dna",
            "score",
            "--variants",
            "variants.tsv.gz",
            "--reference",
            "reference.fa.gz",
            "--model-dir",
            "weights",
            "--model-code-dir",
            "code",
            "--output-dir",
            "output",
        ]
    )
    assert args.genes is None
    assert args.window_length == 8192
    assert args.min_variant_margin == 1024
