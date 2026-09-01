"""Strictly offline adapter for a pinned NTv3 weights/code pair."""

from __future__ import annotations

import importlib
import json
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .dna_variant_scoring import (
    DnaModel,
    OrientationScores,
    ReferenceGenome,
    ScoringContext,
    complemented,
    extract_window,
    reverse_complement,
    sha256_file,
)


EXPECTED_WEIGHT_FILES = ("config.json", "model.safetensors", "tokenizer_config.json", "vocab.json")
EXPECTED_CODE_FILES = (
    "configuration_ntv3_pretrained.py",
    "modeling_ntv3_pretrained.py",
    "tokenization_ntv3.py",
)
EXPECTED_VOCABULARY = {
    "<unk>": 0,
    "<pad>": 1,
    "<mask>": 2,
    "<cls>": 3,
    "<eos>": 4,
    "<bos>": 5,
    "A": 6,
    "T": 7,
    "C": 8,
    "G": 9,
    "N": 10,
}


class ModelPackageError(ValueError):
    pass


def _validated_file_hashes(
    directory: Path, expected_names: Sequence[str], manifest_hashes: Mapping[str, str], label: str
) -> dict[str, str]:
    missing_manifest = sorted(set(expected_names) - set(manifest_hashes))
    if missing_manifest:
        raise ModelPackageError(f"model manifest is missing {label} hashes: {missing_manifest}")
    hashes: dict[str, str] = {}
    for name in expected_names:
        path = directory / name
        if not path.is_file():
            raise ModelPackageError(f"missing {label} file: {path}")
        if manifest_hashes[name] == "GATED_SNAPSHOT_REQUIRED":
            raise ModelPackageError(
                "the checked-in manifest still needs hashes from the gated "
                f"NTv3 code snapshot ({name})"
            )
        hashes[name] = sha256_file(path)
        if hashes[name] != manifest_hashes[name]:
            raise ModelPackageError(
                f"{label} hash mismatch for {path}: expected {manifest_hashes[name]}, got {hashes[name]}"
            )
    return hashes


def validate_model_package(
    model_dir: Path, model_code_dir: Path, manifest_path: Path
) -> dict[str, Any]:
    """Verify revisions, every executable source file, config, and token IDs."""

    if not manifest_path.is_file():
        raise ModelPackageError(f"model manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    try:
        weights = manifest["weights"]
        code = manifest["code"]
        expected = manifest["expected"]
    except KeyError as error:
        raise ModelPackageError(f"model manifest is missing {error.args[0]!r}") from error
    for label, section in (("weights", weights), ("code", code)):
        revision = section.get("revision")
        if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ModelPackageError(f"{label} revision must be a pinned 40-character lowercase git SHA")
    weight_hashes = _validated_file_hashes(
        model_dir, EXPECTED_WEIGHT_FILES, weights.get("sha256", {}), "weight"
    )
    code_hashes = _validated_file_hashes(
        model_code_dir, EXPECTED_CODE_FILES, code.get("sha256", {}), "code"
    )
    if not (model_code_dir / "__init__.py").is_file():
        raise ModelPackageError(f"missing code package initializer: {model_code_dir / '__init__.py'}")

    config = json.loads((model_dir / "config.json").read_text())
    vocabulary = json.loads((model_dir / "vocab.json").read_text())
    checks = {
        "architecture": config.get("architectures"),
        "alphabet_size": config.get("alphabet_size"),
        "num_downsamples": config.get("num_downsamples"),
        "mask_token_id": config.get("mask_token_id"),
        "pad_token_id": config.get("pad_token_id"),
        "vocabulary": vocabulary,
    }
    wanted = {
        "architecture": ["NTv3PreTrained"],
        "alphabet_size": 11,
        "num_downsamples": 7,
        "mask_token_id": EXPECTED_VOCABULARY["<mask>"],
        "pad_token_id": EXPECTED_VOCABULARY["<pad>"],
        "vocabulary": EXPECTED_VOCABULARY,
    }
    manifest_wanted = {
        "alphabet_size": expected.get("alphabet_size"),
        "num_downsamples": expected.get("num_downsamples"),
        "token_ids": expected.get("token_ids"),
    }
    if checks != wanted:
        raise ModelPackageError(f"NTv3 config/vocabulary mismatch: observed={checks!r}")
    if manifest_wanted != {
        "alphabet_size": 11,
        "num_downsamples": 7,
        "token_ids": EXPECTED_VOCABULARY,
    }:
        raise ModelPackageError("model manifest expected-token/config section is invalid")
    return {
        "model": manifest.get("model"),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "weights": {
            "path": str(model_dir.resolve()),
            "repository": weights.get("repository"),
            "revision": weights["revision"],
            "sha256": weight_hashes,
        },
        "code": {
            "path": str(model_code_dir.resolve()),
            "repository": code.get("repository"),
            "revision": code["revision"],
            "sha256": code_hashes,
        },
        "token_ids": {key: EXPECTED_VOCABULARY[key] for key in "ATCGN"},
    }


def _load_code_package(code_dir: Path):
    """Create an isolated package without executing unpinned ``__init__.py``."""

    package_name = "_vep_comparisons_pinned_ntv3"
    existing = sys.modules.get(package_name)
    if existing is not None:
        if Path(existing.__file__).resolve().parent != code_dir.resolve():
            raise ModelPackageError("a different NTv3 code snapshot is already loaded")
        return existing
    package = types.ModuleType(package_name)
    package.__file__ = str(code_dir / "__init__.py")
    package.__package__ = package_name
    package.__path__ = [str(code_dir)]
    sys.modules[package_name] = package
    return package


def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


class NTv3Model(DnaModel):
    """Masked-marginal NTv3 scorer using only validated local files."""

    def __init__(
        self,
        model_dir: Path,
        model_code_dir: Path,
        manifest_path: Path,
        *,
        device: str,
        dtype: str,
    ) -> None:
        import torch

        validate_model_package(model_dir, model_code_dir, manifest_path)
        _load_code_package(model_code_dir)
        package_name = "_vep_comparisons_pinned_ntv3"
        configuration = importlib.import_module(f"{package_name}.configuration_ntv3_pretrained")
        tokenization = importlib.import_module(f"{package_name}.tokenization_ntv3")
        modeling = importlib.import_module(f"{package_name}.modeling_ntv3_pretrained")
        config_class = getattr(configuration, "Ntv3PreTrainedConfig")
        tokenizer_class = getattr(tokenization, "NTv3Tokenizer")
        model_class = getattr(modeling, "NTv3PreTrained")

        config = config_class.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = tokenizer_class.from_pretrained(model_dir, local_files_only=True)
        self.device = torch.device(device)
        self.model = model_class.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            torch_dtype=_torch_dtype(dtype),
        ).to(device=self.device).eval()
        vocabulary = self.tokenizer.get_vocab()
        if vocabulary != EXPECTED_VOCABULARY:
            raise ModelPackageError(f"runtime tokenizer vocabulary mismatch: {vocabulary!r}")
        self.token_ids = {base: vocabulary[base] for base in "ATCGN"}
        self.mask_token_id = vocabulary["<mask>"]

    def _score_batch(
        self,
        sequences: Sequence[str],
        targets: Sequence[int],
        refs: Sequence[str],
        alts: Sequence[str],
    ) -> list[float]:
        import torch

        tokenized = self.tokenizer(
            list(sequences),
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        if input_ids.shape != (len(sequences), len(sequences[0])):
            raise RuntimeError(
                f"NTv3 tokenization changed sequence length: input={len(sequences[0])}, tokens={tuple(input_ids.shape)}"
            )
        for row, target in enumerate(targets):
            input_ids[row, target] = self.mask_token_id
        inputs = {name: tensor.to(self.device) for name, tensor in tokenized.items()}
        inputs["input_ids"] = input_ids.to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
        logits = output.logits
        if logits.shape[:2] != input_ids.shape or logits.shape[-1] != 11:
            raise RuntimeError(f"unexpected NTv3 logits shape: {tuple(logits.shape)}")
        result: list[float] = []
        nucleotide_ids = [self.token_ids[base] for base in "ATCG"]
        nucleotide_offsets = {base: index for index, base in enumerate("ATCG")}
        for row, (target, ref, alt) in enumerate(zip(targets, refs, alts, strict=True)):
            # Copy only the four nucleotide values off the accelerator.
            target_logits = logits[row, target, nucleotide_ids].float().cpu()
            result.append(
                float(
                    target_logits[nucleotide_offsets[alt]]
                    - target_logits[nucleotide_offsets[ref]]
                )
            )
        return result

    def score_contexts(
        self,
        contexts: Sequence[ScoringContext],
        *,
        reference: ReferenceGenome,
        batch_size: int,
    ) -> Mapping[tuple[str, int, int, int, str, str], OrientationScores]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        result: dict[tuple[str, int, int, int, str, str], OrientationScores] = {}
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            sequences: list[str] = []
            targets: list[int] = []
            refs: list[str] = []
            alts: list[str] = []
            for context in batch:
                forward = extract_window(reference, context)
                sequences.extend((forward, reverse_complement(forward)))
                targets.extend((context.target_index, context.window_length - 1 - context.target_index))
                refs.extend((context.ref, complemented(context.ref)))
                alts.extend((context.alt, complemented(context.alt)))
            values = self._score_batch(sequences, targets, refs, alts)
            for offset, context in enumerate(batch):
                result[context.key] = OrientationScores(values[2 * offset], values[2 * offset + 1])
        return result
