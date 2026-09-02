"""Foldseek conversion and validation for SaProt recovery."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .alphafold import AlphaFoldModel


@dataclass(frozen=True)
class Descriptor:
    identifier: str
    sequence: str
    tokens: str


def read_descriptors(path: Path) -> list[Descriptor]:
    descriptors: list[Descriptor] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            identifier, sequence, tokens, *_ = line.rstrip("\n").split("\t")
            descriptors.append(Descriptor(identifier, sequence, tokens))
    return descriptors


def _matches(identifier: str, entry: str) -> bool:
    name = Path(identifier.split(maxsplit=1)[0]).name.removesuffix(".pdb")
    return name == entry or (
        name.startswith(entry)
        and name[len(entry) : len(entry) + 1] in {"_", ".", ":"}
    )


def recover_tokens(
    downloaded: Mapping[str, AlphaFoldModel],
    *,
    pdb_dir: Path,
    descriptor_path: Path,
    foldseek: str,
    threads: int,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    if not downloaded:
        descriptor_path.write_text("", encoding="utf-8")
        return {}
    command = [
        foldseek,
        "structureto3didescriptor",
        "--threads",
        str(threads),
        "--chain-name-mode",
        "1",
        str(pdb_dir),
        str(descriptor_path),
    ]
    result = run_command(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {}

    entries = {model.model_entity_id for model in downloaded.values()}
    by_entry: dict[str, list[Descriptor]] = defaultdict(list)
    for descriptor in read_descriptors(descriptor_path):
        matches = [entry for entry in entries if _matches(descriptor.identifier, entry)]
        if len(matches) != 1:
            return {}
        by_entry[matches[0]].append(descriptor)

    recovered: dict[str, str] = {}
    for transcript, model in downloaded.items():
        descriptors = by_entry[model.model_entity_id]
        if len(descriptors) != 1:
            continue
        descriptor = descriptors[0]
        if descriptor.sequence != model.sequence:
            continue
        if len(descriptor.tokens) != len(descriptor.sequence):
            continue
        recovered[transcript] = descriptor.tokens.lower()
    return recovered
