"""Transcript-aware masked-marginal scoring with RiNALMo-giga."""

from __future__ import annotations

import csv
import gzip
import importlib.metadata
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote

from .dna_variant_scoring import InputValidationError, ShardCompatibilityError, ValidationIssue
from .rna_transcript_scoring import (
    ComponentScores,
    InputInventory,
    PreparedRequest,
    RnaCandidate,
    TranscriptAnnotation,
    candidate_sort_key,
    expected_range,
    file_identity,
    normalize_stable_id,
    prepare_requests,
    read_rna_candidates,
    read_transcript_fasta,
    sha256_file,
    shard_candidates,
    shard_paths,
)

MODEL_NAME = "rinalmo-giga"
WEIGHTS_FILENAME = "rinalmo_giga_pretrained.pt"
SCHEMA_VERSION = 1
MAX_MODEL_TOKENS = 1024
CONTEXT_POLICY = "pretraining-1024-boundary-special-tokens-v1"
SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")
TOKENS = (
    "<cls>", "<pad>", "<eos>", "<unk>", "<mask>",
    "A", "C", "G", "T", "I", "R", "Y", "K", "M", "S", "W",
    "B", "D", "H", "V", "N", "-",
)
TOKEN_IDS = {token: index for index, token in enumerate(TOKENS)}
_CHECKPOINT_CACHE: dict[tuple[str, int, int, str, int, int], dict[str, Any]] = {}
OUTPUT_COLUMNS = (
    "variant", "gene", "feature", "chromosome", "start", "end",
    "cdna_position", "ref", "alt", "strand", "transcript_ref",
    "transcript_alt", "model", "ref_log_probability", "alt_log_probability",
    "score", "transcript_window_start", "transcript_window_end",
    "model_token_count",
)


@dataclass(frozen=True)
class TranscriptWindow:
    start: int
    end: int
    target_token_index: int
    token_count: int
    add_cls: bool
    add_eos: bool


@dataclass(frozen=True)
class RiNALMoScore(ComponentScores):
    transcript_window_start: int
    transcript_window_end: int
    model_token_count: int


@dataclass(frozen=True)
class RiNALMoScoringSummary:
    output_path: Path
    metadata_path: Path
    rows: int
    reused: bool = False


class RequestScorer(Protocol):
    def score_requests(
        self, requests: Sequence[PreparedRequest], *, batch_size: int
    ) -> Mapping[tuple[Any, ...], RiNALMoScore]: ...


ModelFactory = Callable[[Path, Path, str, str], RequestScorer]


def select_transcript_window(length: int, target_index: int) -> TranscriptWindow:
    """Select the deterministic, longest <=1024-token interval around a target."""
    if length <= 0 or not 0 <= target_index < length:
        raise ValueError("target_index must identify a nucleotide in a non-empty transcript")
    choices: list[tuple[tuple[int, int, int], TranscriptWindow]] = []
    for start in range(max(0, target_index - MAX_MODEL_TOKENS + 1), target_index + 1):
        add_cls = start == 0
        end = min(length, start + MAX_MODEL_TOKENS - int(add_cls))
        if end == length and end - start + int(add_cls) + 1 > MAX_MODEL_TOKENS:
            end -= 1
        if end <= target_index:
            continue
        add_eos = end == length
        token_count = end - start + int(add_cls) + int(add_eos)
        window = TranscriptWindow(
            start=start,
            end=end,
            target_token_index=int(add_cls) + target_index - start,
            token_count=token_count,
            add_cls=add_cls,
            add_eos=add_eos,
        )
        left = target_index - start
        right = end - 1 - target_index
        # Longest nucleotides, most balanced target, then most upstream.
        choices.append(((-(end - start), abs(left - right), start), window))
    if not choices:
        raise RuntimeError("no valid RiNALMo transcript window")
    return min(choices, key=lambda item: item[0])[1]


def build_masked_tokens(request: PreparedRequest) -> tuple[list[int], TranscriptWindow]:
    annotation = request.annotation
    window = select_transcript_window(len(annotation.sequence), request.candidate.target_index)
    tokens: list[int] = []
    if window.add_cls:
        tokens.append(TOKEN_IDS["<cls>"])
    for base in annotation.sequence[window.start : window.end]:
        tokens.append(TOKEN_IDS.get(base, TOKEN_IDS["<unk>"]))
    if window.add_eos:
        tokens.append(TOKEN_IDS["<eos>"])
    if len(tokens) != window.token_count:
        raise RuntimeError("RiNALMo token count does not match crop provenance")
    expected = TOKEN_IDS.get(request.transcript_ref)
    if expected is None or tokens[window.target_token_index] != expected:
        raise RuntimeError(f"masked-token alignment mismatch for {request.key!r}")
    tokens[window.target_token_index] = TOKEN_IDS["<mask>"]
    return tokens, window


def _gff_attributes(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw.split(";"):
        key, separator, value = item.partition("=")
        if separator:
            result[unquote(key)] = unquote(value)
    return result


def _transcript_id(attributes: Mapping[str, str], record_type: str) -> str | None:
    raw = attributes.get("transcript_id")
    if raw is None and record_type in ("transcript", "mRNA", "lnc_RNA"):
        raw = attributes.get("ID")
    if raw is None:
        raw = attributes.get("Parent")
    if raw is None:
        return None
    raw = raw.split(",", 1)[0]
    if ":" in raw and raw.split(":", 1)[0] in ("transcript", "rna", "mRNA"):
        raw = raw.split(":", 1)[1]
    return normalize_stable_id(raw)


def read_rinalmo_annotations(
    gff3_path: Path, fasta_path: Path, required_transcripts: set[str]
) -> dict[str, TranscriptAnnotation]:
    """Read only transcript/exon records; CDS records are intentionally ignored."""
    records: dict[str, tuple[str, str, str | None]] = {}
    exons: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    issues: list[ValidationIssue] = []
    opener = gzip.open if gff3_path.suffix == ".gz" else open
    with opener(gff3_path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                issues.append(ValidationIssue("invalid_gff3_record", f"line {line_number}"))
                continue
            record_type = fields[2]
            if record_type not in ("transcript", "mRNA", "lnc_RNA", "exon"):
                continue
            attributes = _gff_attributes(fields[8])
            transcript = _transcript_id(attributes, record_type)
            if transcript not in required_transcripts:
                continue
            try:
                start, end = int(fields[3]) - 1, int(fields[4])
            except ValueError:
                issues.append(ValidationIssue("invalid_gff3_coordinate", f"line {line_number}"))
                continue
            chromosome, strand = fields[0], fields[6]
            if start < 0 or end <= start or strand not in ("+", "-"):
                issues.append(ValidationIssue("invalid_gff3_feature", f"line {line_number}"))
                continue
            if record_type == "exon":
                exons[transcript].append((start, end, chromosome, strand))
            else:
                raw_gene = attributes.get("gene_id") or attributes.get("gene") or attributes.get("Parent")
                if raw_gene and ":" in raw_gene:
                    raw_gene = raw_gene.split(":", 1)[1]
                value = (chromosome, strand, None if raw_gene is None else normalize_stable_id(raw_gene))
                if transcript in records and records[transcript] != value:
                    issues.append(ValidationIssue("conflicting_transcript_record", transcript))
                records[transcript] = value
    if issues:
        raise InputValidationError(issues)
    sequences = read_transcript_fasta(fasta_path, required_transcripts)
    missing = sorted(required_transcripts - set(records))
    if missing:
        raise InputValidationError([ValidationIssue("missing_transcript_gff3", item) for item in missing])
    result: dict[str, TranscriptAnnotation] = {}
    for transcript in sorted(required_transcripts):
        chromosome, strand, gene = records[transcript]
        rows = exons[transcript]
        if not rows:
            issues.append(ValidationIssue("missing_exons", transcript))
            continue
        if any(chrom != chromosome or item_strand != strand for _, _, chrom, item_strand in rows):
            issues.append(ValidationIssue("inconsistent_feature_parent", transcript))
            continue
        genomic = sorted((start, end) for start, end, _, _ in rows)
        if any(left[1] > right[0] for left, right in zip(genomic, genomic[1:])):
            issues.append(ValidationIssue("overlapping_exons", transcript))
            continue
        ordered = tuple(genomic if strand == "+" else reversed(genomic))
        sequence = sequences[transcript]
        length = sum(end - start for start, end in ordered)
        if len(sequence) != length:
            issues.append(ValidationIssue("transcript_length_mismatch", f"{transcript}: FASTA={len(sequence)}, exons={length}"))
            continue
        # Empty annotation channels keep the shared validation object model-neutral.
        result[transcript] = TranscriptAnnotation(
            transcript, gene, chromosome, strand, sequence, ordered,
            tuple(0 for _ in sequence), tuple(0 for _ in sequence),
        )
    if issues:
        raise InputValidationError(issues)
    return result


def _resolve_weights(weights: Path) -> Path:
    return weights / WEIGHTS_FILENAME if weights.is_dir() else weights


def validate_rinalmo_checkpoint(weights: Path, manifest_path: Path) -> dict[str, Any]:
    path = _resolve_weights(weights)
    cache_key: tuple[str, int, int, str, int, int] | None = None
    if path.is_file() and manifest_path.is_file():
        weight_stat = path.stat()
        manifest_stat = manifest_path.stat()
        cache_key = (
            str(path.resolve()), weight_stat.st_size, weight_stat.st_mtime_ns,
            str(manifest_path.resolve()), manifest_stat.st_size, manifest_stat.st_mtime_ns,
        )
        cached = _CHECKPOINT_CACHE.get(cache_key)
        if cached is not None:
            return json.loads(json.dumps(cached))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model") != MODEL_NAME:
        raise ValueError(f"checkpoint manifest must identify {MODEL_NAME!r}")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("checkpoint manifest has an invalid source revision")
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("checkpoint manifest has no expected architecture")
    architecture = {"config_name": "giga", "embed_dim": 1280, "num_blocks": 33, "num_heads": 20, "alphabet_size": 22}
    if any(expected.get(key) != value for key, value in architecture.items()):
        raise ValueError("checkpoint manifest does not describe the RiNALMo-giga architecture")
    if expected.get("tokens") != list(TOKENS) or expected.get("token_ids") != TOKEN_IDS:
        raise ValueError("checkpoint manifest alphabet/token IDs mismatch")
    if path.name != WEIGHTS_FILENAME or not path.is_file():
        raise ValueError(f"RiNALMo weights are missing: {path}")
    expected_hash = manifest.get("sha256", {}).get(WEIGHTS_FILENAME)
    observed_hash = sha256_file(path)
    if not isinstance(expected_hash, str) or observed_hash != expected_hash:
        raise ValueError(f"checkpoint hash mismatch: expected {expected_hash}, observed {observed_hash}")
    identity = {
        "model": MODEL_NAME,
        "repository": manifest.get("repository"),
        "revision": revision,
        "sha256": {WEIGHTS_FILENAME: observed_hash},
        "architecture": architecture,
        "tokens": list(TOKENS),
    }
    if cache_key is not None:
        _CHECKPOINT_CACHE[cache_key] = identity
    return json.loads(json.dumps(identity))


def _installed_rinalmo_revision() -> str | None:
    try:
        direct_url = importlib.metadata.distribution("rinalmo").read_text("direct_url.json")
        if not direct_url:
            return None
        return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        return None


class RiNALMoModel:
    def __init__(self, weights: Path, manifest_path: Path, device: str = "cuda", dtype: str = "bfloat16") -> None:
        identity = validate_rinalmo_checkpoint(weights, manifest_path)
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {dtype}")
        import torch
        from rinalmo.config import model_config
        from rinalmo.data.alphabet import Alphabet
        from rinalmo.model.model import RiNALMo

        installed_revision = _installed_rinalmo_revision()
        if installed_revision != identity["revision"]:
            raise ValueError(f"RiNALMo source revision mismatch: expected {identity['revision']}, observed {installed_revision}")
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.dtype_name = dtype
        self.dtype = getattr(torch, dtype)
        config = model_config("giga")
        config.model.token_dropout.active = False
        config.model.transformer.use_flash_attn = self.device.type == "cuda" and dtype in ("bfloat16", "float16")
        alphabet = Alphabet(**config["alphabet"])
        observed_tokens = tuple(alphabet.idx_to_tkn)
        if observed_tokens != TOKENS or {token: alphabet.get_idx(token) for token in TOKENS} != TOKEN_IDS:
            raise ValueError("installed RiNALMo alphabet/token IDs mismatch")
        if (config.globals.embed_dim, config.model.transformer.num_blocks, config.model.transformer.num_heads) != (1280, 33, 20):
            raise ValueError("installed RiNALMo giga configuration mismatch")
        self.model = RiNALMo(config)
        state = torch.load(_resolve_weights(weights), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device, dtype=self.dtype).eval()

    def score_requests(self, requests: Sequence[PreparedRequest], *, batch_size: int) -> Mapping[tuple[Any, ...], RiNALMoScore]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        torch = self._torch
        prepared = [(request, *build_masked_tokens(request)) for request in requests]
        prepared.sort(key=lambda item: (len(item[1]), candidate_sort_key(item[0].candidate)))
        results: dict[tuple[Any, ...], RiNALMoScore] = {}
        for offset in range(0, len(prepared), batch_size):
            batch = prepared[offset : offset + batch_size]
            max_length = max(len(item[1]) for item in batch)
            token_tensor = torch.full((len(batch), max_length), TOKEN_IDS["<pad>"], dtype=torch.long, device=self.device)
            positions = []
            for index, (_, tokens, window) in enumerate(batch):
                token_tensor[index, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self.device)
                positions.append(window.target_token_index)
                if int(token_tensor[index, window.target_token_index].item()) != TOKEN_IDS["<mask>"]:
                    raise RuntimeError("masked token position was corrupted during batching")
            with torch.inference_mode():
                output = self.model(token_tensor)
                logits = output.get("logits") if isinstance(output, Mapping) else None
                if logits is None or tuple(logits.shape) != (len(batch), max_length, len(TOKENS)):
                    shape = None if logits is None else tuple(logits.shape)
                    raise RuntimeError(f"malformed RiNALMo logits shape: {shape}")
                selected = logits[torch.arange(len(batch), device=self.device), torch.tensor(positions, device=self.device)]
                log_probabilities = torch.log_softmax(selected.float(), dim=-1).cpu()
            for index, (request, _, window) in enumerate(batch):
                values = log_probabilities[index]
                score = RiNALMoScore(
                    float(values[TOKEN_IDS[request.transcript_ref]].item()),
                    float(values[TOKEN_IDS[request.transcript_alt]].item()),
                    window.start, window.end, window.token_count,
                )
                if not all(math.isfinite(value) for value in (score.ref_log_probability, score.alt_log_probability, score.score)):
                    raise RuntimeError(f"model returned non-finite scores for {request.key!r}")
                if request.key in results:
                    raise RuntimeError(f"duplicate model score for {request.key!r}")
                results[request.key] = score
        return results


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(variants: Path, fasta: Path, gff3: Path, checkpoint: Mapping[str, Any], num_shards: int, dtype: str) -> dict[str, Any]:
    return {
        "variants": file_identity(variants), "transcript_fasta": file_identity(fasta),
        "gff3": file_identity(gff3), "checkpoint": checkpoint, "num_shards": num_shards,
        "model": MODEL_NAME, "dtype": dtype, "context_policy": CONTEXT_POLICY,
        "max_model_tokens": MAX_MODEL_TOKENS,
    }


def _counts(inventory: InputInventory, selected: Sequence[RnaCandidate]) -> dict[str, int]:
    return {
        "input_rows": inventory.input_rows, "input_unique_variants": inventory.input_unique_variants,
        "eligible_rows": inventory.eligible_rows, "eligible_unique_variants": inventory.eligible_unique_variants,
        "ineligible_rows": inventory.ineligible_rows, "ineligible_unique_variants": inventory.ineligible_unique_variants,
        "output_rows": len(selected), "output_unique_variants": len({item.variant for item in selected}),
    }


def _candidate_key_from_output(row: Mapping[str, str]) -> tuple[Any, ...]:
    try:
        return (row["variant"], normalize_stable_id(row["gene"]), normalize_stable_id(row["feature"]), row["chromosome"], int(row["start"]), int(row["end"]), int(row["cdna_position"]), row["ref"], row["alt"])
    except (KeyError, ValueError) as error:
        raise ShardCompatibilityError(f"invalid output row: {row!r}") from error


def _read_output(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ShardCompatibilityError(f"output schema mismatch: {path}")
        return list(reader)


def _default_model_factory(weights: Path, manifest: Path, device: str, dtype: str) -> RequestScorer:
    return RiNALMoModel(weights, manifest, device, dtype)


def score_rinalmo_variants(*, variants_path: Path, transcript_fasta_path: Path, gff3_path: Path, weights: Path, manifest_path: Path, output_dir: Path, num_shards: int = 1, shard_index: int = 0, device: str = "cuda", dtype: str = "bfloat16", batch_size: int = 8, model_factory: ModelFactory = _default_model_factory) -> RiNALMoScoringSummary:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    checkpoint = validate_rinalmo_checkpoint(weights, manifest_path)
    candidates, inventory = read_rna_candidates(variants_path)
    selected = shard_candidates(candidates, num_shards=num_shards, shard_index=shard_index)
    annotations = read_rinalmo_annotations(gff3_path, transcript_fasta_path, {item.feature for item in selected})
    requests = prepare_requests(selected, annotations)
    identity = _identity(variants_path, transcript_fasta_path, gff3_path, checkpoint, num_shards, dtype)
    output_path, metadata_path = shard_paths(output_dir, num_shards, shard_index)
    shard = {"index": shard_index, "count": num_shards, "expected_genomic_range": expected_range(selected)}
    counts = _counts(inventory, selected)
    if output_path.exists() or metadata_path.exists():
        if not output_path.is_file() or not metadata_path.is_file():
            raise ShardCompatibilityError(f"incomplete existing shard: {output_path}, {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("identity") != identity or metadata.get("shard") != shard or metadata.get("counts") != counts or metadata.get("ineligible_counts") != dict(inventory.ineligible_counts):
            raise ShardCompatibilityError(f"existing shard metadata is incompatible: {metadata_path}")
        if metadata.get("output", {}).get("sha256") != sha256_file(output_path):
            raise ShardCompatibilityError(f"existing shard output checksum mismatch: {output_path}")
        return RiNALMoScoringSummary(output_path, metadata_path, len(selected), True)
    scorer = model_factory(weights, manifest_path, device, dtype)
    scores = scorer.score_requests(requests, batch_size=batch_size)
    expected_keys = {request.key for request in requests}
    if set(scores) != expected_keys:
        raise RuntimeError(f"model score key mismatch: missing={list(expected_keys-set(scores))[:3]!r}, extra={list(set(scores)-expected_keys)[:3]!r}")
    rows = []
    for request in requests:
        candidate, score = request.candidate, scores[request.key]
        rows.append({
            "variant": candidate.variant, "gene": candidate.gene, "feature": candidate.feature,
            "chromosome": candidate.chromosome, "start": candidate.start, "end": candidate.end,
            "cdna_position": candidate.cdna_position, "ref": candidate.ref, "alt": candidate.alt,
            "strand": request.annotation.strand, "transcript_ref": request.transcript_ref,
            "transcript_alt": request.transcript_alt, "model": MODEL_NAME,
            "ref_log_probability": score.ref_log_probability, "alt_log_probability": score.alt_log_probability,
            "score": score.score, "transcript_window_start": score.transcript_window_start,
            "transcript_window_end": score.transcript_window_end, "model_token_count": score.model_token_count,
        })
    metadata = {"schema_version": SCHEMA_VERSION, "identity": identity, "shard": shard, "counts": counts, "ineligible_counts": dict(inventory.ineligible_counts), "output": {"columns": list(OUTPUT_COLUMNS), "rows": len(rows)}}
    _atomic_write_rows(output_path, rows)
    metadata["output"]["sha256"] = sha256_file(output_path)
    _atomic_write_json(metadata_path, metadata)
    return RiNALMoScoringSummary(output_path, metadata_path, len(rows))


def collate_rinalmo_scores(*, variants_path: Path, transcript_fasta_path: Path, gff3_path: Path, weights: Path, manifest_path: Path, output_dir: Path, output_path: Path, num_shards: int, dtype: str = "bfloat16") -> tuple[Path, Path]:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    checkpoint = validate_rinalmo_checkpoint(weights, manifest_path)
    candidates, inventory = read_rna_candidates(variants_path)
    ordered = sorted(candidates, key=candidate_sort_key)
    annotations = read_rinalmo_annotations(gff3_path, transcript_fasta_path, {item.feature for item in ordered})
    prepare_requests(ordered, annotations)
    identity = _identity(variants_path, transcript_fasta_path, gff3_path, checkpoint, num_shards, dtype)
    rows_by_key: dict[tuple[Any, ...], dict[str, str]] = {}
    for index in range(num_shards):
        expected = shard_candidates(ordered, num_shards=num_shards, shard_index=index)
        output, metadata_path = shard_paths(output_dir, num_shards, index)
        if not output.is_file() or not metadata_path.is_file():
            raise ShardCompatibilityError(f"missing shard {index}: {output}")
        metadata = json.loads(metadata_path.read_text())
        expected_shard = {"index": index, "count": num_shards, "expected_genomic_range": expected_range(expected)}
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("identity") != identity or metadata.get("shard") != expected_shard:
            raise ShardCompatibilityError(f"incompatible shard metadata: {metadata_path}")
        if metadata.get("counts") != _counts(inventory, expected) or metadata.get("ineligible_counts") != dict(inventory.ineligible_counts):
            raise ShardCompatibilityError(f"shard count mismatch: {metadata_path}")
        if metadata.get("output", {}).get("sha256") != sha256_file(output):
            raise ShardCompatibilityError(f"shard output checksum mismatch: {output}")
        rows = _read_output(output)
        if len(rows) != len(expected):
            raise ShardCompatibilityError(f"shard row count mismatch: {output}")
        for row in rows:
            key = _candidate_key_from_output(row)
            if key in rows_by_key:
                raise ShardCompatibilityError(f"duplicate row across shards: {key!r}")
            rows_by_key[key] = row
    expected_keys = [item.key for item in ordered]
    missing, extra = [key for key in expected_keys if key not in rows_by_key], list(set(rows_by_key)-set(expected_keys))
    if missing or extra:
        raise ShardCompatibilityError(f"final row coverage mismatch: missing={missing[:3]!r}, extra={extra[:3]!r}")
    final_rows = [rows_by_key[key] for key in expected_keys]
    _atomic_write_rows(output_path, final_rows)
    metadata_path = output_path.with_suffix(".json")
    metadata = {"schema_version": SCHEMA_VERSION, "identity": identity, "counts": {**_counts(inventory, ordered), "shards": num_shards}, "ineligible_counts": dict(inventory.ineligible_counts), "output": {"columns": list(OUTPUT_COLUMNS), "rows": len(final_rows), "sha256": sha256_file(output_path)}}
    _atomic_write_json(metadata_path, metadata)
    return output_path, metadata_path
