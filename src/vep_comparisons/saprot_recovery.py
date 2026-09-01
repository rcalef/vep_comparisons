"""Conservative recovery of SaProt 3Di tokens from AlphaFoldDB isoforms."""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import json
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .variant_scoring import (
    read_variant_candidates,
    validate_candidates,
)

AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{accession}.json"
USER_AGENT = "vep-comparisons-saprot-recovery/0.1"
PLDDT_MASK_POLICY = "none (match existing unmasked Foldseek artifact)"

MANIFEST_COLUMNS = (
    "transcript",
    "transcript_version",
    "uniprot_base",
    "alphafold_entry",
    "status",
    "reason",
    "gencode_length",
    "alphafold_length",
    "existing_foldseek_length",
    "gencode_md5",
    "alphafold_sequence_checksum",
    "model_version",
    "model_created_date",
    "pdb_url",
    "pdb_sha256",
    "global_metric_value",
    "plddt_mask_policy",
    "variant_count",
    "variant_positions",
)

INVENTORY_COLUMNS = (
    "transcript",
    "transcript_version",
    "uniprot_base",
    "gencode_sequence",
    "gencode_length",
    "gencode_md5",
    "existing_foldseek_length",
    "variant_count",
    "variant_positions",
)


class RecoveryError(RuntimeError):
    """A malformed input or collection-level invariant prevented recovery."""


class RequestFailed(RecoveryError):
    """An HTTP request still failed after retrying."""


@dataclass(frozen=True)
class Translation:
    transcript: str
    transcript_version: str
    sequence: str


@dataclass(frozen=True)
class MismatchCandidate:
    transcript: str
    transcript_version: str
    uniprot_base: str
    sequence: str
    existing_foldseek_length: int | None
    variant_count: int | None
    variant_positions: str

    @property
    def md5(self) -> str:
        return hashlib.md5(self.sequence.encode()).hexdigest()

    @property
    def uniprot_accessions(self) -> tuple[str, ...]:
        return tuple(item for item in self.uniprot_base.split(";") if item)


@dataclass(frozen=True)
class AlphaFoldModel:
    model_entity_id: str
    uniprot_accession: str
    sequence: str
    sequence_checksum: str
    sequence_start: int | None
    sequence_end: int | None
    entity_type: str
    is_complex: bool | None
    latest_version: str
    model_created_date: str
    pdb_url: str
    global_metric_value: str

    @classmethod
    def from_api(cls, value: Mapping[str, Any]) -> AlphaFoldModel:
        return cls(
            model_entity_id=str(value.get("modelEntityId") or ""),
            uniprot_accession=str(value.get("uniprotAccession") or ""),
            sequence=str(value.get("sequence") or ""),
            sequence_checksum=str(
                value.get("sequenceChecksum")
                or value.get("uniprotSequenceChecksum")
                or ""
            ),
            sequence_start=_optional_int(value.get("sequenceStart")),
            sequence_end=_optional_int(value.get("sequenceEnd")),
            entity_type=str(value.get("entityType") or ""),
            is_complex=(
                value.get("isComplex")
                if isinstance(value.get("isComplex"), bool)
                else None
            ),
            latest_version=str(value.get("latestVersion") or ""),
            model_created_date=str(value.get("modelCreatedDate") or ""),
            pdb_url=str(value.get("pdbUrl") or ""),
            global_metric_value=str(value.get("globalMetricValue") or ""),
        )


@dataclass(frozen=True)
class Descriptor:
    identifier: str
    sequence: str
    tokens: str


@dataclass(frozen=True)
class RecoverySummary:
    candidates: int
    recovered: int
    unresolved: int
    recovered_fasta: Path
    manifest: Path
    merged_fasta: Path


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_transcript(value: str) -> str:
    fields = value.split(maxsplit=1)[0].split("|")
    identifier = next((field for field in fields if field.startswith("ENST")), fields[0])
    return identifier.split(".", maxsplit=1)[0]


def versioned_transcript(value: str) -> str:
    fields = value.split(maxsplit=1)[0].split("|")
    return next((field for field in fields if field.startswith("ENST")), fields[0])


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    if path.suffix == ".bz2":
        return bz2.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_fasta(path: Path) -> dict[str, Translation]:
    """Read a FASTA while retaining the versioned ENST identifier."""

    records: dict[str, Translation] = {}
    header: str | None = None
    chunks: list[str] = []

    def commit() -> None:
        if header is None:
            return
        versioned = versioned_transcript(header)
        transcript = normalize_transcript(versioned)
        sequence = "".join(chunks)
        if transcript in records:
            raise RecoveryError(f"Duplicate FASTA transcript: {transcript}")
        records[transcript] = Translation(transcript, versioned, sequence)

    with _open_text(path, "rt") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                header = line[1:]
                chunks = []
            elif header is None:
                raise RecoveryError(f"Sequence before first FASTA header in {path}")
            else:
                chunks.append(line)
    commit()
    return records


def _field(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _parse_optional_int(value: str, *, field: str, transcript: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise RecoveryError(
            f"Invalid {field} for {transcript}: {value!r}"
        ) from error


def read_mismatch_candidates(
    path: Path, translations_path: Path
) -> list[MismatchCandidate]:
    """Read and enrich the notebook's mismatch table, once per transcript."""

    translations = read_fasta(translations_path)
    candidates: dict[str, MismatchCandidate] = {}
    with _open_text(path, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RecoveryError(f"Mismatch table has no header: {path}")
        for row in reader:
            raw_transcript = _field(row, "transcript", "feature", "transcript_version")
            if not raw_transcript:
                raise RecoveryError("Mismatch row is missing a transcript")
            transcript = normalize_transcript(raw_transcript)
            translation = translations.get(transcript)
            if translation is None:
                raise RecoveryError(f"Missing GENCODE translation for {transcript}")
            supplied_sequence = _field(row, "gencode_sequence", "sequence")
            if supplied_sequence and supplied_sequence != translation.sequence:
                raise RecoveryError(
                    f"Mismatch inventory sequence disagrees with FASTA for {transcript}"
                )
            supplied_md5 = _field(row, "gencode_md5", "sequence_md5")
            actual_md5 = hashlib.md5(translation.sequence.encode()).hexdigest()
            if supplied_md5 and supplied_md5.lower() != actual_md5:
                raise RecoveryError(
                    f"Mismatch inventory MD5 disagrees with FASTA for {transcript}"
                )
            supplied_length = _field(row, "gencode_length", "sequence_length")
            if supplied_length and _parse_optional_int(
                supplied_length, field="gencode_length", transcript=transcript
            ) != len(translation.sequence):
                raise RecoveryError(
                    f"Mismatch inventory length disagrees with FASTA for {transcript}"
                )
            accession_field = _field(row, "uniprot_base", "uniprot_id")
            if not accession_field:
                raise RecoveryError(f"Mismatch row is missing UniProt ID for {transcript}")
            base_accessions: set[str] = set()
            for accession in re.split(r"[;,]", accession_field):
                accession = accession.strip()
                if not accession:
                    continue
                base_accessions.add(
                    accession.rsplit("-", maxsplit=1)[0]
                    if accession.rsplit("-", 1)[-1].isdigit()
                    else accession
                )
            if not base_accessions:
                raise RecoveryError(
                    f"Mismatch row has no usable UniProt ID for {transcript}"
                )
            base_accession = ";".join(sorted(base_accessions))
            candidate = MismatchCandidate(
                transcript=transcript,
                transcript_version=_field(row, "transcript_version")
                or translation.transcript_version,
                uniprot_base=base_accession,
                sequence=translation.sequence,
                existing_foldseek_length=_parse_optional_int(
                    _field(row, "existing_foldseek_length", "structure_token_length"),
                    field="existing_foldseek_length",
                    transcript=transcript,
                ),
                variant_count=_parse_optional_int(
                    _field(row, "variant_count", "n_variants"),
                    field="variant_count",
                    transcript=transcript,
                ),
                variant_positions=_field(row, "variant_positions", "protein_positions"),
            )
            previous = candidates.get(transcript)
            if previous is not None:
                comparable_previous = (
                    previous.transcript_version,
                    previous.sequence,
                    previous.variant_count,
                    previous.variant_positions,
                )
                comparable_candidate = (
                    candidate.transcript_version,
                    candidate.sequence,
                    candidate.variant_count,
                    candidate.variant_positions,
                )
                if comparable_previous != comparable_candidate:
                    raise RecoveryError(
                        f"Conflicting mismatch rows for transcript {transcript}"
                    )
                candidate = MismatchCandidate(
                    transcript=candidate.transcript,
                    transcript_version=candidate.transcript_version,
                    uniprot_base=";".join(
                        sorted(
                            set(previous.uniprot_accessions)
                            | set(candidate.uniprot_accessions)
                        )
                    ),
                    sequence=candidate.sequence,
                    existing_foldseek_length=(
                        candidate.existing_foldseek_length
                        if candidate.existing_foldseek_length
                        == previous.existing_foldseek_length
                        else None
                    ),
                    variant_count=candidate.variant_count,
                    variant_positions=candidate.variant_positions,
                )
            candidates[transcript] = candidate
    return sorted(candidates.values(), key=lambda item: item.transcript)


def write_inventory(path: Path, candidates: Sequence[MismatchCandidate]) -> None:
    rows = [
        {
            "transcript": item.transcript,
            "transcript_version": item.transcript_version,
            "uniprot_base": item.uniprot_base,
            "gencode_sequence": item.sequence,
            "gencode_length": len(item.sequence),
            "gencode_md5": item.md5,
            "existing_foldseek_length": item.existing_foldseek_length,
            "variant_count": item.variant_count,
            "variant_positions": item.variant_positions,
        }
        for item in candidates
    ]
    _atomic_write_tsv(path, INVENTORY_COLUMNS, rows)


def _is_monomeric_protein(model: AlphaFoldModel) -> bool:
    return model.entity_type.lower() == "protein" and model.is_complex is False


def select_exact_model(
    candidate: MismatchCandidate, models: Iterable[AlphaFoldModel]
) -> tuple[AlphaFoldModel | None, str, str]:
    """Return the sole exact, full-length monomer or a conservative outcome."""

    distinct: dict[str, AlphaFoldModel] = {}
    all_models = list(models)
    for model in all_models:
        if not _is_monomeric_protein(model):
            continue
        if (
            model.sequence == candidate.sequence
            and model.sequence_start == 1
            and model.sequence_end == len(candidate.sequence)
            and model.pdb_url
        ):
            key = model.model_entity_id or model.pdb_url
            distinct[key] = model
    matches = list(distinct.values())
    if len(matches) == 1:
        return matches[0], "selected_exact", "unique exact full-length model"
    if len(matches) > 1:
        entries = ",".join(sorted(model.model_entity_id for model in matches))
        return None, "ambiguous_exact_match", f"multiple exact models: {entries}"
    if not all_models:
        return None, "no_alphafold_entry", "AlphaFoldDB returned no models"
    for model in all_models:
        if not _is_monomeric_protein(model):
            continue
        if (
            model.sequence
            and model.sequence in candidate.sequence
            and (
                len(model.sequence) < len(candidate.sequence)
                or model.sequence_start != 1
                or model.sequence_end != len(candidate.sequence)
            )
        ):
            return None, "fragment_only", "only a partial matching model was found"
    return None, "no_exact_sequence_match", "no exact full-length sequence match"


def _isoform_ids(uniprot_record: Mapping[str, Any]) -> list[str]:
    ids: set[str] = set()
    for comment in uniprot_record.get("comments", []):
        if comment.get("commentType") != "ALTERNATIVE PRODUCTS":
            continue
        for isoform in comment.get("isoforms", []):
            for isoform_id in isoform.get("isoformIds", []):
                if isinstance(isoform_id, str):
                    ids.add(isoform_id)
    return sorted(ids)


def _models_from_payload(payload: Any) -> list[AlphaFoldModel]:
    values = payload if isinstance(payload, list) else [payload]
    return [
        AlphaFoldModel.from_api(value)
        for value in values
        if isinstance(value, Mapping) and value.get("modelEntityId")
    ]


def parse_foldseek_descriptors(path: Path) -> list[Descriptor]:
    descriptors: list[Descriptor] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise RecoveryError(
                    f"Malformed Foldseek descriptor line {line_number}: expected 3 columns"
                )
            descriptors.append(Descriptor(fields[0], fields[1], fields[2]))
    return descriptors


def _descriptor_matches_entry(identifier: str, entry: str) -> bool:
    # With chain-name mode 1, Foldseek appends the PDB TITLE after whitespace.
    # Titles can contain '/', so isolate the structure identifier before
    # applying filesystem path handling.
    name = Path(identifier.split(maxsplit=1)[0]).name
    if name.endswith(".pdb"):
        name = name[:-4]
    if name == entry:
        return True
    return name.startswith(entry) and name[len(entry) : len(entry) + 1] in {
        "_",
        ".",
        ":",
    }


def _read_structure_fasta(path: Path) -> dict[str, str]:
    return {key: value.sequence for key, value in read_fasta(path).items()}


def _atomic_write_tsv(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_fasta(path: Path, records: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as raw_handle:
        temporary = Path(raw_handle.name)
    try:
        if path.suffix == ".gz":
            handle: TextIO = gzip.open(temporary, "wt", encoding="utf-8")
        elif path.suffix == ".bz2":
            handle = bz2.open(temporary, "wt", encoding="utf-8")
        else:
            handle = temporary.open("w", encoding="utf-8")
        with handle:
            for transcript in sorted(records):
                sequence = records[transcript]
                handle.write(f">{transcript}\n")
                for start in range(0, len(sequence), 80):
                    handle.write(sequence[start : start + 80] + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class RecoveryRunner:
    """Run all recovery stages while persisting caches and a manifest."""

    def __init__(
        self,
        *,
        output_dir: Path,
        retries: int = 5,
        backoff_seconds: float = 1.0,
        max_requests: int = 4,
        timeout_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.output_dir = output_dir
        self.api_dir = output_dir / "api"
        self.pdb_dir = output_dir / "pdb"
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.max_requests = max_requests
        self.timeout_seconds = timeout_seconds
        self.sleep = sleep

    def _request_bytes(self, url: str, *, not_found_empty: bool = False) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                request = Request(url, headers={"User-Agent": USER_AGENT})
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read()
            except HTTPError as error:
                if error.code == 404 and not_found_empty:
                    return b"[]"
                last_error = error
                if error.code < 500 and error.code != 429:
                    break
            except (URLError, TimeoutError, OSError) as error:
                last_error = error
            if attempt < self.retries:
                self.sleep(self.backoff_seconds * (2**attempt))
        raise RequestFailed(f"request failed for {url}: {last_error}")

    def _cached_json(
        self, cache_path: Path, url: str, *, not_found_empty: bool = False
    ) -> Any:
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RecoveryError(f"Invalid API cache {cache_path}: {error}") from error
        content = self._request_bytes(url, not_found_empty=not_found_empty)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise RequestFailed(f"non-JSON response from {url}") from error
        _atomic_write_json(cache_path, payload)
        return payload

    def _discover_accession(
        self, accession: str, candidates: Sequence[MismatchCandidate]
    ) -> tuple[list[AlphaFoldModel], list[str]]:
        errors: list[str] = []
        try:
            base_payload = self._cached_json(
                self.api_dir / f"{accession}.json",
                AFDB_API.format(accession=accession),
                not_found_empty=True,
            )
        except RequestFailed as error:
            base_payload = []
            errors.append(str(error))
        models = _models_from_payload(base_payload)

        # AlphaFoldDB currently returns isoforms from many base-accession
        # queries. Avoid a second service unless at least one transcript still
        # has no usable exact match in that response.
        needs_isoform_lookup = any(
            select_exact_model(candidate, models)[0] is None
            for candidate in candidates
        )
        if not needs_isoform_lookup:
            return models, errors

        try:
            uniprot_payload = self._cached_json(
                self.api_dir / f"uniprot_{accession}.json",
                UNIPROT_API.format(accession=accession),
            )
            isoforms = [item for item in _isoform_ids(uniprot_payload) if item != accession]
        except RequestFailed as error:
            isoforms = []
            errors.append(str(error))

        known_entries = {model.model_entity_id for model in models}
        for isoform in isoforms:
            try:
                payload = self._cached_json(
                    self.api_dir / f"{isoform}.json",
                    AFDB_API.format(accession=isoform),
                    not_found_empty=True,
                )
            except RequestFailed as error:
                errors.append(str(error))
                continue
            for model in _models_from_payload(payload):
                if model.model_entity_id not in known_entries:
                    models.append(model)
                    known_entries.add(model.model_entity_id)
        return models, errors

    def discover(
        self, candidates: Sequence[MismatchCandidate]
    ) -> tuple[dict[str, AlphaFoldModel], dict[str, dict[str, Any]]]:
        by_accession: dict[str, list[MismatchCandidate]] = defaultdict(list)
        for candidate in candidates:
            for accession in candidate.uniprot_accessions:
                by_accession[accession].append(candidate)

        discovered: dict[str, tuple[list[AlphaFoldModel], list[str]]] = {}
        with ThreadPoolExecutor(max_workers=self.max_requests) as executor:
            futures = {
                executor.submit(
                    self._discover_accession, accession, accession_candidates
                ): accession
                for accession, accession_candidates in by_accession.items()
            }
            for future in as_completed(futures):
                accession = futures[future]
                try:
                    discovered[accession] = future.result()
                except Exception as error:  # retain the failed accession in the manifest
                    discovered[accession] = ([], [str(error)])

        selected: dict[str, AlphaFoldModel] = {}
        manifest: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            models: list[AlphaFoldModel] = []
            errors: list[str] = []
            seen_models: set[str] = set()
            for accession in candidate.uniprot_accessions:
                accession_models, accession_errors = discovered[accession]
                errors.extend(accession_errors)
                for model in accession_models:
                    key = model.model_entity_id or model.pdb_url
                    if key not in seen_models:
                        models.append(model)
                        seen_models.add(key)
            model, status, reason = select_exact_model(candidate, models)
            if errors:
                model = None
                status = "discovery_failed"
                reason = "; ".join(errors)
            if model is not None:
                selected[candidate.transcript] = model
            manifest[candidate.transcript] = self._manifest_row(
                candidate, model, status, reason
            )
        return selected, manifest

    @staticmethod
    def _manifest_row(
        candidate: MismatchCandidate,
        model: AlphaFoldModel | None,
        status: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "transcript": candidate.transcript,
            "transcript_version": candidate.transcript_version,
            "uniprot_base": candidate.uniprot_base,
            "alphafold_entry": model.model_entity_id if model else "",
            "status": status,
            "reason": reason,
            "gencode_length": len(candidate.sequence),
            "alphafold_length": len(model.sequence) if model else "",
            "existing_foldseek_length": candidate.existing_foldseek_length,
            "gencode_md5": candidate.md5,
            "alphafold_sequence_checksum": model.sequence_checksum if model else "",
            "model_version": model.latest_version if model else "",
            "model_created_date": model.model_created_date if model else "",
            "pdb_url": model.pdb_url if model else "",
            "pdb_sha256": "",
            "global_metric_value": model.global_metric_value if model else "",
            "plddt_mask_policy": PLDDT_MASK_POLICY,
            "variant_count": candidate.variant_count,
            "variant_positions": candidate.variant_positions,
        }

    def _download_pdb(self, model: AlphaFoldModel) -> tuple[Path, str]:
        destination = self.pdb_dir / f"{model.model_entity_id}.pdb"
        if destination.exists():
            content = destination.read_bytes()
            if b"\nATOM " in content or content.startswith(b"ATOM "):
                return destination, hashlib.sha256(content).hexdigest()
            self._quarantine_pdb(destination)
        content = self._request_bytes(model.pdb_url)
        if not content or not (b"\nATOM " in content or content.startswith(b"ATOM ")):
            raise RequestFailed(f"downloaded PDB is empty or incomplete: {model.pdb_url}")
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("wb", dir=self.pdb_dir, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(destination)
        return destination, hashlib.sha256(content).hexdigest()

    def _quarantine_pdb(self, path: Path) -> Path:
        quarantine_dir = self.output_dir / "pdb_unaccepted"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        destination = quarantine_dir / path.name
        counter = 1
        while destination.exists():
            destination = quarantine_dir / f"{path.stem}.{counter}{path.suffix}"
            counter += 1
        path.replace(destination)
        return destination

    def download(
        self,
        selected: Mapping[str, AlphaFoldModel],
        manifest: dict[str, dict[str, Any]],
    ) -> dict[str, AlphaFoldModel]:
        by_entry: dict[str, AlphaFoldModel] = {
            model.model_entity_id: model for model in selected.values()
        }
        outcomes: dict[str, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=self.max_requests) as executor:
            futures = {
                executor.submit(self._download_pdb, model): entry
                for entry, model in by_entry.items()
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    _, checksum = future.result()
                    outcomes[entry] = ("", checksum)
                except Exception as error:
                    outcomes[entry] = (str(error), "")

        downloaded: dict[str, AlphaFoldModel] = {}
        for transcript, model in selected.items():
            error, checksum = outcomes[model.model_entity_id]
            if error:
                manifest[transcript]["status"] = "download_failed"
                manifest[transcript]["reason"] = error
            else:
                downloaded[transcript] = model
                manifest[transcript]["pdb_sha256"] = checksum

        expected = {f"{model.model_entity_id}.pdb" for model in downloaded.values()}
        actual = (
            {path.name for path in self.pdb_dir.iterdir() if path.is_file()}
            if self.pdb_dir.exists()
            else set()
        )
        unexpected = sorted(actual - expected)
        if unexpected:
            for name in unexpected:
                self._quarantine_pdb(self.pdb_dir / name)
        remaining = (
            {path.name for path in self.pdb_dir.iterdir() if path.is_file()}
            if self.pdb_dir.exists()
            else set()
        )
        if remaining != expected:
            raise RecoveryError("Could not isolate the accepted recovery PDB files")
        return downloaded

    def run_foldseek(
        self,
        downloaded: Mapping[str, AlphaFoldModel],
        manifest: dict[str, dict[str, Any]],
        *,
        foldseek: str,
        threads: int,
        run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> dict[str, str]:
        descriptor_path = self.output_dir / "foldseek_descriptors.tsv"
        if not downloaded:
            descriptor_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor_path.write_text("", encoding="utf-8")
            return {}
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix="foldseek_descriptors.",
            suffix=".tsv",
            dir=descriptor_path.parent,
            delete=False,
        ) as handle:
            pending_descriptor_path = Path(handle.name)
        command = [
            foldseek,
            "structureto3didescriptor",
            "--threads",
            str(threads),
            "--chain-name-mode",
            "1",
            str(self.pdb_dir),
            str(pending_descriptor_path),
        ]
        try:
            result = run_command(command, check=False, capture_output=True, text=True)
        except OSError as error:
            result = subprocess.CompletedProcess(command, 127, "", str(error))
        if result.returncode != 0:
            pending_descriptor_path.unlink(missing_ok=True)
            reason = f"Foldseek exited {result.returncode}: {result.stderr.strip()}"
            for transcript in downloaded:
                manifest[transcript]["status"] = "foldseek_failed"
                manifest[transcript]["reason"] = reason
            return {}
        if not pending_descriptor_path.exists():
            for transcript in downloaded:
                manifest[transcript]["status"] = "foldseek_missing_descriptor"
                manifest[transcript]["reason"] = "Foldseek did not create its output"
            return {}
        pending_descriptor_path.replace(descriptor_path)

        try:
            descriptors = parse_foldseek_descriptors(descriptor_path)
        except RecoveryError as error:
            for transcript in downloaded:
                manifest[transcript]["status"] = "foldseek_failed"
                manifest[transcript]["reason"] = str(error)
            return {}

        descriptors_by_entry: dict[str, list[Descriptor]] = defaultdict(list)
        for descriptor in descriptors:
            matches = {
                model.model_entity_id
                for model in downloaded.values()
                if _descriptor_matches_entry(descriptor.identifier, model.model_entity_id)
            }
            if len(matches) != 1:
                reason = (
                    f"descriptor {descriptor.identifier!r} maps to {len(matches)} models"
                )
                for transcript in downloaded:
                    manifest[transcript]["status"] = "foldseek_failed"
                    manifest[transcript]["reason"] = reason
                return {}
            descriptors_by_entry[matches.pop()].append(descriptor)

        recovered: dict[str, str] = {}
        for transcript, model in downloaded.items():
            matching = descriptors_by_entry[model.model_entity_id]
            if not matching:
                manifest[transcript]["status"] = "foldseek_missing_descriptor"
                manifest[transcript]["reason"] = "no descriptor for downloaded model"
                continue
            if len(matching) != 1:
                manifest[transcript]["status"] = "foldseek_multiple_descriptors"
                manifest[transcript]["reason"] = (
                    f"downloaded model produced {len(matching)} descriptors"
                )
                continue
            descriptor = matching[0]
            if descriptor.sequence != model.sequence:
                manifest[transcript]["status"] = "foldseek_sequence_mismatch"
                manifest[transcript]["reason"] = (
                    "Foldseek AA sequence differs from AlphaFoldDB/GENCODE"
                )
                continue
            if len(descriptor.sequence) != len(descriptor.tokens):
                manifest[transcript]["status"] = "foldseek_sequence_mismatch"
                manifest[transcript]["reason"] = (
                    f"Foldseek AA={len(descriptor.sequence)}, "
                    f"3Di={len(descriptor.tokens)}"
                )
                continue
            recovered[transcript] = descriptor.tokens.lower()
            manifest[transcript]["status"] = "recovered_exact"
            manifest[transcript]["reason"] = "all sequence and descriptor checks passed"
        return recovered

    def write_manifest(self, manifest: Mapping[str, Mapping[str, Any]]) -> Path:
        path = self.output_dir / "recovery_manifest.tsv"
        _atomic_write_tsv(
            path,
            MANIFEST_COLUMNS,
            (manifest[key] for key in sorted(manifest)),
        )
        return path


def merge_tokens(
    *,
    existing_path: Path,
    recovered: Mapping[str, str],
    translations_path: Path,
    output_path: Path,
) -> dict[str, str]:
    """Validate and atomically write existing + recovered transcript tokens."""

    existing = _read_structure_fasta(existing_path)
    translations = read_fasta(translations_path)
    overlap = sorted(set(existing) & set(recovered))
    if overlap:
        raise RecoveryError(
            "Recovery would replace existing transcript(s): " + ", ".join(overlap)
        )
    merged = {**existing, **recovered}
    issues: list[str] = []
    for transcript, tokens in merged.items():
        translation = translations.get(transcript)
        if translation is None:
            issues.append(f"{transcript}: missing translation")
        elif len(tokens) != len(translation.sequence):
            issues.append(
                f"{transcript}: translation={len(translation.sequence)}, 3Di={len(tokens)}"
            )
    if issues:
        raise RecoveryError("Invalid merged token collection: " + "; ".join(issues[:10]))
    _write_fasta(output_path, merged)
    return merged


def recover_saprot_structures(
    *,
    mismatches_path: Path,
    translations_path: Path,
    existing_tokens_path: Path,
    output_dir: Path,
    foldseek: str = "foldseek",
    threads: int = 1,
    merged_output: Path | None = None,
    variants_path: Path | None = None,
    retries: int = 5,
    backoff_seconds: float = 1.0,
    max_requests: int = 4,
    timeout_seconds: float = 60.0,
) -> RecoverySummary:
    """Execute the exact-match recovery workflow and return artifact paths."""

    if threads < 1 or max_requests < 1 or retries < 0:
        raise RecoveryError("threads/max_requests must be positive and retries nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_mismatch_candidates(mismatches_path, translations_path)
    write_inventory(output_dir / "mismatch_candidates.tsv", candidates)
    runner = RecoveryRunner(
        output_dir=output_dir,
        retries=retries,
        backoff_seconds=backoff_seconds,
        max_requests=max_requests,
        timeout_seconds=timeout_seconds,
    )
    selected, manifest = runner.discover(candidates)
    runner.write_manifest(manifest)
    try:
        downloaded = runner.download(selected, manifest)
    except RecoveryError as error:
        for transcript in selected:
            if manifest[transcript]["status"] == "selected_exact":
                manifest[transcript]["status"] = "foldseek_failed"
                manifest[transcript]["reason"] = str(error)
        runner.write_manifest(manifest)
        raise
    runner.write_manifest(manifest)
    recovered = runner.run_foldseek(
        downloaded, manifest, foldseek=foldseek, threads=threads
    )
    runner.write_manifest(manifest)

    if variants_path is not None and recovered:
        candidates_to_validate = [
            candidate
            for candidate in read_variant_candidates(variants_path)
            if candidate.transcript in recovered
        ]
        sequences = {
            key: value.sequence for key, value in read_fasta(translations_path).items()
        }
        issues = validate_candidates(
            candidates_to_validate,
            sequences,
            structure_tokens=recovered,
            require_structure=True,
        )
        structure_issues = [
            issue
            for issue in issues
            if issue.category
            in {"missing_structure_tokens", "sequence_structure_length_mismatch"}
        ]
        for issue in structure_issues:
            transcript = issue.detail.split(":", maxsplit=1)[0]
            recovered.pop(transcript, None)
            manifest[transcript]["status"] = "scoring_validation_failed"
            manifest[transcript]["reason"] = (
                f"{issue.category}: {issue.detail}"
            )

    runner.write_manifest(manifest)

    recovered_path = output_dir / "recovered_foldseek_tokens.fa.bz2"
    _write_fasta(recovered_path, recovered)
    merged_path = merged_output or output_dir / "merged_foldseek_tokens.fa.bz2"
    merge_tokens(
        existing_path=existing_tokens_path,
        recovered=recovered,
        translations_path=translations_path,
        output_path=merged_path,
    )

    manifest_path = runner.write_manifest(manifest)
    recovered_count = sum(
        row["status"] == "recovered_exact" for row in manifest.values()
    )
    return RecoverySummary(
        candidates=len(candidates),
        recovered=recovered_count,
        unresolved=len(candidates) - recovered_count,
        recovered_fasta=recovered_path,
        manifest=manifest_path,
        merged_fasta=merged_path,
    )
