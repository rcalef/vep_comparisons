"""AlphaFoldDB model discovery and download for SaProt recovery."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .inputs import MismatchCandidate

AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{accession}.json"
USER_AGENT = "vep-comparisons-saprot-recovery/0.1"


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


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def select_exact_model(
    candidate: MismatchCandidate,
    models: Iterable[AlphaFoldModel],
) -> AlphaFoldModel | None:
    matches = {
        model.model_entity_id or model.pdb_url: model
        for model in models
        if model.entity_type.lower() == "protein"
        and model.is_complex is False
        and model.sequence == candidate.sequence
        and model.sequence_start == 1
        and model.sequence_end == len(candidate.sequence)
        and model.pdb_url
    }
    return next(iter(matches.values())) if len(matches) == 1 else None


def _isoform_ids(record: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            isoform_id
            for comment in record.get("comments", [])
            if comment.get("commentType") == "ALTERNATIVE PRODUCTS"
            for isoform in comment.get("isoforms", [])
            for isoform_id in isoform.get("isoformIds", [])
            if isinstance(isoform_id, str)
        }
    )


def _models(payload: Any) -> list[AlphaFoldModel]:
    values = payload if isinstance(payload, list) else [payload]
    return [
        AlphaFoldModel.from_api(value)
        for value in values
        if isinstance(value, Mapping) and value.get("modelEntityId")
    ]


class AlphaFoldClient:
    def __init__(
        self,
        *,
        pdb_dir: Path,
        retries: int = 5,
        backoff_seconds: float = 1.0,
        max_requests: int = 4,
        timeout_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.pdb_dir = pdb_dir
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
        raise RuntimeError(f"request failed for {url}: {last_error}")

    def _request_json(self, url: str, *, not_found_empty: bool = False) -> Any:
        return json.loads(self._request_bytes(url, not_found_empty=not_found_empty))

    def _discover_accession(
        self,
        accession: str,
        candidates: Sequence[MismatchCandidate],
    ) -> list[AlphaFoldModel]:
        models = _models(
            self._request_json(
                AFDB_API.format(accession=accession),
                not_found_empty=True,
            )
        )
        if all(select_exact_model(candidate, models) for candidate in candidates):
            return models
        record = self._request_json(UNIPROT_API.format(accession=accession))
        known = {model.model_entity_id for model in models}
        for isoform in _isoform_ids(record):
            if isoform == accession:
                continue
            for model in _models(
                self._request_json(
                    AFDB_API.format(accession=isoform),
                    not_found_empty=True,
                )
            ):
                if model.model_entity_id not in known:
                    models.append(model)
                    known.add(model.model_entity_id)
        return models

    def discover(
        self,
        candidates: Sequence[MismatchCandidate],
    ) -> dict[str, AlphaFoldModel]:
        by_accession: dict[str, list[MismatchCandidate]] = defaultdict(list)
        for candidate in candidates:
            for accession in candidate.uniprot_accessions:
                by_accession[accession].append(candidate)
        discovered: dict[str, list[AlphaFoldModel]] = {}
        with ThreadPoolExecutor(max_workers=self.max_requests) as executor:
            futures = {
                executor.submit(self._discover_accession, accession, members): accession
                for accession, members in by_accession.items()
            }
            for future in as_completed(futures):
                accession = futures[future]
                try:
                    discovered[accession] = future.result()
                except Exception:
                    discovered[accession] = []

        selected: dict[str, AlphaFoldModel] = {}
        for candidate in candidates:
            models = {
                model.model_entity_id: model
                for accession in candidate.uniprot_accessions
                for model in discovered[accession]
            }
            model = select_exact_model(candidate, models.values())
            if model is not None:
                selected[candidate.transcript] = model
        return selected

    def _download(self, model: AlphaFoldModel) -> Path:
        content = self._request_bytes(model.pdb_url)
        if not content or not (content.startswith(b"ATOM ") or b"\nATOM " in content):
            raise ValueError(f"Incomplete PDB from {model.pdb_url}")
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        path = self.pdb_dir / f"{model.model_entity_id}.pdb"
        path.write_bytes(content)
        return path

    def download(
        self,
        selected: Mapping[str, AlphaFoldModel],
    ) -> dict[str, AlphaFoldModel]:
        entries = {model.model_entity_id: model for model in selected.values()}
        downloaded_entries: set[str] = set()
        with ThreadPoolExecutor(max_workers=self.max_requests) as executor:
            futures = {
                executor.submit(self._download, model): entry
                for entry, model in entries.items()
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    future.result()
                    downloaded_entries.add(entry)
                except Exception:
                    pass
        return {
            transcript: model
            for transcript, model in selected.items()
            if model.model_entity_id in downloaded_entries
        }
