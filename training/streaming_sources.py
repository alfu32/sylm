"""Bounded, no-retention source acquisition shared by both trainers.

The iterator owns one source buffer at a time. Consumers must finish all
annotation/training work before requesting the next item. The ledger contains
provenance and hashes only; it never receives source text or annotations.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

from syntaxlm import SourceSpec, _decode_source, _language_from_uri, _read_bounded, _source_specs


@dataclass
class StreamedSource:
    spec: SourceSpec
    raw: bytes
    text: str
    encoding: str
    content_sha256: str


class SourceLedger:
    """Append-only metadata ledger with an explicit run identifier."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle: TextIO | None = None
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    def __enter__(self) -> "SourceLedger":
        self.handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def record(self, source: StreamedSource | None, status: str, *,
               spec: SourceSpec | None = None, reason: str | None = None,
               labels: dict | None = None, teacher_versions: dict | None = None) -> None:
        if self.handle is None:
            raise RuntimeError("SourceLedger must be used as a context manager")
        if source is not None:
            spec = source.spec
        if spec is None:
            raise ValueError("record requires a source or spec")
        record = {
            "runId": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "providerId": spec.provider_id,
            "sourceId": spec.source_id,
            "sourceUri": spec.uri,
            "repositoryCommit": spec.repository_commit,
            "relativePath": spec.relative_path,
            "contentSha256": source.content_sha256 if source else None,
            "byteCount": len(source.raw) if source else None,
            "encoding": source.encoding if source else None,
            "licenseId": spec.license_id,
            "canonicalLanguageId": spec.language or _language_from_uri(spec.uri),
            "split": spec.split,
            "teacherVersions": teacher_versions or {},
            "labelCoverage": labels or {},
            "status": status,
            "reason": reason,
        }
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.handle.flush()


def source_specs(manifest: str | None, sources: list[str], default_license: str | None) -> list[SourceSpec]:
    """Load metadata only. A manifest is never interpreted as source code."""
    result = _source_specs(manifest, sources, default_license)
    return [
        replace(spec, language=spec.language or _language_from_uri(spec.uri))
        for spec in result
    ]


def iter_sources(specs: list[SourceSpec], max_bytes: int, ledger: SourceLedger,
                 license_policy: str = "require") -> Iterator[StreamedSource]:
    """Yield accepted files one at a time and discard buffers on the next item.

    Acquisition failures and license skips are recorded and do not stop a
    multilingual run. A consumer exception is deliberately allowed to surface;
    training errors must not be disguised as source failures.
    """
    for spec in specs:
        if license_policy == "require" and not spec.license_id:
            ledger.record(None, "SKIPPED", spec=spec, reason="missing licenseId")
            continue
        source: StreamedSource | None = None
        try:
            raw = _read_bounded(spec.uri, max_bytes)
            text, encoding = _decode_source(raw)
            source = StreamedSource(
                spec=spec,
                raw=raw,
                text=text,
                encoding=encoding,
                content_sha256=hashlib.sha256(raw).hexdigest(),
            )
            yield source
        except (OSError, UnicodeError, ValueError, urllib.error.URLError) as error:
            ledger.record(None, "FAILED", spec=spec,
                          reason=f"{type(error).__name__}: {str(error)[:300]}")
        finally:
            # The generator drops its references after the consumer resumes it.
            del source


def iter_mlcpd_sources(files: list[str], max_bytes: int, ledger: SourceLedger,
                       max_examples: int | None = None) -> Iterator[StreamedSource]:
    """Stream MLCPD Parquet rows without retaining source or dataset shards.

    Hugging Face dataset caches are directed to a temporary cache directory;
    it is removed when this iterator finishes. Source text and parser JSON are
    held only for the current row. The dataset's MIT card/license metadata is
    recorded as provenance, while original-source licensing remains a release
    audit requirement.
    """
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError("MLCPD streaming requires `datasets`; install requirements.txt") from error
    dataset_root = os.environ.get("LOCAL_LANG_MODEL_DATASET_CACHE")
    temporary = None
    if not dataset_root:
        volatile_root = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
        temporary = tempfile.TemporaryDirectory(prefix="local-lang-models-hf-", dir=volatile_root)
        dataset_root = temporary.name
    os.environ.setdefault("HF_HOME", dataset_root)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(dataset_root, "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(dataset_root, "hub"))
    yielded = 0
    try:
        for filename in files:
            dataset = load_dataset(
                "jugalgajjar/MultiLang-Code-Parser-Dataset",
                data_files=filename,
                split="train",
                streaming=True,
            )
            for row_number, row in enumerate(dataset):
                if max_examples is not None and yielded >= max_examples:
                    return
                code = str(row.get("code", ""))
                raw = code.encode("utf-8")
                if not code or len(raw) > max_bytes:
                    continue
                language = str(row.get("language", "unknown")).lower()
                spec = SourceSpec(
                    uri=f"hf://jugalgajjar/MultiLang-Code-Parser-Dataset/{filename}#row={row_number}",
                    language=language,
                    license_id="MLCPD-MIT-card-review-required",
                    source_id=f"mlcpd:{filename}:{row_number}",
                    split="train",
                    provider_id="mlcpd-stream-v1",
                )
                source = StreamedSource(
                    spec=spec,
                    raw=raw,
                    text=code,
                    encoding="utf-8",
                    content_sha256=hashlib.sha256(raw).hexdigest(),
                )
                yield source
                ledger.record(source, "USED", labels={
                    "dataset": "MLCPD", "row": row_number,
                    "parserPayloadPresent": bool(row.get("universal_schema")),
                }, teacher_versions={"parser": "MLCPD-tree-sitter-universal-schema"})
                yielded += 1
                del source, raw, code, row
    finally:
        if temporary is not None:
            temporary.cleanup()
