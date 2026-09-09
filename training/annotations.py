"""Validated, metadata-only supervision records for streamed source files.

Annotation manifests contain coordinates and labels, not source text.  The
source is reacquired through :mod:`training.streaming_sources`, converted from
the public UTF-16 coordinate contract to UTF-8 byte coordinates, used for a
training step, and then released.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from syntaxlm import SourceSpec


ROLE_ALIASES = {
    "name": "identifier",
    "function": "identifier",
    "type": "identifier",
    "builtin": "identifier",
    "string": "string_literal",
    "number": "number_literal",
    "line_comment": "comment",
    "block_comment": "comment",
    "array": "array_literal",
    "object": "object_literal",
}


@dataclass(frozen=True)
class AnnotationRecord:
    spec: SourceSpec
    teacher: dict
    content_sha256: str | None
    role_spans: tuple[dict, ...]
    definitions: tuple[dict, ...]
    usages: tuple[dict, ...]


def _source_spec(item: dict, line_number: int) -> SourceSpec:
    if not item.get("uri"):
        raise ValueError(f"annotation manifest line {line_number} has no uri")
    return SourceSpec(
        uri=str(item["uri"]),
        language=item.get("language"),
        license_id=item.get("licenseId"),
        source_id=item.get("sourceId"),
        repository_commit=item.get("repositoryCommit"),
        relative_path=item.get("relativePath"),
        split=item.get("split", "train"),
        provider_id=item.get("providerId", "annotation-manifest"),
    )


def load_annotations(path: str | Path | None) -> list[AnnotationRecord]:
    """Load and structurally validate annotation metadata.

    This deliberately does not validate coordinates against source until the
    corresponding source is streamed.  That keeps this file source-free and
    allows the same annotation record to be checked against the exact source
    hash at training time.
    """
    if not path:
        return []
    result: list[AnnotationRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid annotation JSON on line {line_number}: {error}") from error
            spec = _source_spec(item, line_number)
            role_spans = tuple(item.get("roleSpans", item.get("roles", ())))
            definitions = tuple(item.get("definitions", ()))
            usages = tuple(item.get("usages", ()))
            if not isinstance(role_spans, tuple) or not all(isinstance(value, dict) for value in role_spans):
                raise ValueError(f"annotation line {line_number} roleSpans must be objects")
            if not all(isinstance(value, dict) for value in definitions + usages):
                raise ValueError(f"annotation line {line_number} definitions/usages must be objects")
            result.append(AnnotationRecord(
                spec=spec,
                teacher=dict(item.get("teacher", {})),
                content_sha256=item.get("contentSha256"),
                role_spans=role_spans,
                definitions=definitions,
                usages=usages,
            ))
    return result


def annotation_index(records: list[AnnotationRecord]) -> dict[str, AnnotationRecord]:
    """Index records by source identity without storing source contents."""
    result: dict[str, AnnotationRecord] = {}
    for record in records:
        for key in (record.spec.uri, record.spec.source_id):
            if key:
                if key in result and result[key] != record:
                    raise ValueError(f"duplicate annotation identity: {key}")
                result[key] = record
    return result


def annotation_for(record_index: dict[str, AnnotationRecord], spec: SourceSpec) -> AnnotationRecord | None:
    return record_index.get(spec.uri) or (record_index.get(spec.source_id) if spec.source_id else None)


def utf16_to_char_offset(text: str, offset: int) -> int:
    """Convert a UTF-16 offset to a Unicode code-point offset.

    An offset in the middle of a surrogate pair is rejected instead of being
    silently shifted, because a shifted training label corrupts editor ranges.
    """
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("annotation offsets must be non-negative integers")
    units = 0
    for index, char in enumerate(text):
        if units == offset:
            return index
        width = 2 if ord(char) > 0xFFFF else 1
        if units < offset < units + width:
            raise ValueError("annotation offset falls inside a UTF-16 surrogate pair")
        units += width
    if units == offset:
        return len(text)
    raise ValueError("annotation offset is outside the source")


def utf16_range_to_bytes(text: str, start: int, end: int) -> tuple[int, int]:
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise ValueError("annotation range must have integer start <= end")
    char_start = utf16_to_char_offset(text, start)
    char_end = utf16_to_char_offset(text, end)
    boundaries = [0]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode("utf-8")))
    return boundaries[char_start], boundaries[char_end]


def range_from(item: dict, field: str = "range") -> tuple[int, int]:
    value = item.get(field, item)
    if isinstance(value, dict):
        return int(value["start"]), int(value["end"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"annotation requires {field}={{start,end}} or [start,end]")


def canonical_role(value: str) -> str:
    return ROLE_ALIASES.get(value, value)
