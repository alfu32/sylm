#!/usr/bin/env python3
"""Generate source-free annotation manifests from streamed teacher results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .streaming_sources import SourceLedger, iter_sources, source_specs
from .teachers import ExternalJsonTeacher, PythonAstTeacher, TeacherContext, validate_annotation_result


def _load_teacher_config(path: str | None) -> dict[str, ExternalJsonTeacher]:
    if not path:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    result = {}
    for language, config in raw.items():
        result[language] = ExternalJsonTeacher(
            command=[str(value) for value in config["command"]],
            name=str(config["name"]),
            version=str(config["version"]),
            timeout_seconds=int(config.get("timeoutSeconds", 30)),
            max_output_bytes=int(config.get("maxOutputBytes", 8 * 1024 * 1024)),
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="stream source through parser/compiler/indexer teachers")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--manifest")
    parser.add_argument("--license-id")
    parser.add_argument("--license-policy", choices=("require", "allow"), default="require")
    parser.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--teacher-config", help="JSON map of language to external stdin/stdout teacher command")
    parser.add_argument("--output", required=True, help="source-free annotation JSONL output")
    parser.add_argument("--ledger", default="teacher-source-use.jsonl")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = source_specs(args.manifest, args.source, args.license_id)
    teachers = _load_teacher_config(args.teacher_config)
    python_teacher = PythonAstTeacher()
    used = 0
    failed = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as annotations, SourceLedger(args.ledger) as ledger:
        for source in iter_sources(specs, args.max_bytes, ledger, args.license_policy):
            language = (source.spec.language or "unknown").lower()
            teacher = python_teacher if language in {"python", "py"} else teachers.get(language)
            if teacher is None:
                ledger.record(source, "SKIPPED", reason="no authoritative teacher configured",
                              labels={"teacher": "missing"})
                failed += 1
                del source
                continue
            try:
                result = teacher.annotate(source.text, TeacherContext(
                    language=language,
                    source_id=source.spec.source_id,
                    uri=source.spec.uri,
                    content_sha256=source.content_sha256,
                ))
                result = validate_annotation_result(result, source.text)
                record = {
                    "uri": source.spec.uri,
                    "language": source.spec.language,
                    "licenseId": source.spec.license_id,
                    "sourceId": source.spec.source_id,
                    "repositoryCommit": source.spec.repository_commit,
                    "relativePath": source.spec.relative_path,
                    "split": source.spec.split,
                    "providerId": source.spec.provider_id,
                    "contentSha256": source.content_sha256,
                    **result,
                }
                annotations.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                annotations.flush()
                ledger.record(source, "ANNOTATED", labels={
                    "roleSpans": len(result.get("roleSpans", [])),
                    "definitions": len(result.get("definitions", [])),
                    "usages": len(result.get("usages", [])),
                    "constructs": len(result.get("constructs", [])),
                    "relations": len(result.get("relations", [])),
                    "teacher": result.get("teacher", {}).get("name"),
                }, teacher_versions={"annotation": result.get("teacher", {})})
                used += 1
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                ledger.record(source, "FAILED", reason=f"{type(error).__name__}: {str(error)[:300]}")
                failed += 1
            finally:
                del source
    print(json.dumps({"annotated": used, "failedOrSkipped": failed,
                      "output": str(output), "ledger": args.ledger}, separators=(",", ":")))
    return 0 if used else 2


if __name__ == "__main__":
    raise SystemExit(main())
