"""No-retention parser/compiler/indexer teacher interfaces.

Teachers receive one bounded source in memory and return coordinate metadata.
They must never write source or annotation payloads to disk.  External
teachers communicate through stdin/stdout so Tree-sitter wrappers, compiler
frontends, SCIP indexers, and project-specific analyzers can be plugged in
without making any of them Kotlin runtime dependencies.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from typing import Any, Protocol

from syntaxlm import Token, tokenize_code


@dataclass(frozen=True)
class TeacherContext:
    language: str
    source_id: str | None = None
    uri: str | None = None
    content_sha256: str | None = None


class Teacher(Protocol):
    name: str
    version: str

    def annotate(self, source: str, context: TeacherContext) -> dict[str, Any]:
        ...


def _utf16_offset(source: str, codepoint_offset: int) -> int:
    return len(source[:codepoint_offset].encode("utf-16-le")) // 2


def _utf16_range(source: str, start: int, end: int) -> dict[str, int]:
    return {"start": _utf16_offset(source, start), "end": _utf16_offset(source, end)}


def _line_starts(source: str) -> list[int]:
    return [0] + [match.end() for match in re.finditer("\\n", source)]


def _python_ast_offset(source: str, starts: list[int], line: int, utf8_column: int) -> int:
    """Convert Python AST's UTF-8 byte column to a Python string offset."""
    if line < 1 or line > len(starts):
        raise ValueError("AST line is outside source")
    line_start = starts[line - 1]
    line_text = source[line_start:source.find("\n", line_start) if source.find("\n", line_start) >= 0 else len(source)]
    encoded = line_text.encode("utf-8")
    if utf8_column > len(encoded):
        raise ValueError("AST column is outside source line")
    return line_start + len(encoded[:utf8_column].decode("utf-8", errors="strict"))


def _node_range(source: str, starts: list[int], node: ast.AST) -> tuple[int, int] | None:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return None
    start = _python_ast_offset(source, starts, node.lineno, node.col_offset)
    end = _python_ast_offset(source, starts, node.end_lineno, node.end_col_offset)
    return start, end


def _token_ranges(source: str, language: str) -> list[Token]:
    return tokenize_code(source, language)


def _token_in_range(tokens: list[Token], start: int, end: int, text: str) -> Token | None:
    for token in tokens:
        if start <= token.start and token.end <= end and token.text == text:
            return token
    return None


class PythonAstTeacher:
    """Parser-derived Python teacher with explicit partial semantic status.

    The AST is authoritative for node boundaries and lexical tokens retain
    comments/strings.  Name-to-definition matching is intentionally marked
    partial: production semantic links should come from a compiler/indexer
    teacher, not this single-file fallback.
    """

    name = "python-ast"
    version = "1"

    def annotate(self, source: str, context: TeacherContext) -> dict[str, Any]:
        tree = ast.parse(source, filename=context.source_id or "<source>")
        starts = _line_starts(source)
        tokens = _token_ranges(source, "python")
        role_spans = [
            {**_utf16_range(source, token.start, token.end), "role": {
                "name": "identifier", "function": "identifier", "type": "identifier",
                "builtin": "identifier", "string": "string_literal", "number": "number_literal",
                "comment": "comment", "operator": "operator", "punctuation": "punctuation",
            }.get(token.hint, "unknown")}
            for token in tokens
        ]
        definitions: list[dict[str, Any]] = []
        usages: list[dict[str, Any]] = []
        constructs: list[dict[str, Any]] = []
        definition_names: dict[str, str] = {}
        next_id = 0

        for node in ast.walk(tree):
            node_range = _node_range(source, starts, node)
            if node_range is None:
                continue
            node_start, node_end = node_range
            construct_kind = {
                ast.ClassDef: "class",
                ast.FunctionDef: "function",
                ast.AsyncFunctionDef: "async_function",
                ast.Lambda: "lambda",
                ast.Import: "import",
                ast.ImportFrom: "import_from",
                ast.Try: "exception_handling",
                ast.With: "resource_scope",
                ast.AsyncWith: "async_resource_scope",
                ast.For: "for_loop",
                ast.AsyncFor: "async_for_loop",
                ast.ListComp: "list_comprehension",
                ast.SetComp: "set_comprehension",
                ast.DictComp: "dict_comprehension",
                ast.GeneratorExp: "generator_expression",
            }.get(type(node))
            if construct_kind:
                constructs.append({**_utf16_range(source, node_start, node_end), "kind": construct_kind})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                token = _token_in_range(tokens, node_start, node_end, node.name)
                if token:
                    definition_id = f"d{next_id}"
                    next_id += 1
                    definition_names.setdefault(node.name, definition_id)
                    definitions.append({
                        "id": definition_id,
                        "nameRange": _utf16_range(source, token.start, token.end),
                        "declarationRange": _utf16_range(source, node_start, node_end),
                        "kind": "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "type",
                    })
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                token = _token_in_range(tokens, node_start, node_end, node.id)
                if token:
                    usages.append({
                        **_utf16_range(source, token.start, token.end),
                        "definitionId": definition_names.get(node.id),
                        "kind": "variable",
                        "status": "resolved" if node.id in definition_names else "unresolved",
                    })
        return {
            "roleSpans": role_spans,
            "definitions": definitions,
            "usages": usages,
            "constructs": constructs,
            "relations": [],
            "teacher": {"name": self.name, "version": self.version, "semanticStatus": "partial"},
        }


class ExternalJsonTeacher:
    """Run a configured parser/compiler/indexer over stdin/stdout only."""

    def __init__(self, command: list[str], name: str, version: str, timeout_seconds: int = 30,
                 max_output_bytes: int = 8 * 1024 * 1024) -> None:
        if not command:
            raise ValueError("external teacher command cannot be empty")
        self.command = command
        self.name = name
        self.version = version
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def annotate(self, source: str, context: TeacherContext) -> dict[str, Any]:
        request = json.dumps({
            "language": context.language,
            "sourceId": context.source_id,
            "uri": context.uri,
            "contentSha256": context.content_sha256,
            "source": source,
        }, ensure_ascii=False).encode("utf-8")
        completed = subprocess.run(
            self.command,
            input=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
        )
        if len(completed.stdout) > self.max_output_bytes:
            raise ValueError("external teacher output exceeds configured limit")
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"external teacher failed ({completed.returncode}): {error}")
        result = json.loads(completed.stdout.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("external teacher output must be a JSON object")
        result["teacher"] = {
            **dict(result.get("teacher", {})),
            "name": self.name,
            "version": self.version,
        }
        # A teacher must return coordinates, never source-bearing fields.
        forbidden = {"source", "text", "code", "tokens", "raw"}
        if forbidden.intersection(result):
            raise ValueError("external teacher output contains forbidden source payload")
        return result


def validate_annotation_result(result: dict[str, Any], source: str) -> dict[str, Any]:
    """Validate public UTF-16 ranges while retaining no source in the result."""
    allowed = {"roleSpans", "definitions", "usages", "constructs", "relations", "teacher"}
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"teacher returned unsupported fields: {sorted(unknown)}")
    result = {key: value for key, value in result.items() if key in allowed}
    for collection_name in ("roleSpans", "definitions", "usages", "constructs", "relations"):
        collection = result.get(collection_name, [])
        if not isinstance(collection, list):
            raise ValueError(f"teacher field {collection_name} must be an array")
        for item in collection:
            if not isinstance(item, dict):
                raise ValueError(f"teacher field {collection_name} contains a non-object")
            ranges = []
            if collection_name == "definitions":
                ranges.append(item.get("nameRange"))
                ranges.append(item.get("declarationRange"))
            else:
                ranges.append(item)
            for value in ranges:
                if value is None:
                    continue
                start, end = int(value["start"]), int(value["end"])
                # Conversion validates bounds and surrogate boundaries without
                # changing the manifest's UTF-16 coordinate representation.
                _ = _utf16_to_codepoint(source, start)
                _ = _utf16_to_codepoint(source, end)
                if end < start:
                    raise ValueError("teacher range has end before start")
    return result


def _utf16_to_codepoint(source: str, offset: int) -> int:
    if offset < 0:
        raise ValueError("teacher offset must be non-negative")
    units = 0
    for index, char in enumerate(source):
        if units == offset:
            return index
        width = 2 if ord(char) > 0xFFFF else 1
        if units < offset < units + width:
            raise ValueError("teacher offset splits a UTF-16 surrogate pair")
        units += width
    if units != offset:
        raise ValueError("teacher offset exceeds source")
    return len(source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run a no-retention source parser/indexer teacher")
    parser.add_argument("--language", required=True)
    parser.add_argument("--source-id")
    parser.add_argument("--teacher", choices=("python-ast",), default="python-ast")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = sys.stdin.read()
    if args.teacher == "python-ast" and args.language.lower() not in {"python", "py"}:
        raise SystemExit("python-ast teacher only supports Python")
    result = PythonAstTeacher().annotate(source, TeacherContext(args.language, args.source_id))
    print(json.dumps(validate_annotation_result(result, source), ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
