#!/usr/bin/env python3
"""A tiny learned syntax highlighter with no third-party dependencies."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import keyword
import re
import struct
import sys
import tokenize
import urllib.error
import urllib.request
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


KINDS = (
    "plain",
    "keyword",
    "name",
    "function",
    "type",
    "builtin",
    "string",
    "number",
    "comment",
    "operator",
    "punctuation",
    "decorator",
)

PYTHON_BUILTINS = {
    "abs", "all", "any", "bool", "bytes", "callable", "dict", "enumerate",
    "filter", "float", "format", "frozenset", "getattr", "hasattr", "hash",
    "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "object", "open", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "sorted", "str", "sum", "super", "tuple",
    "type", "vars", "zip", "__import__",
}

KEYWORDS = {
    "python": set(keyword.kwlist),
    "javascript": {
        "as", "async", "await", "break", "case", "catch", "class", "const",
        "continue", "debugger", "default", "delete", "do", "else", "export",
        "extends", "finally", "for", "from", "function", "get", "if", "import",
        "in", "instanceof", "let", "new", "of", "return", "set", "static",
        "super", "switch", "this", "throw", "try", "typeof", "var", "void",
        "while", "with", "yield", "true", "false", "null", "undefined",
    },
    "json": set(),
    "kotlin": {
        "abstract", "actual", "annotation", "as", "break", "by", "catch", "class",
        "companion", "const", "constructor", "continue", "data", "do", "else", "enum",
        "expect", "final", "finally", "for", "fun", "if", "import", "in",
        "infix", "inner", "interface", "internal", "is", "lateinit", "noinline",
        "null", "object", "open", "operator", "out", "override", "package", "private",
        "protected", "public", "reified", "return", "sealed", "select", "setparam",
        "super", "suspend", "tailrec", "this", "throw", "try", "typealias", "typeof",
        "val", "var", "vararg", "when", "where", "while", "true", "false", "null",
    },
}


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    hint: str


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    text: str


def normalize_language(language: str) -> str:
    language = language.lower()
    if language in {"js", "jsx", "javascript", "ts", "tsx", "typescript"}:
        return "javascript"
    if language in {"py", "python3"}:
        return "python"
    if language in {"jsonc"}:
        return "json"
    return language


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\\n", source):
        offsets.append(match.end())
    return offsets


def _position_to_offset(offsets: list[int], position: tuple[int, int]) -> int:
    line, column = position
    if line <= 0:
        return column
    if line > len(offsets):
        return offsets[-1] + column
    return offsets[line - 1] + column


def _python_tokens(source: str) -> Iterator[Token]:
    offsets = _line_offsets(source)
    raw = list(tokenize.generate_tokens(io.StringIO(source).readline))
    significant = [t for t in raw if t.type not in {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}]
    for index, item in enumerate(significant):
        start = _position_to_offset(offsets, item.start)
        end = _position_to_offset(offsets, item.end)
        text = source[start:end]
        if not text:
            continue
        if item.type == tokenize.COMMENT:
            hint = "comment"
        elif item.type == tokenize.STRING:
            hint = "string"
        elif item.type == tokenize.NUMBER:
            hint = "number"
        elif item.type == tokenize.NAME:
            if text in KEYWORDS["python"]:
                hint = "keyword"
            elif text in PYTHON_BUILTINS:
                hint = "builtin"
            elif text[:1].isupper():
                hint = "type"
            elif text.startswith("_") and text.endswith("_"):
                hint = "builtin"
            else:
                following = significant[index + 1].string if index + 1 < len(significant) else ""
                preceding = significant[index - 1].string if index else ""
                hint = "function" if following == "(" or preceding == "def" else "name"
        elif item.type == tokenize.OP:
            hint = "punctuation" if text in "()[]{}:;,.'" else "operator"
            if text == "@":
                hint = "decorator"
        else:
            hint = "plain"
        yield Token(text, start, end, hint)


GENERIC_TOKEN = re.compile(
    r"(?P<space>\s+)|(?P<line_comment>//[^\n]*|#[^\n]*)|(?P<block_comment>/\*[\s\S]*?\*/)|"
    r"(?P<string>(?:\b[rubfRUBF]{0,3})?(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`))|"
    r"(?P<number>\b(?:0[xX][0-9a-fA-F]+|0[bB][01]+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\b)|"
    r"(?P<word>[A-Za-z_$][\w$]*)|(?P<operator>===|!==|==|!=|=>|<=|>=|&&|\|\||\+\+|--|\*\*|//|::|[+\-*/%=<>!&|^~?])|"
    r"(?P<punct>[{}()\[\].,;:])|(?P<other>.)"
)


def _generic_tokens(source: str, language: str) -> Iterator[Token]:
    language = normalize_language(language)
    keywords = KEYWORDS.get(language, set())
    for match in GENERIC_TOKEN.finditer(source):
        group = match.lastgroup
        text = match.group(0)
        if group == "space":
            continue
        if group in {"line_comment", "block_comment"}:
            hint = "comment"
        elif group == "string":
            hint = "string"
        elif group == "number":
            hint = "number"
        elif group == "word":
            if language == "json":
                hint = "plain"
            elif text in keywords:
                hint = "keyword"
            elif text in {"console", "Math", "JSON", "Promise", "require"}:
                hint = "builtin"
            elif text[:1].isupper():
                hint = "type"
            else:
                next_text = source[match.end():].lstrip()[:1]
                hint = "function" if next_text == "(" else "name"
        elif group == "operator":
            hint = "operator"
        elif group == "punct":
            hint = "punctuation"
        else:
            hint = "plain"
        yield Token(text, match.start(), match.end(), hint)


def tokenize_code(source: str, language: str) -> list[Token]:
    language = normalize_language(language)
    if language == "python":
        try:
            return list(_python_tokens(source))
        except (IndentationError, SyntaxError, tokenize.TokenError):
            pass
    return list(_generic_tokens(source, language))


def token_shape(text: str) -> str:
    if text.isdigit() or (text and all(c.isdigit() or c in ".xXabcdefABCDEF" for c in text)):
        return "number"
    if text.isidentifier():
        if text.isupper():
            return "UPPER"
        if text[:1].isupper():
            return "Capitalized"
        if "_" in text:
            return "snake"
        return "word"
    if text.startswith(("'", '"', "`")):
        return "quoted"
    return re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "0", text))


def features(tokens: list[Token], index: int, previous_kind: str | None) -> list[str]:
    token = tokens[index]
    previous = tokens[index - 1].text if index else "<BOS>"
    following = tokens[index + 1].text if index + 1 < len(tokens) else "<EOS>"
    result = [
        "bias",
        f"text={token.text}",
        f"lower={token.text.lower()}",
        f"shape={token_shape(token.text)}",
        f"hint={token.hint}",
        f"prev={previous}",
        f"next={following}",
        f"prev_kind={previous_kind or '<BOS>'}",
    ]
    return result


class AveragedPerceptron:
    def __init__(self) -> None:
        self.weights: dict[str, dict[str, float]] = {}
        self._totals: dict[tuple[str, str], float] = {}
        self._stamps: dict[tuple[str, str], int] = {}
        self.step = 0

    def _weight(self, feature: str, kind: str) -> float:
        return self.weights.get(feature, {}).get(kind, 0.0)

    def _touch(self, feature: str, kind: str) -> None:
        key = (feature, kind)
        last = self._stamps.get(key, 0)
        value = self._weight(feature, kind)
        self._totals[key] = self._totals.get(key, 0.0) + (self.step - last) * value
        self._stamps[key] = self.step

    def _change(self, feature: str, kind: str, amount: float) -> None:
        self._touch(feature, kind)
        self.weights.setdefault(feature, {})[kind] = self._weight(feature, kind) + amount

    def scores(self, fs: Iterable[str]) -> dict[str, float]:
        result = {kind: 0.0 for kind in KINDS}
        for feature in fs:
            for kind, value in self.weights.get(feature, {}).items():
                result[kind] += value
        return result

    def predict(self, fs: Iterable[str]) -> str:
        scores = self.scores(fs)
        return max(KINDS, key=lambda kind: (scores[kind], -KINDS.index(kind)))

    def update(self, gold: str, guess: str, fs: Iterable[str]) -> None:
        self.step += 1
        if gold == guess:
            return
        for feature in fs:
            self._change(feature, gold, 1.0)
            self._change(feature, guess, -1.0)

    def averaged(self) -> "AveragedPerceptron":
        for feature, kinds in list(self.weights.items()):
            for kind in list(kinds):
                self._touch(feature, kind)
                key = (feature, kind)
                kinds[kind] = self._totals.get(key, 0.0) / max(self.step, 1)
        return self

    def to_dict(self) -> dict:
        return {"version": 1, "kinds": list(KINDS), "weights": self.weights}

    @classmethod
    def from_dict(cls, data: dict) -> "AveragedPerceptron":
        model = cls()
        model.weights = {str(k): {str(label): float(v) for label, v in values.items()} for k, values in data["weights"].items()}
        return model


MATRIX_MAGIC = b"SYLM"
MATRIX_VERSION = 1


def _write_u16(handle, value: int) -> None:
    handle.write(struct.pack("<H", value))


def _write_u32(handle, value: int) -> None:
    handle.write(struct.pack("<I", value))


def write_matrix_binary(model: AveragedPerceptron, path: str | Path) -> None:
    """Write a versioned feature vocabulary plus float32 score matrix.

    Layout (all integers/floats little-endian):
      magic[4] = SYLM, version[u16], kind_count[u16], feature_count[u32]
      repeated kind_count: byte_length[u16], UTF-8 kind
      repeated feature_count: byte_length[u16], UTF-8 feature, kind_count float32
    """
    features_in_model = sorted(model.weights)
    with open(path, "wb") as handle:
        handle.write(MATRIX_MAGIC)
        _write_u16(handle, MATRIX_VERSION)
        _write_u16(handle, len(KINDS))
        _write_u32(handle, len(features_in_model))
        for kind in KINDS:
            encoded = kind.encode("utf-8")
            _write_u16(handle, len(encoded))
            handle.write(encoded)
        for feature in features_in_model:
            encoded = feature.encode("utf-8")
            if len(encoded) > 65535:
                raise ValueError(f"feature is too long for matrix format: {feature[:80]}")
            _write_u16(handle, len(encoded))
            handle.write(encoded)
            row = model.weights[feature]
            handle.write(struct.pack("<" + "f" * len(KINDS), *(row.get(kind, 0.0) for kind in KINDS)))


def bootstrap_examples() -> list[tuple[str, str]]:
    return [
        ("python", "def greet(name):\n    # say hello\n    return f'hello {name}'\n"),
        ("python", "class Counter:\n    def __init__(self, start=0):\n        self.value = start\n        self.value += 1\n"),
        ("python", "from pathlib import Path\nitems = [1, 2, 3]\nprint(sum(items))\n"),
        ("javascript", "function greet(name) {\n  const message = `hello ${name}`;\n  return message;\n}"),
        ("javascript", "export class Counter {\n  constructor(start = 0) { this.value = start; }\n}"),
        ("javascript", "const values = [1, 2, 3];\nconsole.log(values.map(x => x * 2));"),
        ("json", '{\n  "name": "syntaxlm",\n  "version": 1,\n  "enabled": true\n}'),
        ("kotlin", "fun twice(x: Int): Int = x * 2\n"),
        ("kotlin", "data class User(val name: String, val active: Boolean)\n"),
    ]


def load_examples(path: str | None) -> list[tuple[str, list[Token], list[str]]]:
    if not path:
        return [(language, tokenize_code(code, language), [t.hint for t in tokenize_code(code, language)]) for language, code in bootstrap_examples()]
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            code = item["code"]
            language = item.get("language", "text")
            tokens = tokenize_code(code, language)
            labels = [t.hint for t in tokens]
            for span in item.get("spans", []):
                for i, token in enumerate(tokens):
                    if token.start >= span["start"] and token.end <= span["end"]:
                        labels[i] = span["kind"]
            examples.append((language, tokens, labels))
    if not examples:
        raise ValueError("training data did not contain any examples")
    return examples


@dataclass(frozen=True)
class SourceSpec:
    """A source identity and its training metadata, never the source contents."""

    uri: str
    language: str | None = None
    license_id: str | None = None
    source_id: str | None = None
    repository_commit: str | None = None
    relative_path: str | None = None
    split: str = "train"
    provider_id: str = "raw-file"


LANGUAGE_EXTENSIONS = {
    ".py": "python", ".pyw": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".json": "json", ".jsonc": "json",
    ".kt": "kotlin", ".kts": "kotlin", ".java": "java", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".cs": "csharp", ".go": "go",
    ".rs": "rust", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".scala": "scala", ".sql": "sql", ".lua": "lua", ".sh": "shell",
    ".ps1": "powershell", ".r": "r", ".R": "r", ".m": "objective-c",
}


def _language_from_uri(uri: str) -> str | None:
    suffix = Path(uri.split("?", 1)[0]).suffix
    return LANGUAGE_EXTENSIONS.get(suffix.lower())


def _source_specs(manifest: str | None, sources: list[str], default_license: str | None) -> list[SourceSpec]:
    result: list[SourceSpec] = []
    if manifest:
        with open(manifest, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid source manifest JSON on line {line_number}: {error}") from error
                if not item.get("uri"):
                    raise ValueError(f"source manifest line {line_number} has no uri")
                result.append(SourceSpec(
                    uri=str(item["uri"]),
                    language=item.get("language"),
                    license_id=item.get("licenseId", default_license),
                    source_id=item.get("sourceId"),
                    repository_commit=item.get("repositoryCommit"),
                    relative_path=item.get("relativePath"),
                    split=str(item.get("split", "train")),
                    provider_id=str(item.get("providerId", "raw-file")),
                ))
    for uri in sources:
        result.append(SourceSpec(uri=uri, language=_language_from_uri(uri), license_id=default_license))
    if not result:
        raise ValueError("provide --source or --manifest")
    return result


def _read_bounded(uri: str, max_bytes: int) -> bytes:
    """Read one source into a bounded buffer; never creates a source cache."""
    if max_bytes <= 0:
        raise ValueError("--max-bytes must be positive")
    if uri == "-":
        stream = sys.stdin.buffer
        close = False
    elif uri.startswith(("http://", "https://")):
        stream = urllib.request.urlopen(uri, timeout=30)  # no urllib cache is used
        close = True
        content_length = stream.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            stream.close()
            raise ValueError(f"source exceeds --max-bytes ({content_length} > {max_bytes})")
    else:
        stream = open(uri, "rb")
        close = True
    try:
        result = bytearray()
        while True:
            chunk = stream.read(min(64 * 1024, max_bytes - len(result) + 1))
            if not chunk:
                break
            result.extend(chunk)
            if len(result) > max_bytes:
                raise ValueError(f"source exceeds --max-bytes ({len(result)} > {max_bytes})")
        return bytes(result)
    finally:
        if close:
            stream.close()


def _decode_source(raw: bytes) -> tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8"), "utf-8-sig"
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("source is not decodable")


def _write_ledger(handle, spec: SourceSpec, status: str, *, run_id: str, raw: bytes | None = None,
                  language: str | None = None, reason: str | None = None,
                  encoding: str | None = None, label_coverage: dict | None = None) -> None:
    """Write metadata only. Never include source text, tokens, or annotations."""
    record = {
        "runId": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providerId": spec.provider_id,
        "sourceId": spec.source_id,
        "sourceUri": spec.uri,
        "repositoryCommit": spec.repository_commit,
        "relativePath": spec.relative_path,
        "contentSha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
        "byteCount": len(raw) if raw is not None else None,
        "encoding": encoding,
        "licenseId": spec.license_id,
        "canonicalLanguageId": language,
        "split": spec.split,
        "teacherVersions": {"bootstrapLexer": "syntaxlm-legacy-1"},
        "labelCoverage": label_coverage or {},
        "status": status,
        "reason": reason,
    }
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def train_stream(specs: list[SourceSpec], epochs: int, max_bytes: int, ledger_path: str,
                 output: str, output_format: str, license_policy: str = "require") -> dict:
    """Train by reacquiring one source per epoch and discarding it after use.

    This is intentionally the streaming bootstrap path for the existing SYLM v1
    model. It does not claim to provide the neural three-model architecture yet.
    """
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    if epochs > 1 and any(spec.uri == "-" for spec in specs):
        raise ValueError("stdin is one-shot; use --epochs 1 or a reacquirable source")
    model = AveragedPerceptron()
    used = 0
    skipped = 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as ledger:
        for epoch in range(epochs):
            for spec in specs:
                if license_policy == "require" and not spec.license_id:
                    _write_ledger(ledger, spec, "SKIPPED", run_id=run_id, reason="missing licenseId")
                    skipped += 1
                    continue
                raw: bytes | None = None
                try:
                    raw = _read_bounded(spec.uri, max_bytes)
                    source, encoding = _decode_source(raw)
                    language = normalize_language(spec.language or _language_from_uri(spec.uri) or "text")
                    tokens = tokenize_code(source, language)
                    labels = [token.hint if token.hint in KINDS else "plain" for token in tokens]
                    previous_kind = None
                    for index, gold in enumerate(labels):
                        feature_list = features(tokens, index, previous_kind)
                        guess = model.predict(feature_list)
                        model.update(gold, guess, feature_list)
                        previous_kind = gold
                    _write_ledger(
                        ledger, spec, "USED", run_id=run_id, raw=raw, language=language, encoding=encoding,
                        label_coverage={"tokens": len(tokens), "epoch": epoch + 1},
                    )
                    used += 1
                except (OSError, UnicodeError, ValueError, urllib.error.URLError) as error:
                    _write_ledger(ledger, spec, "FAILED", run_id=run_id,
                                  reason=type(error).__name__ + ": " + str(error)[:300])
                finally:
                    # Drop the only source-bearing references before the next example.
                    del raw
                    if "source" in locals():
                        del source
                    if "tokens" in locals():
                        del tokens
    model.averaged()
    if output_format == "bin":
        write_matrix_binary(model, output)
    else:
        Path(output).write_text(json.dumps(model.to_dict(), separators=(",", ":")), encoding="utf-8")
    return {"used": used, "skipped": skipped, "epochs": epochs, "output": output, "ledger": ledger_path}


def train(examples: list[tuple[str, list[Token], list[str]]], epochs: int) -> AveragedPerceptron:
    model = AveragedPerceptron()
    for _ in range(max(1, epochs)):
        for _language, tokens, labels in examples:
            previous_kind = None
            for index, gold in enumerate(labels):
                fs = features(tokens, index, previous_kind)
                guess = model.predict(fs)
                model.update(gold if gold in KINDS else "plain", guess, fs)
                previous_kind = gold
    return model.averaged()


def predict(model: AveragedPerceptron, source: str, language: str) -> list[Span]:
    tokens = tokenize_code(source, language)
    result: list[Span] = []
    previous_kind = None
    previous_highlighted_end = None
    previous_highlighted_start = None
    for index, token in enumerate(tokens):
        kind = model.predict(features(tokens, index, previous_kind))
        previous_kind = kind
        if kind == "plain":
            previous_highlighted_end = None
            previous_highlighted_start = None
            continue
        span = Span(_utf16_offset(source, token.start), _utf16_offset(source, token.end), kind, token.text)
        if result and previous_highlighted_end == token.start and result[-1].kind == span.kind:
            old = result[-1]
            result[-1] = Span(old.start, span.end, old.kind, source[previous_highlighted_start:token.end])
        else:
            result.append(span)
            previous_highlighted_start = token.start
        previous_highlighted_end = token.end
    return result


def _utf16_offset(source: str, codepoint_offset: int) -> int:
    """Return the offset Kotlin/JVM String APIs use for this Python string."""
    return len(source[:codepoint_offset].encode("utf-16-le")) // 2


ANSI = {
    "keyword": "1;35", "function": "1;34", "type": "1;36", "builtin": "36",
    "string": "32", "number": "33", "comment": "2;37", "operator": "1;33",
    "punctuation": "1;37", "decorator": "1;32", "name": "0;37",
}


def ansi_output(source: str, spans: list[Span]) -> str:
    chunks: list[str] = []
    cursor = 0
    for span in spans:
        # Span offsets are UTF-16 for Kotlin interoperability; locate the token
        # in Python's codepoint-indexed string for terminal rendering.
        start = source.find(span.text, cursor)
        if start < 0:
            continue
        chunks.append(source[cursor:start])
        chunks.append(f"\033[{ANSI.get(span.kind, '0')}m{span.text}\033[0m")
        cursor = start + len(span.text)
    chunks.append(source[cursor:])
    return "".join(chunks)


def command_train(args: argparse.Namespace) -> int:
    examples = load_examples(args.data)
    model = train(examples, args.epochs)
    payload = model.to_dict()
    payload["training_examples"] = len(examples)
    output_format = args.format
    if output_format == "auto":
        output_format = "bin" if Path(args.output).suffix == ".bin" else "json"
    if output_format == "bin":
        write_matrix_binary(model, args.output)
    else:
        Path(args.output).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.output} ({len(model.weights)} features, {len(examples)} examples, {output_format})")
    return 0


def command_train_stream(args: argparse.Namespace) -> int:
    specs = _source_specs(args.manifest, args.source, args.license_id)
    output_format = args.format
    if output_format == "auto":
        output_format = "bin" if Path(args.output).suffix == ".bin" else "json"
    result = train_stream(
        specs,
        epochs=args.epochs,
        max_bytes=args.max_bytes,
        ledger_path=args.ledger,
        output=args.output,
        output_format=output_format,
        license_policy=args.license_policy,
    )
    print(json.dumps(result, separators=(",", ":")))
    return 0


def command_highlight(args: argparse.Namespace) -> int:
    source = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
    if args.model and Path(args.model).exists():
        model = AveragedPerceptron.from_dict(json.loads(Path(args.model).read_text(encoding="utf-8")))
    else:
        model = train(load_examples(None), 8)
    spans = predict(model, source, args.language)
    if args.format == "ansi":
        sys.stdout.write(ansi_output(source, spans))
    else:
        print(json.dumps({"language": normalize_language(args.language), "spans": [asdict(s) for s in spans]}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny learned syntax highlighting model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train and save a model")
    train_parser.add_argument("--data", help="JSONL annotations; omitted for built-in bootstrap corpus")
    train_parser.add_argument("--output", default="syntaxlm.json")
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--format", choices=("auto", "json", "bin"), default="auto")
    train_parser.set_defaults(func=command_train)
    stream_parser = subparsers.add_parser(
        "train-stream",
        help="train the legacy matrix model from bounded, non-retained source streams",
    )
    stream_parser.add_argument("--source", action="append", default=[],
                               help="local file, raw HTTP(S) file, or - for stdin; repeatable")
    stream_parser.add_argument("--manifest", help="JSONL source metadata manifest")
    stream_parser.add_argument("--license-id", help="license for direct --source entries")
    stream_parser.add_argument("--license-policy", choices=("require", "allow"), default="require",
                               help="require a manifest licenseId unless explicitly relaxed")
    stream_parser.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024,
                               help="maximum in-memory size of one source")
    stream_parser.add_argument("--ledger", default="source-use.jsonl",
                               help="metadata-only source-use ledger")
    stream_parser.add_argument("--output", default="syntaxlm.matrix.bin")
    stream_parser.add_argument("--epochs", type=int, default=1)
    stream_parser.add_argument("--format", choices=("auto", "json", "bin"), default="auto")
    stream_parser.set_defaults(func=command_train_stream)
    highlight_parser = subparsers.add_parser("highlight", help="predict highlighting spans")
    highlight_parser.add_argument("--language", "-l", default="python")
    highlight_parser.add_argument("--model", "-m")
    highlight_parser.add_argument("--format", choices=("json", "ansi"), default="json")
    highlight_parser.add_argument("file", nargs="?", default="-")
    highlight_parser.set_defaults(func=command_highlight)
    return parser


if __name__ == "__main__":
    try:
        arguments = build_parser().parse_args()
        raise SystemExit(arguments.func(arguments))
    except BrokenPipeError:
        pass
