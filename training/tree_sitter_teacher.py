#!/usr/bin/env python3
"""Tree-sitter syntax teacher using the external JSON teacher protocol."""

from __future__ import annotations

import json
import sys

from .grammar_registry import grammar_candidates


def _utf16_from_byte(source_bytes: bytes, byte_offset: int) -> int:
    return len(source_bytes[:byte_offset].decode("utf-8").encode("utf-16-le")) // 2


def _role(node_type: str) -> str | None:
    value = node_type.lower()
    if "comment" in value:
        return "comment"
    if "string" in value or "template" in value or "heredoc" in value:
        return "string_literal"
    if "number" in value or "integer" in value or "float" in value:
        return "number_literal"
    if value in {"identifier", "type_identifier", "field_identifier", "property_identifier", "variable_name"}:
        return "identifier"
    if "array" in value or "list_literal" in value:
        return "array_literal"
    if "object" in value or "dictionary" in value or "map_literal" in value:
        return "object_literal"
    if "operator" in value:
        return "operator"
    if value in {"true", "false", "null", "nil", "none", "boolean_literal", "null_literal"}:
        return "boolean_literal" if value in {"true", "false", "boolean_literal"} else "null_literal"
    return None


def _construct(node_type: str) -> str | None:
    value = node_type.lower()
    mappings = (
        ("class", "class"), ("interface", "interface"), ("trait", "trait"),
        ("struct", "struct"), ("enum", "enum"), ("record", "record"),
        ("function", "function"), ("method", "method"), ("constructor", "constructor"),
        ("lambda", "lambda"), ("closure", "closure"), ("import", "import"),
        ("include", "include"), ("namespace", "namespace"), ("module", "module"),
        ("package", "package"), ("macro", "macro"), ("preproc", "preprocessor_directive"),
        ("try", "exception_handling"), ("catch", "exception_handler"),
        ("match", "pattern_match"), ("switch", "pattern_match"),
        ("comprehension", "comprehension"), ("generator", "generator"),
        ("call", "call"), ("inherit", "inheritance"), ("extends", "inheritance"),
        ("implements", "implementation"), ("override", "override"),
    )
    for marker, result in mappings:
        if marker in value:
            return result
    return None


def annotate(request: dict) -> dict:
    language = str(request["language"])
    candidates = grammar_candidates(language)
    if not candidates:
        raise ValueError(f"no Tree-sitter grammar configured for {language}")
    import tree_sitter_language_pack as pack
    pack.configure(pack.PackConfig(cache_dir=".treesitter-cache"))
    grammar = next((candidate for candidate in candidates if pack.has_language(candidate)), None)
    if grammar is None:
        raise ValueError(f"Tree-sitter grammar is not cached for {language}")
    source = str(request.get("source", ""))
    source_bytes = source.encode("utf-8")
    tree = pack.get_parser(grammar).parse(source_bytes)
    role_spans = []
    constructs = []
    identifiers = []
    cursor = tree.walk()
    visited = []
    while True:
        node = cursor.node
        start, end = node.start_byte, node.end_byte
        role = _role(node.type)
        if role and end > start:
            role_spans.append({"start": _utf16_from_byte(source_bytes, start),
                               "end": _utf16_from_byte(source_bytes, end), "role": role})
        construct = _construct(node.type)
        if construct and end > start:
            constructs.append({"start": _utf16_from_byte(source_bytes, start),
                               "end": _utf16_from_byte(source_bytes, end), "kind": construct})
        if _role(node.type) == "identifier" and end > start:
            parent = node.parent.type.lower() if node.parent else ""
            identifiers.append((start, end, source_bytes[start:end].decode("utf-8"), parent))
        if cursor.goto_first_child():
            visited.append(False)
            continue
        while visited and not cursor.goto_next_sibling():
            cursor.goto_parent()
            visited.pop()
        if not visited:
            break
    # Tree-sitter is a parser, not a name resolver. These conservative labels
    # are silver supervision: declaration-shaped parents become definitions;
    # other identifiers become usages, linked only to the nearest preceding
    # same-name definition in this file. Ambiguous/cross-file cases remain
    # unresolved rather than being presented as compiler-grade gold.
    definition_markers = ("declarator", "declaration", "parameter", "function", "class", "struct", "type", "field")
    definitions, usages, relations = [], [], []
    prior = {}
    for start, end, name, parent in sorted(identifiers):
        a, b = _utf16_from_byte(source_bytes, start), _utf16_from_byte(source_bytes, end)
        is_definition = any(marker in parent for marker in definition_markers)
        if is_definition:
            identifier = f"definition:{a}:{b}"
            definitions.append({"id": identifier, "nameRange": {"start": a, "end": b}, "kind": "unknown"})
            prior.setdefault(name, []).append(identifier)
        else:
            usages.append({"start": a, "end": b, "kind": "unknown"})
            target = prior.get(name, [])[-1] if prior.get(name) else None
            relations.append({"usageRange": {"start": a, "end": b}, "definitionId": target,
                              "status": "resolved" if target else "no_definition"})
    return {
        "roleSpans": role_spans,
        "definitions": definitions,
        "usages": usages,
        "constructs": constructs,
        "relations": relations,
        "teacher": {"name": "tree-sitter-heuristic", "version": "1.16.2", "semanticStatus": "silver-not-gold",
                    "occurrenceCoverage": "partial"},
    }


def main() -> int:
    request = json.loads(sys.stdin.read())
    result = annotate(request)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
