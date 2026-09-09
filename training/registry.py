"""Stable multilingual language registry used in SYL2 metadata and targets."""

from __future__ import annotations

import hashlib
import json


# IDs are stable and must not be regenerated from current popularity ranks.
LANGUAGES = (
    "python", "c", "cpp", "java", "csharp", "javascript", "visual-basic",
    "sql", "r", "rust", "fortran", "go", "delphi", "php", "scratch",
    "assembly", "ada", "swift", "objective-c", "cobol", "julia", "ruby",
    "perl", "sas", "classic-visual-basic", "kotlin", "matlab", "caml",
    "prolog", "gml", "lua", "powershell", "d", "plsql", "abap",
    "transact-sql", "vbscript", "ocaml", "typescript", "zig",
    "unknown", "json", "xml", "html", "jsx", "tsx", "svelte",
)

LANGUAGE_TO_INDEX = {language: index for index, language in enumerate(LANGUAGES)}
REGISTRY_VERSION = "tiobe-top40-plus-supplemental-2026-09"
REGISTRY_HASH = hashlib.sha256(json.dumps(
    {"version": REGISTRY_VERSION, "languages": LANGUAGES},
    separators=(",", ":"),
).encode("utf-8")).hexdigest()


def language_index(language: str | None) -> int:
    if not language:
        return LANGUAGE_TO_INDEX["unknown"]
    value = language.lower().replace("_", "-")
    aliases = {
        "js": "javascript", "ts": "typescript", "py": "python", "c++": "cpp",
        "c#": "csharp", "vb": "visual-basic", "vb.net": "visual-basic",
        "objectivec": "objective-c", "pl/sql": "plsql", "t-sql": "transact-sql",
    }
    return LANGUAGE_TO_INDEX.get(aliases.get(value, value), LANGUAGE_TO_INDEX["unknown"])
