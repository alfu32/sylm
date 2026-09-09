"""Mapping from project language IDs to Tree-sitter Language Pack IDs."""

from __future__ import annotations

from .registry import LANGUAGES


# Exact grammar names are preferred. Fallbacks are syntax-family coverage and
# must remain visible in metadata; they do not establish dialect support.
TREE_SITTER_LANGUAGE_MAP = {
    "python": ("python",), "c": ("c",), "cpp": ("cpp",), "java": ("java",),
    "csharp": ("csharp",), "javascript": ("javascript",), "visual-basic": ("vb",),
    "sql": ("sql",), "r": ("r",), "rust": ("rust",), "fortran": ("fortran",),
    "go": ("go",), "delphi": ("pascal",), "php": ("php",), "scratch": (),
    "assembly": ("asm", "x86asm", "nasm"), "ada": ("ada",), "swift": ("swift",),
    "objective-c": ("objc",), "cobol": ("cobol",), "julia": ("julia",),
    "ruby": ("ruby",), "perl": ("perl",), "sas": ("sas",),
    "classic-visual-basic": ("vb",), "kotlin": ("kotlin",), "matlab": ("matlab",),
    "caml": ("ocaml",), "prolog": ("prolog",), "gml": (), "lua": ("lua",),
    "powershell": ("powershell",), "d": ("d",), "plsql": ("sql", "postgres"),
    "abap": (), "transact-sql": ("tsql",), "vbscript": ("vb",), "ocaml": ("ocaml",),
    "typescript": ("typescript",), "zig": ("zig",), "unknown": (),
}


def grammar_candidates(language: str) -> tuple[str, ...]:
    return TREE_SITTER_LANGUAGE_MAP.get(language, ())


def grammar_plan() -> dict[str, dict]:
    return {
        language: {
            "candidates": list(grammar_candidates(language)),
            "coverage": "exact" if len(grammar_candidates(language)) == 1 and language not in {
                "delphi", "caml", "plsql", "classic-visual-basic", "vbscript"
            } else ("fallback" if grammar_candidates(language) else "unsupported"),
        }
        for language in LANGUAGES
    }
