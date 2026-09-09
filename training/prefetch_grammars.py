#!/usr/bin/env python3
"""Prefetch required Tree-sitter grammars into an explicit local cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .grammar_registry import grammar_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="prefetch the project's Tree-sitter grammar set")
    parser.add_argument("--cache-dir", default=".treesitter-cache")
    parser.add_argument("--report", default="grammar-coverage.json")
    parser.add_argument("--language", action="append", help="project language ID; repeatable")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import tree_sitter_language_pack as pack

    pack.configure(pack.PackConfig(cache_dir=args.cache_dir))
    plan = grammar_plan()
    selected = args.language or list(plan)
    available = set(pack.manifest_languages())
    downloaded = []
    unsupported = []
    for language in selected:
        entry = plan[language]
        grammar = next((candidate for candidate in entry["candidates"] if candidate in available), None)
        entry["selected"] = grammar
        if grammar is None:
            unsupported.append(language)
            continue
        pack.prefetch([grammar])
        downloaded.append(grammar)
        entry["downloaded"] = True
    report = {
        "pack": "tree-sitter-language-pack",
        "cacheDir": args.cache_dir,
        "languages": {language: plan[language] for language in selected},
        "downloadedGrammars": sorted(set(downloaded)),
        "unsupportedProjectLanguages": unsupported,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"downloaded": len(set(downloaded)), "unsupported": unsupported,
                      "report": args.report}, separators=(",", ":")))
    return 0 if not unsupported else 2


if __name__ == "__main__":
    raise SystemExit(main())
