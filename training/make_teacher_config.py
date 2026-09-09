#!/usr/bin/env python3
"""Create the pinned Tree-sitter external-teacher configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .grammar_registry import grammar_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="teacher-config.json")
    args = parser.parse_args()
    config = {}
    for language, entry in grammar_plan().items():
        if entry["candidates"]:
            config[language] = {
                "command": [".venv/bin/python", "-m", "training.tree_sitter_teacher", "--language", language],
                "name": "tree-sitter-syntax",
                "version": "1.16.2",
                "timeoutSeconds": 30,
                "maxOutputBytes": 8388608,
                "semanticStatus": "syntax-only",
            }
    Path(args.output).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"teachers": len(config), "output": args.output}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
