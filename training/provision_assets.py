#!/usr/bin/env python3
"""Provision reproducible parser/dataset assets for a training run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .grammar_registry import grammar_plan
from .registry import LANGUAGES, REGISTRY_HASH, REGISTRY_VERSION


DEFAULT_MLCPD_FILES = [
    "c_parsed_1.parquet", "c_sharp_parsed_1.parquet", "cpp_parsed_1.parquet",
    "java_parsed_1.parquet", "javascript_parsed_1.parquet", "python_parsed_1.parquet",
    "go_parsed_1.parquet", "rust_parsed_1.parquet", "typescript_parsed_1.parquet",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="provision assets for a multilingual training run")
    parser.add_argument("--install-dependencies", action="store_true")
    parser.add_argument("--grammar-cache", default=".treesitter-cache")
    parser.add_argument("--grammar-report", default="grammar-coverage.json")
    parser.add_argument("--teacher-config", default="teacher-config.json")
    parser.add_argument("--run-manifest", default="training-run.json")
    parser.add_argument("--mlcpd-file", action="append", dest="mlcpd_files")
    parser.add_argument("--max-mlcpd-examples", type=int, default=10000)
    return parser


def _install() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)


def _prefetch(cache: str, report_path: str) -> dict:
    import tree_sitter_language_pack as pack
    pack.configure(pack.PackConfig(cache_dir=cache))
    plan = grammar_plan()
    available = set(pack.manifest_languages())
    downloaded = []
    unsupported = []
    for language, entry in plan.items():
        grammar = next((candidate for candidate in entry["candidates"] if candidate in available), None)
        entry["selected"] = grammar
        if grammar is None:
            unsupported.append(language)
            continue
        pack.prefetch([grammar])
        downloaded.append(grammar)
        entry["downloaded"] = True
    report = {"pack": "tree-sitter-language-pack", "cacheDir": cache,
              "languages": plan, "downloadedGrammars": sorted(set(downloaded)),
              "unsupportedProjectLanguages": unsupported}
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = build_parser().parse_args()
    if args.install_dependencies:
        _install()
    report = _prefetch(args.grammar_cache, args.grammar_report)
    teacher_config = {}
    for language, entry in report["languages"].items():
        if entry.get("candidates"):
            teacher_config[language] = {
                "command": [sys.executable, "-m", "training.tree_sitter_teacher", "--language", language],
                "name": "tree-sitter-syntax", "version": "1.16.2",
                "timeoutSeconds": 30, "maxOutputBytes": 8388608,
                "semanticStatus": "syntax-only",
            }
    Path(args.teacher_config).write_text(json.dumps(teacher_config, indent=2) + "\n", encoding="utf-8")
    files = args.mlcpd_files or DEFAULT_MLCPD_FILES
    run_manifest = {
        "registryVersion": REGISTRY_VERSION, "registryHash": REGISTRY_HASH,
        "languages": list(LANGUAGES),
        "datasets": [{"name": "MLCPD", "repository": "jugalgajjar/MultiLang-Code-Parser-Dataset",
                       "files": files, "streaming": True, "maxExamplesPerEpoch": args.max_mlcpd_examples,
                       "rawRetention": "none", "licenseReview": "required"}],
        "grammarReport": args.grammar_report, "teacherConfig": args.teacher_config,
        "semanticTeachers": "SCIP/compiler/CodeQL adapters must be configured separately",
    }
    Path(args.run_manifest).write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"grammarCount": len(report["downloadedGrammars"]),
                      "unsupported": report["unsupportedProjectLanguages"],
                      "teacherCount": len(teacher_config), "mlcpdFiles": files,
                      "runManifest": args.run_manifest}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
