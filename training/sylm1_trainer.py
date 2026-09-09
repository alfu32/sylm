#!/usr/bin/env python3
"""Train the legacy SYLM v1 matrix from a multilingual source stream.

This trainer is intentionally compatible with the existing Kotlin
``SyntaxLmModel``. Its labels are weak labels from the legacy tokenizer; it is
not the learned, grammar-free SYL2 model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from syntaxlm import AveragedPerceptron, KINDS, features, normalize_language, tokenize_code, write_matrix_binary

from .streaming_sources import SourceLedger, iter_sources, source_specs


def train_sources(specs, *, epochs: int, max_bytes: int, ledger_path: str,
                  output: str, output_format: str, license_policy: str) -> dict:
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    if epochs > 1 and any(item.uri == "-" for item in specs):
        raise ValueError("stdin is one-shot; use --epochs 1 or a reacquirable source")

    model = AveragedPerceptron()
    used = 0
    per_language: dict[str, int] = {}
    with SourceLedger(ledger_path) as ledger:
        for epoch in range(epochs):
            for source in iter_sources(specs, max_bytes, ledger, license_policy):
                language = normalize_language(source.spec.language or "text")
                tokens = tokenize_code(source.text, language)
                labels = [token.hint if token.hint in KINDS else "plain" for token in tokens]
                previous_kind = None
                for index, gold in enumerate(labels):
                    feature_list = features(tokens, index, previous_kind)
                    guess = model.predict(feature_list)
                    model.update(gold, guess, feature_list)
                    previous_kind = gold
                ledger.record(
                    source,
                    "USED",
                    labels={"tokens": len(tokens), "epoch": epoch + 1},
                    teacher_versions={"bootstrapLexer": "syntaxlm-legacy-1"},
                )
                per_language[language] = per_language.get(language, 0) + 1
                used += 1
                del tokens, labels, source
    model.averaged()
    skipped_or_failed = len(specs) * epochs - used
    if output_format == "bin":
        write_matrix_binary(model, output)
    else:
        Path(output).write_text(json.dumps(model.to_dict(), separators=(",", ":")), encoding="utf-8")
    return {
        "trainer": "sylm1",
        "format": output_format,
        "output": output,
        "ledger": ledger_path,
        "used": used,
        "skippedOrFailed": skipped_or_failed,
        "epochs": epochs,
        "languages": per_language,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="streaming multilingual SYLM v1 trainer")
    parser.add_argument("--source", action="append", default=[], help="local/raw HTTP file; repeatable")
    parser.add_argument("--manifest", help="JSONL source metadata manifest")
    parser.add_argument("--license-id", help="license for direct --source entries")
    parser.add_argument("--license-policy", choices=("require", "allow"), default="require")
    parser.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--ledger", default="source-use-sylm1.jsonl")
    parser.add_argument("--output", default="sylm1.matrix.bin")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--format", choices=("auto", "json", "bin"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = source_specs(args.manifest, args.source, args.license_id)
    output_format = args.format
    if output_format == "auto":
        output_format = "bin" if Path(args.output).suffix == ".bin" else "json"
    print(json.dumps(train_sources(
        specs,
        epochs=args.epochs,
        max_bytes=args.max_bytes,
        ledger_path=args.ledger,
        output=args.output,
        output_format=output_format,
        license_policy=args.license_policy,
    ), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
