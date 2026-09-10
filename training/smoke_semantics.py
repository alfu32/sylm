"""Bounded integration smoke test using tiny authored, temporary Python fixtures.

This is a pipeline test, not a corpus or quality benchmark. Its annotations are
complete by construction. No fetched code or semantic-gold claims are involved.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="semantic-fixtures-") as directory:
        temporary = Path(directory)
        annotations = []
        for index, names in enumerate(("xy", "ab", "pq")):
            x, y = names
            text = f"{x} = 1\n{y} = 2\n{x} + {y}\n"
            source = temporary / f"sample{index}.py"
            source.write_text(text)
            annotations.append({
                "uri": str(source), "language": "python", "licenseId": "authored-test-fixture",
                "split": "validation" if index == 2 else "train",
                "contentSha256": hashlib.sha256(text.encode()).hexdigest(),
                "teacher": {"name": "authored-fixture", "version": "2", "occurrenceCoverage": "complete"},
                "definitions": [{"id": x, "start": 0, "end": 1, "kind": "variable"},
                                {"id": y, "start": 6, "end": 7, "kind": "variable"}],
                "usages": [{"start": 12, "end": 13, "definitionId": x, "kind": "variable"},
                           {"start": 16, "end": 17, "definitionId": y, "kind": "variable"}],
            })
        manifest = temporary / "annotations.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in annotations))
        subprocess.run([sys.executable, "-m", "training.semantic_trainer",
                        "--annotation-manifest", str(manifest), "--output-dir", str(root),
                        "--ledger", str(root) + "-source-use.jsonl", "--epochs", "40",
                        "--intelligence-epochs", "10", "--sequence-bytes", "32", "--learning-rate", ".003"], check=True)
    result = json.loads((root / "training.json").read_text())
    if result["intelligenceSteps"] < 1:
        raise RuntimeError("smoke run produced no eligible links; inspect upstream detector outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
