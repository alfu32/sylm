import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from syntaxlm import SourceSpec, load_examples, predict, train, tokenize_code, train_stream, write_matrix_binary  # noqa: E402


class SyntaxLmTests(unittest.TestCase):
    def test_python_offsets_and_hints(self):
        source = "def greet(name):\n    return 'hi'\n"
        tokens = tokenize_code(source, "python")
        self.assertEqual(source[tokens[0].start:tokens[0].end], "def")
        self.assertEqual(tokens[0].hint, "keyword")
        self.assertEqual(tokens[-1].hint, "string")

    def test_model_predicts_useful_spans(self):
        model = train(load_examples(None), 8)
        source = "def greet(name):\n    return 'hi'\n"
        spans = predict(model, source, "python")
        kinds = {span.kind for span in spans}
        self.assertIn("keyword", kinds)
        self.assertIn("string", kinds)
        for span in spans:
            self.assertEqual(source[span.start:span.end], span.text)

    def test_cli_json(self):
        process = subprocess.run(
            [sys.executable, str(ROOT / "syntaxlm.py"), "highlight", "--language", "javascript"],
            input="const x = 42;",
            text=True,
            capture_output=True,
            check=True,
        )
        output = json.loads(process.stdout)
        self.assertEqual(output["language"], "javascript")
        self.assertTrue(any(span["kind"] == "keyword" for span in output["spans"]))

    def test_binary_export_header(self):
        model = train(load_examples(None), 2)
        with tempfile.NamedTemporaryFile() as handle:
            write_matrix_binary(model, handle.name)
            handle.seek(0)
            self.assertEqual(handle.read(6), b"SYLM\x01\x00")

    def test_spans_use_utf16_offsets(self):
        model = train(load_examples(None), 8)
        source = 'print("😀")'
        string_span = next(span for span in predict(model, source, "python") if span.text == '"😀"')
        self.assertEqual(string_span.start, 6)
        self.assertEqual(string_span.end, 10)

    def test_streaming_training_writes_only_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "# private marker\ndef greet(name):\n    return 'hello 😀 ' + name\n"
            source_path = root / "example.py"
            source_path.write_text(source, encoding="utf-8")
            ledger = root / "ledger.jsonl"
            output = root / "model.bin"
            result = train_stream(
                [SourceSpec(str(source_path), language="python", license_id="MIT")],
                epochs=1,
                max_bytes=1024,
                ledger_path=str(ledger),
                output=str(output),
                output_format="bin",
            )
            self.assertEqual(result["used"], 1)
            self.assertTrue(output.exists())
            ledger_text = ledger.read_text(encoding="utf-8")
            self.assertIn('"status":"USED"', ledger_text)
            self.assertIn('"contentSha256":', ledger_text)
            self.assertNotIn("private marker", ledger_text)
            self.assertNotIn("hello 😀", ledger_text)

    def test_streaming_training_requires_license_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "example.py"
            source_path.write_text("x = 1\n", encoding="utf-8")
            ledger = root / "ledger.jsonl"
            result = train_stream(
                [SourceSpec(str(source_path), language="python")],
                epochs=1,
                max_bytes=1024,
                ledger_path=str(ledger),
                output=str(root / "model.bin"),
                output_format="bin",
            )
            self.assertEqual(result["used"], 0)
            self.assertEqual(result["skipped"], 1)
            self.assertIn("missing licenseId", ledger.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
