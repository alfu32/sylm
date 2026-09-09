import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from syntaxlm import load_examples, predict, train, tokenize_code, write_matrix_binary  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
