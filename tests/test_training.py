import json
import tempfile
import unittest
from pathlib import Path

from syntaxlm import SourceSpec
from training.registry import LANGUAGES, REGISTRY_HASH, language_index
from training.annotations import load_annotations, utf16_range_to_bytes
from training.streaming_sources import SourceLedger, iter_sources
from training.sylm1_trainer import train_sources
from training.syl2_trainer import COMPLETION_VOCABULARY, EOS, _should_stop, _supervision_for_source
from training.teachers import PythonAstTeacher, TeacherContext, validate_annotation_result


class TrainingPipelineTests(unittest.TestCase):
    def test_registry_is_stable_and_contains_unknown(self):
        self.assertEqual(len(LANGUAGES), 41)
        self.assertNotEqual(REGISTRY_HASH, "")
        self.assertEqual(language_index("typescript"), language_index("ts"))
        self.assertEqual(LANGUAGES[language_index("not-a-language")], "unknown")

    def test_shared_stream_ledger_contains_no_source_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "sample.py"
            source_path.write_text("# secret-source\nvalue = 42\n", encoding="utf-8")
            ledger_path = root / "ledger.jsonl"
            with SourceLedger(ledger_path) as ledger:
                sources = list(iter_sources(
                    [SourceSpec(str(source_path), language="python", license_id="MIT")],
                    1024,
                    ledger,
                ))
                self.assertEqual(sources[0].text.splitlines()[-1], "value = 42")
                ledger.record(sources[0], "USED", labels={"bytes": len(sources[0].raw)})
                del sources
            text = ledger_path.read_text(encoding="utf-8")
            self.assertNotIn("secret-source", text)
            self.assertIn("contentSha256", text)

    def test_sylm1_module_trainer_exports_legacy_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "sample.py"
            source_path.write_text("def answer():\n    return 42\n", encoding="utf-8")
            result = train_sources(
                [SourceSpec(str(source_path), language="python", license_id="MIT")],
                epochs=1,
                max_bytes=1024,
                ledger_path=str(root / "ledger.jsonl"),
                output=str(root / "sylm1.matrix.bin"),
                output_format="bin",
                license_policy="require",
            )
            self.assertEqual(result["used"], 1)
            self.assertEqual((root / "sylm1.matrix.bin").read_bytes()[:6], b"SYLM\x01\x00")
            record = json.loads((root / "ledger.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "USED")

    def test_syl2_plateau_stop_is_gated_by_precision(self):
        class Args:
            target_precision = 0.99
            plateau_precision_gate = 0.80
            min_epochs = 3
            early_stop_patience = 2
            min_precision_delta = 0.0001
            curvature_threshold = 0.00001

        self.assertEqual(_should_stop([0.61, 0.62001, 0.62002, 0.62003], Args()), (False, None))
        self.assertEqual(_should_stop([0.81, 0.81001, 0.81002, 0.81003], Args()), (True, "precision_plateau"))
        self.assertEqual(_should_stop([0.99], Args()), (True, "target_precision"))

    def test_completion_head_can_emit_eos(self):
        self.assertGreater(EOS, 0)
        self.assertGreater(COMPLETION_VOCABULARY, EOS)

    def test_python_teacher_returns_valid_source_free_annotations(self):
        source = "import os\nclass Box:\n    def get(self):\n        return os.getcwd()\n"
        result = PythonAstTeacher().annotate(source, TeacherContext("python", "box.py"))
        result = validate_annotation_result(result, source)
        self.assertTrue(result["constructs"])
        self.assertTrue(result["definitions"])
        self.assertNotIn("source", result)
        self.assertEqual(result["teacher"]["semanticStatus"], "partial")

    def test_supervised_annotations_use_utf16_and_produce_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "sample.py"
            source = "# 😀\ndef answer():\n    return answer\n"
            source_path.write_text(source, encoding="utf-8")
            manifest = root / "annotations.jsonl"
            manifest.write_text(json.dumps({
                "uri": str(source_path),
                "language": "python",
                "licenseId": "MIT",
                "teacher": {"name": "checked-parser", "version": "v1"},
                "roleSpans": [{"start": 0, "end": 4, "role": "comment"}],
                "definitions": [{"id": "d1", "nameRange": {"start": 9, "end": 15}, "kind": "function"}],
                "usages": [{"start": 29, "end": 35, "definitionId": "d1", "kind": "function"}],
            }) + "\n", encoding="utf-8")
            records = load_annotations(manifest)
            self.assertEqual(utf16_range_to_bytes(source, 0, 4), (0, 6))
            role, symbols, role_coverage, link_count, teacher = _supervision_for_source(
                source, "python", records[0]
            )
            self.assertEqual(role_coverage, 6)
            self.assertEqual(link_count, 1)
            self.assertEqual(teacher, "v1")
            self.assertEqual(symbols[2][0]["definitionId"], "d1")


if __name__ == "__main__":
    unittest.main()
