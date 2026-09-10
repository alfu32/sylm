import dataclasses
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from syntaxlm import SourceSpec
from training.annotations import AnnotationRecord
from training.semantic_models import (
    SymbolRecord, artifact_hash, candidate_features, detect, detector_targets, export,
    make_detector, make_intelligence, pair_features, rank, windows,
)
from training.semantic_trainer import intelligence_step, link_examples, validate_annotation


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class SemanticTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)

    def record(self, complete=False):
        return AnnotationRecord(SourceSpec("doc"), {"occurrenceCoverage": "complete" if complete else "partial"},
                                None, (), ({"id": "gold-d", "start": 0, "end": 1, "kind": "variable"},),
                                ({"start": 4, "end": 5, "definitionId": "gold-d"},))

    def test_partial_masks_and_unicode_windows(self):
        tags, _ = detector_targets("x = x", self.record(), "definitions")
        self.assertEqual(tags, [1, -100, -100, -100, 0])
        tags, _ = detector_targets("x = x", self.record(True), "usages")
        self.assertEqual(tags, [0, 0, 0, 0, 1])
        text = "a😀éz"
        chunks = list(windows(text, 4))
        self.assertEqual("".join(text[a:b] for a, b, _, _ in chunks), text)
        self.assertTrue(all(end - start <= 4 for _, _, start, end in chunks))

    def test_ranker_inputs_are_predictions_and_missing_candidates_are_masked(self):
        u = SymbolRecord("pred-u", "doc", "rev", 4, 5, "predicted-name", 4, .37)
        d = SymbolRecord("pred-d", "doc", "rev", 0, 1, "different-prediction", 2, .62)
        examples, counts = link_examples([u], [d], self.record())
        self.assertEqual(counts["eligible"], 1)
        self.assertEqual(examples[0], (candidate_features(u, [d], 4), 0))
        self.assertEqual(examples[0][0][0][2], 0.) # names not repaired using gold
        self.assertEqual(examples[0][0][0][4], .37)
        _, coverage = link_examples([u], [], self.record())
        self.assertEqual(coverage["missingDefinition"], 1)
        model = make_intelligence()
        before = model.hidden.weight.detach().clone()
        optimizer = self.torch.optim.AdamW(model.parameters(), lr=.01)
        self.assertTrue(intelligence_step(model, optimizer, examples))
        self.assertFalse(self.torch.equal(before, model.hidden.weight))

    def test_distance_and_revision_and_future_context(self):
        u = SymbolRecord("u", "doc", "rev", 100, 104, "name", 1, .9, 110)
        d = SymbolRecord("d", "doc", "rev", 20, 24, "name", 1, .8, 110)
        feature = pair_features(u, d, 110)
        self.assertEqual(len(feature), 32)
        self.assertAlmostEqual(feature[28], -10 / 4096)
        self.assertAlmostEqual(feature[31], 90 / 4096)
        with self.assertRaisesRegex(ValueError, "stale"):
            pair_features(u, dataclasses.replace(d, revision="old"))
        model = make_intelligence()
        with self.assertRaisesRegex(ValueError, "after the cursor"):
            rank(model, u, [d], query_position=104, completion_context=True)
        result = rank(model, u, [], query_position=110, completion_context=True)
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["candidateProbabilities"], [1.])

    def test_training_query_offsets_are_source_boundaries(self):
        record = dataclasses.replace(self.record(), definitions=(), usages=(
            {"start": 2, "end": 3, "queryPosition": 3},))
        validate_annotation("😀x", record)
        for invalid in (1, 4, -1, True, 1.5):
            bad = dataclasses.replace(record, usages=({"start": 2, "end": 3, "queryPosition": invalid},))
            with self.assertRaisesRegex(ValueError, "queryPosition"):
                validate_annotation("😀x", bad)

    def test_null_supervision_and_candidate_limits(self):
        u = SymbolRecord("u", "doc", "rev", 4, 5, "x", 1, .9)
        record = dataclasses.replace(self.record(), usages=(
            {"start": 4, "end": 5, "status": "no_definition"},))
        examples, counts = link_examples([u], [], record)
        self.assertEqual(counts["eligible"], 1)
        self.assertEqual(examples[0], (candidate_features(u, [], 4), 0))
        partial = dataclasses.replace(record, usages=({"start": 4, "end": 5},))
        examples, counts = link_examples([u], [], partial)
        self.assertEqual(len(examples), 0)
        self.assertEqual(counts["unsupportedTarget"], 1)
        d = SymbolRecord("d", "doc", "rev", 0, 1, "x", 1, .9)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            candidate_features(u, [d, d])
        too_many = [dataclasses.replace(d, id=str(i)) for i in range(513)]
        with self.assertRaisesRegex(ValueError, "512"):
            candidate_features(u, too_many)
        examples, counts = link_examples([u], too_many, self.record())
        self.assertEqual(len(examples), 0)
        self.assertEqual(counts["candidateOverflow"], 1)

    @unittest.skipUnless(shutil.which("kotlinc") and shutil.which("kotlin"), "requires Kotlin")
    def test_kotlin_weights_logits_ranges_features_and_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            defs, uses, intel = make_detector(), make_detector(18), make_intelligence()
            export(root / "definitions.model.bin", defs, "definitions", sequence_bytes=8, trained_steps=1)
            export(root / "usages.model.bin", uses, "usages", sequence_bytes=8, trained_steps=1)
            export(root / "code-intelligence.model.bin", intel, "code-intelligence", trained_steps=1,
                   dependencies={role: artifact_hash(root / (role + ".model.bin")) for role in ("definitions", "usages")})
            jar = root / "test.jar"
            subprocess.run(["kotlinc", "src/main/kotlin/syntaxlm/SemanticModels.kt", "tests/SemanticCheck.kt", "-d", str(jar)], check=True, capture_output=True)
            result = subprocess.run(["kotlin", "-classpath", str(jar), "syntaxlm.SemanticCheck", str(root)], check=True, capture_output=True, text=True)
            lines = [json.loads(line) for line in result.stdout.splitlines()]
            with self.torch.inference_mode():
                outputs = defs(self.torch.tensor([[256, 97, 0, 255, 195, 169]]))
            for actual, expected in zip(lines[:3], outputs):
                self.torch.testing.assert_close(self.torch.tensor(actual), expected[0], rtol=1e-4, atol=2e-6)
            records, _ = detect(defs, "# 😀\nαx = 1\nuse(αx)\n", "doc", "rev", "definitions", 8)
            expected = [[r.start, r.end, r.kind, r.confidence, r.context_end] for r in records]
            self.torch.testing.assert_close(self.torch.tensor(lines[3]), self.torch.tensor(expected), rtol=1e-4, atol=2e-6)
            u = SymbolRecord("u", "doc", "rev", 88, 91, "box", 1, .7, 100)
            d = SymbolRecord("d", "doc", "rev", 4, 7, "box", 1, .9, 100)
            self.torch.testing.assert_close(self.torch.tensor(lines[4]), self.torch.tensor(pair_features(u, d, 100)))
            expected = rank(intel, u, [d], 0, 100, True)["candidateProbabilities"]
            self.torch.testing.assert_close(self.torch.tensor(lines[5]), self.torch.tensor(expected))
            broken = root / "definitions.model.bin"
            content = bytearray(broken.read_bytes()); content[-1] ^= 1; broken.write_bytes(content)
            rejected = subprocess.run(["kotlin", "-classpath", str(jar), "syntaxlm.SemanticCheck", str(root)], capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            # A valid checksum is insufficient: the ranker must match its exact detectors.
            export(broken, make_detector(23), "definitions", sequence_bytes=8, trained_steps=1)
            rejected = subprocess.run(["kotlin", "-classpath", str(jar), "syntaxlm.SemanticCheck", str(root)], capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(b"different detector artifacts", rejected.stderr)
