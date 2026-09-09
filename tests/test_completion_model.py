"""Neural checks; skipped when optional training dependencies are absent."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from training.syl2_trainer import _make_models
from training.syl2_format import read_syl2, write_syl2


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class CompletionModelTests(unittest.TestCase):
    def test_configurable_model_preserves_causality_state_and_export(self):
        torch, _, model, _, _ = _make_models(
            17, completion_embedding=12, completion_hidden=24, completion_layers=3)
        model.eval()
        ids = torch.tensor([[256, 97, 98, 99, 100]])
        with torch.no_grad():
            full = model(ids)
            first = model(ids[:, :3])
            last = model(ids[:, 3:], first[3])
        torch.testing.assert_close(full[0][:, :3], first[0])
        torch.testing.assert_close(full[0][:, 3:], last[0])
        torch.testing.assert_close(full[3], last[3])
        self.assertEqual(tuple(full[3].shape), (3, 1, 24))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            write_syl2(path, {"taskId": "next-word"}, model.state_dict())
            _, tensors = read_syl2(path)
            self.assertEqual(tensors["encoder.weight_hh_l2"][0], (72, 24))
            self.assertEqual(tensors["embedding.weight"][0], (259, 12))

    def test_invalid_dimensions_fail_before_building(self):
        with self.assertRaises(ValueError):
            _make_models(completion_layers=0)
