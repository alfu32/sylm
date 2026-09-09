#!/usr/bin/env python3
"""Train the first SYL2 artifacts from bounded multilingual source streams.

The completion head is genuinely self-supervised: source bytes are its targets.
The role and symbol heads use the legacy tokenizer as weak supervision. Semantic
definition links are therefore marked PARTIAL in metadata until compiler/indexer
annotations are supplied by a later teacher pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from syntaxlm import normalize_language, tokenize_code

from .registry import LANGUAGES, REGISTRY_HASH, REGISTRY_VERSION, language_index
from .streaming_sources import SourceLedger, iter_sources, source_specs
from .syl2_format import write_syl2


BYTE_VOCABULARY = 259
BOS = 256
EOS = 257
PAD = 258
COMPLETION_VOCABULARY = 257
ROLE_LABELS = (
    "plain", "keyword", "identifier", "string_literal", "number_literal",
    "comment", "operator", "punctuation", "array_literal", "object_literal",
    "boolean_literal", "null_literal", "unknown", "decorator",
)
OCCURRENCE_LABELS = ("none", "definition", "usage", "both", "unknown")


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ModuleNotFoundError as error:
        raise SystemExit("SYL2 training requires torch and numpy; run `python3 -m pip install -r requirements.txt`") from error
    return torch, nn, functional


def _byte_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _utf8_boundaries(text: str) -> list[int]:
    result = [0]
    for char in text:
        result.append(result[-1] + len(char.encode("utf-8")))
    return result


def _role_targets(text: str, language: str) -> list[int]:
    mapping = {
        "plain": "plain", "keyword": "keyword", "name": "identifier", "function": "identifier",
        "type": "identifier", "builtin": "identifier", "string": "string_literal", "number": "number_literal",
        "comment": "comment", "operator": "operator", "punctuation": "punctuation", "decorator": "decorator",
    }
    labels = [ROLE_LABELS.index("plain")] * len(text.encode("utf-8"))
    boundaries = _utf8_boundaries(text)
    for token in tokenize_code(text, language):
        role = mapping.get(token.hint, "unknown")
        start, end = boundaries[token.start], boundaries[token.end]
        for index in range(start, min(end, len(labels))):
            labels[index] = ROLE_LABELS.index(role)
    return labels


def _occurrence_targets(text: str, language: str) -> list[int]:
    labels = [0] * len(text.encode("utf-8"))
    boundaries = _utf8_boundaries(text)
    token_roles = {"function": "definition", "name": "usage", "type": "usage", "builtin": "usage"}
    for token in tokenize_code(text, language):
        role = token_roles.get(token.hint)
        if role is None:
            continue
        value = OCCURRENCE_LABELS.index(role)
        for index in range(boundaries[token.start], min(boundaries[token.end], len(labels))):
            labels[index] = value
    return labels


def _boundary_targets(values: list[int]) -> list[float]:
    """A deliberately simple auxiliary boundary target for self-supervision."""
    return [1.0 if value in b" \t\r\n.,;:()[]{}<>+-=*/!&|" else 0.0 for value in values]


def _make_models(seed: int | None = None):
    torch, nn, functional = _torch()
    if seed is not None:
        torch.manual_seed(seed)
    language_count = len(LANGUAGES)

    class CausalByteModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(BYTE_VOCABULARY, 96, padding_idx=PAD)
            self.encoder = nn.GRU(96, 192, num_layers=2, batch_first=True)
            self.byte_head = nn.Linear(192, COMPLETION_VOCABULARY)
            self.boundary_head = nn.Linear(192, 1)
            self.language_head = nn.Linear(192, language_count)

        def forward(self, ids, hidden=None):
            output, hidden = self.encoder(self.embedding(ids), hidden)
            return self.byte_head(output), self.boundary_head(output).squeeze(-1), self.language_head(output[:, -1]), hidden

    class RoleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(BYTE_VOCABULARY, 64, padding_idx=PAD)
            self.encoder = nn.GRU(64, 64, num_layers=2, bidirectional=True, batch_first=True)
            self.role_head = nn.Linear(128, len(ROLE_LABELS))
            self.language_head = nn.Linear(128, language_count)

        def forward(self, ids):
            output, _hidden = self.encoder(self.embedding(ids))
            return self.role_head(output), self.language_head(output.mean(dim=1))

    class SymbolModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(BYTE_VOCABULARY, 64, padding_idx=PAD)
            self.encoder = nn.GRU(64, 64, num_layers=2, bidirectional=True, batch_first=True)
            self.occurrence_head = nn.Linear(128, len(OCCURRENCE_LABELS))
            self.symbol_kind_head = nn.Linear(128, 8)
            self.language_head = nn.Linear(128, language_count)
            self.link_hidden = nn.Linear(400, 64)
            self.link_score = nn.Linear(64, 1)

        def forward(self, ids):
            output, _hidden = self.encoder(self.embedding(ids))
            return (
                self.occurrence_head(output),
                self.symbol_kind_head(output),
                self.language_head(output.mean(dim=1)),
            )

    return torch, functional, CausalByteModel(), RoleModel(), SymbolModel()


def _role_step(torch, functional, model, optimizer, data, targets, language, sequence_bytes, device):
    model.train()
    total = 0.0
    chunks = 0
    language_target = torch.tensor([language], dtype=torch.long, device=device)
    for start in range(0, len(data), sequence_bytes):
        end = min(start + sequence_bytes, len(data))
        chunk = data[start:end]
        ids = torch.tensor([[BOS] + chunk], dtype=torch.long, device=device)
        target = torch.tensor([targets[start:end]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        role_logits, language_logits = model(ids)
        loss = functional.cross_entropy(role_logits[:, 1:].reshape(-1, len(ROLE_LABELS)), target.reshape(-1))
        loss = loss + 0.10 * functional.cross_entropy(language_logits, language_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        chunks += 1
    return total / max(chunks, 1)


def _symbol_step(torch, functional, model, optimizer, data, targets, language, sequence_bytes, device):
    model.train()
    total = 0.0
    chunks = 0
    language_target = torch.tensor([language], dtype=torch.long, device=device)
    for start in range(0, len(data), sequence_bytes):
        end = min(start + sequence_bytes, len(data))
        ids = torch.tensor([[BOS] + data[start:end]], dtype=torch.long, device=device)
        target = torch.tensor([targets[start:end]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        occurrence_logits, _kind_logits, language_logits = model(ids)
        loss = functional.cross_entropy(
            occurrence_logits[:, 1:].reshape(-1, len(OCCURRENCE_LABELS)), target.reshape(-1)
        )
        loss = loss + 0.10 * functional.cross_entropy(language_logits, language_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        chunks += 1
    return total / max(chunks, 1)


def _classification_precision(torch, logits, target) -> tuple[int, int]:
    if target.numel() == 0:
        return 0, 0
    prediction = torch.argmax(logits, dim=-1)
    correct = int((prediction.reshape(-1) == target.reshape(-1)).sum().detach().cpu())
    return correct, int(target.numel())


def _completion_precision_counts(torch, functional, model, data, language, sequence_bytes, device) -> tuple[int, int]:
    if not data:
        return 0, 0
    model.eval()
    hidden = None
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(data), sequence_bytes):
            end = min(start + sequence_bytes, len(data))
            target_values = data[start:end]
            if start == 0:
                input_values = [BOS] + target_values[:-1]
            else:
                input_values = data[start - 1:end - 1]
            ids = torch.tensor([input_values], dtype=torch.long, device=device)
            target = torch.tensor([target_values], dtype=torch.long, device=device)
            byte_logits, _boundary_logits, _language_logits, hidden = model(ids, hidden)
            item_correct, item_total = _classification_precision(torch, byte_logits, target)
            correct += item_correct
            total += item_total
            hidden = hidden.detach()
    return correct, total


def _role_precision_counts(torch, model, data, targets, sequence_bytes, device) -> tuple[int, int]:
    if not data:
        return 0, 0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(data), sequence_bytes):
            end = min(start + sequence_bytes, len(data))
            ids = torch.tensor([[BOS] + data[start:end]], dtype=torch.long, device=device)
            target = torch.tensor([targets[start:end]], dtype=torch.long, device=device)
            role_logits, _language_logits = model(ids)
            item_correct, item_total = _classification_precision(torch, role_logits[:, 1:], target)
            correct += item_correct
            total += item_total
    return correct, total


def _symbol_precision_counts(torch, model, data, targets, sequence_bytes, device) -> tuple[int, int]:
    if not data:
        return 0, 0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(data), sequence_bytes):
            end = min(start + sequence_bytes, len(data))
            ids = torch.tensor([[BOS] + data[start:end]], dtype=torch.long, device=device)
            target = torch.tensor([targets[start:end]], dtype=torch.long, device=device)
            occurrence_logits, _kind_logits, _language_logits = model(ids)
            item_correct, item_total = _classification_precision(torch, occurrence_logits[:, 1:], target)
            correct += item_correct
            total += item_total
    return correct, total


def _precision(correct: int, total: int) -> float | None:
    return None if total == 0 else correct / total


def _completion_step(torch, functional, model, optimizer, data, language, sequence_bytes, device):
    if not data:
        return 0.0
    model.train()
    hidden = None
    total = 0.0
    chunks = 0
    language_target = torch.tensor([language], dtype=torch.long, device=device)
    for start in range(0, len(data), sequence_bytes):
        end = min(start + sequence_bytes, len(data))
        target_values = data[start:end]
        if start == 0:
            input_values = [BOS] + target_values[:-1]
        else:
            input_values = data[start - 1:end - 1]
        ids = torch.tensor([input_values], dtype=torch.long, device=device)
        target = torch.tensor([target_values], dtype=torch.long, device=device)
        boundary = torch.tensor([_boundary_targets(target_values)], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        byte_logits, boundary_logits, language_logits, hidden = model(ids, hidden)
        loss = functional.cross_entropy(byte_logits.reshape(-1, COMPLETION_VOCABULARY), target.reshape(-1))
        loss = loss + 0.10 * functional.binary_cross_entropy_with_logits(boundary_logits, boundary)
        loss = loss + 0.05 * functional.cross_entropy(language_logits, language_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        hidden = hidden.detach()
        total += float(loss.detach().cpu())
        chunks += 1

    # Teach explicit document termination without retaining any source.
    ids = torch.tensor([[data[-1]]], dtype=torch.long, device=device)
    target = torch.tensor([EOS], dtype=torch.long, device=device)
    optimizer.zero_grad(set_to_none=True)
    byte_logits, _boundary_logits, _language_logits, _hidden = model(ids, hidden)
    loss = functional.cross_entropy(byte_logits[:, -1], target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return (total + float(loss.detach().cpu())) / max(chunks + 1, 1)


def _export_model(model, path: Path, *, task: str, status: str, args, losses: dict) -> None:
    metadata = {
        "taskId": task,
        "architecture": "SYL2-neural-v1",
        "status": status,
        "registryVersion": REGISTRY_VERSION,
        "registryHash": REGISTRY_HASH,
        "languages": list(LANGUAGES),
        "input": {"encoding": "utf-8", "vocabulary": "bytes+bos+eos+pad", "byteVocabularySize": BYTE_VOCABULARY},
        "output": {"floatType": "float32", "sequenceBytes": args.sequence_bytes},
        "training": {
            "epochsRequested": args.epochs,
            "epochsCompleted": args.epochs_completed,
            "learningRate": args.learning_rate,
            "seed": args.seed,
            "losses": losses,
            "earlyStopping": args.early_stop_summary,
        },
        "provenance": {"ledger": args.ledger, "retention": "source-not-retained"},
        "coverage": {"sourceCount": args.source_count, "languageCounts": args.language_counts},
    }
    if task == "identifier-relation":
        metadata["semanticSupervision"] = "unavailable; occurrence labels are weak bootstrap labels; link head is exported but untrained"
    write_syl2(path, metadata, model.state_dict())


def _split_specs(specs: Iterable) -> tuple[list, list]:
    specs = list(specs)
    training = [item for item in specs if (item.split or "train").lower() not in ("validation", "valid", "val", "dev")]
    validation = [item for item in specs if (item.split or "train").lower() in ("validation", "valid", "val", "dev")]
    if validation or len(training) < 5:
        return training, validation
    validation = [item for index, item in enumerate(training) if index % 10 == 0]
    training = [item for index, item in enumerate(training) if index % 10 != 0]
    return training, validation


def _evaluate_sources(args, torch, functional, causal, roles, symbols, validation_specs, tasks, device) -> dict:
    counts = {
        "completion": [0, 0],
        "roles": [0, 0],
        "symbols": [0, 0],
    }
    with SourceLedger(args.ledger) as ledger:
        for source in iter_sources(validation_specs, args.max_bytes, ledger, args.license_policy):
            language_name = normalize_language(source.spec.language or "text")
            language = language_index(language_name)
            data = _byte_ids(source.text)
            labels = {"tasks": sorted(tasks), "phase": "validation", "bytes": len(data)}
            if "completion" in tasks:
                correct, total = _completion_precision_counts(torch, functional, causal, data, language,
                                                              args.sequence_bytes, device)
                counts["completion"][0] += correct
                counts["completion"][1] += total
            if "roles" in tasks:
                correct, total = _role_precision_counts(torch, roles, data, _role_targets(source.text, language_name),
                                                        args.sequence_bytes, device)
                counts["roles"][0] += correct
                counts["roles"][1] += total
            if "symbols" in tasks:
                correct, total = _symbol_precision_counts(torch, symbols, data,
                                                          _occurrence_targets(source.text, language_name),
                                                          args.sequence_bytes, device)
                counts["symbols"][0] += correct
                counts["symbols"][1] += total
            ledger.record(
                source,
                "USED",
                labels=labels,
                teacher_versions={"validation": "streamed-heldout-v1"},
            )
            del data, source
    metrics = {}
    aggregate_correct = 0
    aggregate_total = 0
    for task, (correct, total) in counts.items():
        if task not in tasks:
            continue
        metrics[task] = {"precision": _precision(correct, total), "correct": correct, "total": total}
        aggregate_correct += correct
        aggregate_total += total
    metrics["aggregate"] = {"precision": _precision(aggregate_correct, aggregate_total),
                            "correct": aggregate_correct, "total": aggregate_total}
    return metrics


def _should_stop(history: list[float], args) -> tuple[bool, str | None]:
    if not history:
        return False, None
    latest = history[-1]
    if latest >= args.target_precision:
        return True, "target_precision"
    if len(history) < max(3, args.min_epochs):
        return False, None
    if len(history) < args.early_stop_patience + 2:
        return False, None
    if latest < args.plateau_precision_gate:
        return False, None
    window = history[-(args.early_stop_patience + 2):]
    improvements = [window[index] - window[index - 1] for index in range(1, len(window))]
    curvatures = [improvements[index] - improvements[index - 1] for index in range(1, len(improvements))]
    plateau = max(improvements[-args.early_stop_patience:]) <= args.min_precision_delta
    flat_curve = max(abs(value) for value in curvatures[-args.early_stop_patience:]) <= args.curvature_threshold
    if plateau and flat_curve:
        return True, "precision_plateau"
    return False, None


def train_sources(args) -> dict:
    random.seed(args.seed)
    torch, functional, causal, roles, symbols = _make_models(args.seed)
    device = torch.device(args.device)
    causal.to(device)
    roles.to(device)
    symbols.to(device)
    tasks = set(args.tasks)
    causal_optimizer = torch.optim.AdamW(causal.parameters(), lr=args.learning_rate) if "completion" in tasks else None
    roles_optimizer = torch.optim.AdamW(roles.parameters(), lr=args.learning_rate) if "roles" in tasks else None
    symbols_optimizer = torch.optim.AdamW(symbols.parameters(), lr=args.learning_rate) if "symbols" in tasks else None
    training_specs, validation_specs = _split_specs(args.specs)
    if not training_specs:
        raise ValueError("no training source was selected; check manifest split values")
    source_count = 0
    language_counts: dict[str, int] = {}
    losses = {"completion": [], "roles": [], "symbols": []}
    validation_history = []
    stop_reason = None
    with SourceLedger(args.ledger) as ledger:
        for epoch in range(args.epochs):
            for source in iter_sources(training_specs, args.max_bytes, ledger, args.license_policy):
                language_name = normalize_language(source.spec.language or "text")
                language = language_index(language_name)
                data = _byte_ids(source.text)
                labels = {"tasks": sorted(tasks), "phase": "train", "epoch": epoch + 1, "bytes": len(data)}
                if roles_optimizer is not None and data:
                    value = _role_step(torch, functional, roles, roles_optimizer, data,
                                       _role_targets(source.text, language_name), language,
                                       args.sequence_bytes, device)
                    losses["roles"].append(value)
                if symbols_optimizer is not None and data:
                    value = _symbol_step(torch, functional, symbols, symbols_optimizer, data,
                                         _occurrence_targets(source.text, language_name), language,
                                         args.sequence_bytes, device)
                    losses["symbols"].append(value)
                if causal_optimizer is not None:
                    value = _completion_step(torch, functional, causal, causal_optimizer, data,
                                             language, args.sequence_bytes, device)
                    losses["completion"].append(value)
                ledger.record(
                    source,
                    "USED",
                    labels=labels,
                    teacher_versions={
                        "language": "manifest-or-extension-v1",
                        "roles": "legacy-tokenizer-weak-v1" if roles_optimizer else None,
                        "symbols": "legacy-tokenizer-weak-v1" if symbols_optimizer else None,
                        "completion": "raw-byte-self-supervised-v1" if causal_optimizer else None,
                    },
                )
                source_count += 1
                language_counts[language_name] = language_counts.get(language_name, 0) + 1
                del data, source
            if validation_specs:
                metrics = _evaluate_sources(args, torch, functional, causal, roles, symbols,
                                            validation_specs, tasks, device)
                aggregate = metrics["aggregate"]["precision"]
                if aggregate is not None:
                    validation_history.append({"epoch": epoch + 1, "metrics": metrics})
                    should_stop, reason = _should_stop([item["metrics"]["aggregate"]["precision"]
                                                        for item in validation_history], args)
                    if should_stop:
                        stop_reason = reason
                        break
    if source_count == 0:
        raise ValueError("no source was used; check licenseId and the source manifest")

    args.source_count = source_count
    args.language_counts = language_counts
    args.epochs_completed = validation_history[-1]["epoch"] if validation_history else args.epochs
    args.early_stop_summary = {
        "targetPrecision": args.target_precision,
        "plateauPrecisionGate": args.plateau_precision_gate,
        "curvatureThreshold": args.curvature_threshold,
        "minPrecisionDelta": args.min_precision_delta,
        "minEpochs": args.min_epochs,
        "patience": args.early_stop_patience,
        "stopReason": stop_reason,
        "validationAvailable": bool(validation_specs),
        "history": validation_history,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if causal_optimizer is not None:
        path = output_dir / "next-word.model.bin"
        _export_model(causal.cpu(), path, task="next-word", status="SELF_SUPERVISED", args=args,
                      losses={"mean": sum(losses["completion"]) / max(len(losses["completion"]), 1),
                              "steps": len(losses["completion"])})
        outputs["completion"] = str(path)
    if roles_optimizer is not None:
        path = output_dir / "token-role.model.bin"
        _export_model(roles.cpu(), path, task="token-role", status="WEAKLY_SUPERVISED", args=args,
                      losses={"mean": sum(losses["roles"]) / max(len(losses["roles"]), 1),
                              "steps": len(losses["roles"])})
        outputs["roles"] = str(path)
    if symbols_optimizer is not None:
        path = output_dir / "identifier-relation.model.bin"
        _export_model(symbols.cpu(), path, task="identifier-relation", status="PARTIAL", args=args,
                      losses={"mean": sum(losses["symbols"]) / max(len(losses["symbols"]), 1),
                              "steps": len(losses["symbols"])})
        outputs["symbols"] = str(path)
    return {
        "trainer": "syl2",
        "sources": source_count,
        "languages": language_counts,
        "validation": args.early_stop_summary,
        "outputs": outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="streaming multilingual SYL2 trainer/exporter")
    parser.add_argument("--source", action="append", default=[], help="local/raw HTTP file; repeatable")
    parser.add_argument("--manifest", help="JSONL source metadata manifest")
    parser.add_argument("--license-id", help="license for direct --source entries")
    parser.add_argument("--license-policy", choices=("require", "allow"), default="require")
    parser.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--sequence-bytes", type=int, default=1024)
    parser.add_argument("--ledger", default="source-use-syl2.jsonl")
    parser.add_argument("--output-dir", default="syl2")
    parser.add_argument("--tasks", nargs="+", choices=("all", "completion", "roles", "symbols"), default=["all"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--plateau-precision-gate", type=float, default=0.80,
                        help="minimum validation precision before curvature plateau can stop training")
    parser.add_argument("--curvature-threshold", type=float, default=1e-5,
                        help="absolute second derivative threshold for validation precision plateau detection")
    parser.add_argument("--min-precision-delta", type=float, default=1e-4,
                        help="minimum validation precision improvement considered meaningful")
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sequence_bytes < 1 or args.max_bytes < 1:
        raise SystemExit("--sequence-bytes and --max-bytes must be positive")
    if not 0.0 < args.target_precision <= 1.0:
        raise SystemExit("--target-precision must be in (0, 1]")
    if not 0.0 < args.plateau_precision_gate <= args.target_precision:
        raise SystemExit("--plateau-precision-gate must be in (0, --target-precision]")
    if args.curvature_threshold < 0.0 or args.min_precision_delta < 0.0:
        raise SystemExit("--curvature-threshold and --min-precision-delta must be non-negative")
    if args.min_epochs < 1 or args.early_stop_patience < 1:
        raise SystemExit("--min-epochs and --early-stop-patience must be positive")
    args.tasks = {"completion", "roles", "symbols"} if "all" in args.tasks else set(args.tasks)
    args.specs = source_specs(args.manifest, args.source, args.license_id)
    if args.epochs > 1 and any(item.uri == "-" for item in args.specs):
        raise SystemExit("stdin is one-shot; use --epochs 1 or a reacquirable source")
    print(json.dumps(train_sources(args), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
