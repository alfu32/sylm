#!/usr/bin/env python3
"""Train SYL2 artifacts from bounded multilingual source streams.

Raw source remains self-supervision for completion.  When an annotation
manifest is supplied, role and identifier heads use gold UTF-16 ranges and the
identifier model also trains its dynamic usage-to-definition link head.  Files
without a matching annotation record may still be used for bootstrap losses,
but their status is recorded as weak/partial rather than being presented as
semantic supervision.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from syntaxlm import normalize_language, tokenize_code

from .annotations import (
    AnnotationRecord,
    annotation_for,
    annotation_index,
    canonical_role,
    load_annotations,
    range_from,
    utf16_range_to_bytes,
)
from .registry import LANGUAGES, REGISTRY_HASH, REGISTRY_VERSION, language_index
from .streaming_sources import SourceLedger, iter_sources, source_specs
from .syl2_format import write_syl2


BYTE_VOCABULARY = 259
BOS = 256
EOS = 257
PAD = 258
# 256 byte values plus BOS/EOS are input IDs; completion predicts bytes plus
# EOS. BOS is never a completion target, so the output IDs are 0..257.
COMPLETION_VOCABULARY = 258
ROLE_LABELS = (
    "plain", "keyword", "identifier", "string_literal", "number_literal",
    "comment", "operator", "punctuation", "array_literal", "object_literal",
    "boolean_literal", "null_literal", "unknown", "decorator",
)
OCCURRENCE_LABELS = ("none", "definition", "usage", "both", "unknown")
SYMBOL_KIND_LABELS = ("unknown", "variable", "function", "type", "field", "parameter", "module", "constant")


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


def _annotated_role_targets(text: str, record: AnnotationRecord) -> tuple[list[int], int]:
    """Return masked byte targets and the number of labeled bytes."""
    labels = [-100] * len(text.encode("utf-8"))
    covered = 0
    for span in record.role_spans:
        start, end = range_from(span)
        start_byte, end_byte = utf16_range_to_bytes(text, start, end)
        role = canonical_role(str(span.get("role", span.get("label", span.get("kind", "unknown")))))
        if role not in ROLE_LABELS:
            role = "unknown"
        for index in range(start_byte, end_byte):
            if labels[index] == -100:
                covered += 1
            labels[index] = ROLE_LABELS.index(role)
    return labels, covered


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


def _kind_index(value: str | None) -> int:
    value = (value or "unknown").lower().replace("-", "_")
    aliases = {"function_declaration": "function", "class": "type", "property": "field"}
    return SYMBOL_KIND_LABELS.index(aliases.get(value, value) if aliases.get(value, value) in SYMBOL_KIND_LABELS else "unknown")


def _annotated_symbol_targets(text: str, record: AnnotationRecord) -> tuple[list[int], list[int], list[dict], dict[str, tuple[int, int]]]:
    """Build masked occurrence/kind byte targets and byte-coordinate links."""
    size = len(text.encode("utf-8"))
    occurrence = [-100] * size
    kinds = [-100] * size
    definitions: dict[str, tuple[int, int]] = {}
    definition_kinds: dict[str, int] = {}
    for item in record.definitions:
        definition_id = str(item.get("id", ""))
        if not definition_id:
            raise ValueError("definition annotation requires a non-empty id")
        name_start, name_end = range_from(item, "nameRange")
        start_byte, end_byte = utf16_range_to_bytes(text, name_start, name_end)
        if definition_id in definitions:
            raise ValueError(f"duplicate definition id: {definition_id}")
        definitions[definition_id] = (start_byte, end_byte)
        definition_kinds[definition_id] = _kind_index(item.get("kind"))
        for index in range(start_byte, end_byte):
            occurrence[index] = OCCURRENCE_LABELS.index("definition")
            kinds[index] = definition_kinds[definition_id]
    links: list[dict] = []
    for item in record.usages:
        start, end = range_from(item)
        start_byte, end_byte = utf16_range_to_bytes(text, start, end)
        occurrence_value = str(item.get("occurrence", "usage"))
        if occurrence_value not in {"definition", "usage", "both", "unknown"}:
            raise ValueError(f"unsupported occurrence label: {occurrence_value}")
        kind = _kind_index(item.get("kind"))
        for index in range(start_byte, end_byte):
            occurrence[index] = OCCURRENCE_LABELS.index(occurrence_value)
            kinds[index] = kind
        links.append({
            "start": start_byte,
            "end": end_byte,
            "definitionId": item.get("definitionId"),
            "status": item.get("status", "resolved" if item.get("definitionId") else "unresolved"),
        })
    return occurrence, kinds, links, definitions


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
            self.link_null = nn.Parameter(torch.zeros(128))

        def encode(self, ids):
            output, _hidden = self.encoder(self.embedding(ids))
            return output

        def forward(self, ids):
            output = self.encode(ids)
            return (
                self.occurrence_head(output),
                self.symbol_kind_head(output),
                self.language_head(output.mean(dim=1)),
            )

        def score_link(self, usage_vector, definition_vector, features):
            pair = torch.cat((usage_vector, definition_vector, usage_vector * definition_vector, features), dim=-1)
            return self.link_score(torch.tanh(self.link_hidden(pair))).squeeze(-1)

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
        loss = 0.10 * functional.cross_entropy(language_logits, language_target)
        if bool((target != -100).any()):
            loss = loss + functional.cross_entropy(
                role_logits[:, 1:].reshape(-1, len(ROLE_LABELS)), target.reshape(-1), ignore_index=-100
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        chunks += 1
    return total / max(chunks, 1)


def _link_features(torch, usage_start: int, definition_start: int, source_size: int, *, null_candidate: bool = False):
    distance = (definition_start - usage_start) / max(source_size, 1)
    return torch.tensor([
        1.0 if null_candidate else 0.0,
        distance, abs(distance),
        min(abs(definition_start - usage_start), 1024) / 1024.0,
        1.0 if definition_start <= usage_start else 0.0,
        1.0 if definition_start > usage_start else 0.0,
        1.0 if usage_start == definition_start else 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=torch.float32)


def _link_loss(torch, functional, model, output, links, definitions, chunk_start, chunk_end, source_size, device):
    if not links or not definitions:
        return None, 0
    local_definitions = {
        definition_id: position for definition_id, position in definitions.items()
        if chunk_start <= position[0] < chunk_end
    }
    if not local_definitions:
        return None, 0
    losses = []
    evaluated = 0
    ordered = list(local_definitions.items())
    for link in links:
        usage_start = int(link["start"])
        if not chunk_start <= usage_start < chunk_end:
            continue
        target_id = link.get("definitionId")
        if target_id not in local_definitions:
            # A document-level candidate encoder is a later optimization. Do
            # not turn a cross-window link into a false null target.
            continue
        usage_vector = output[0, usage_start - chunk_start + 1]
        scores = []
        candidate_ids = []
        for definition_id, (definition_start, _definition_end) in ordered:
            definition_vector = output[0, definition_start - chunk_start + 1]
            features = _link_features(torch, usage_start, definition_start, source_size).to(device)
            scores.append(model.score_link(usage_vector, definition_vector, features))
            candidate_ids.append(definition_id)
        null_features = _link_features(torch, usage_start, usage_start, source_size, null_candidate=True).to(device)
        scores.append(model.score_link(usage_vector, model.link_null, null_features))
        candidate_ids.append(None)
        target_index = candidate_ids.index(target_id)
        losses.append(functional.cross_entropy(torch.stack(scores).reshape(1, -1), torch.tensor([target_index], device=device)))
        evaluated += 1
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), evaluated


def _symbol_step(torch, functional, model, optimizer, data, targets, language, sequence_bytes, device, *, links=None, definitions=None):
    model.train()
    total = 0.0
    chunks = 0
    language_target = torch.tensor([language], dtype=torch.long, device=device)
    if isinstance(targets, tuple):
        occurrence_targets, kind_targets, link_targets, definition_targets = targets
    else:
        occurrence_targets, kind_targets, link_targets, definition_targets = targets, None, links or [], definitions or {}
    for start in range(0, len(data), sequence_bytes):
        end = min(start + sequence_bytes, len(data))
        ids = torch.tensor([[BOS] + data[start:end]], dtype=torch.long, device=device)
        target = torch.tensor([occurrence_targets[start:end]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model.encode(ids)
        occurrence_logits = model.occurrence_head(output)
        kind_logits = model.symbol_kind_head(output)
        language_logits = model.language_head(output.mean(dim=1))
        loss = 0.0 * language_logits.sum()
        if bool((target != -100).any()):
            loss = loss + functional.cross_entropy(
                occurrence_logits[:, 1:].reshape(-1, len(OCCURRENCE_LABELS)), target.reshape(-1), ignore_index=-100
            )
        if kind_targets is not None:
            kind_target = torch.tensor([kind_targets[start:end]], dtype=torch.long, device=device)
            if bool((kind_target != -100).any()):
                loss = loss + functional.cross_entropy(
                    kind_logits[:, 1:].reshape(-1, len(SYMBOL_KIND_LABELS)), kind_target.reshape(-1), ignore_index=-100
                )
            link_loss, _evaluated = _link_loss(
                torch, functional, model, output, link_targets, definition_targets,
                start, end, len(data), device,
            )
            if link_loss is not None:
                loss = loss + link_loss
        loss = loss + 0.10 * functional.cross_entropy(language_logits, language_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        chunks += 1
    return total / max(chunks, 1)


def _classification_precision(torch, logits, target) -> tuple[int, int]:
    valid = target != -100
    if target.numel() == 0 or not bool(valid.any()):
        return 0, 0
    prediction = torch.argmax(logits, dim=-1)
    prediction = prediction.reshape(-1)[valid.reshape(-1)]
    expected = target.reshape(-1)[valid.reshape(-1)]
    correct = int((prediction == expected).sum().detach().cpu())
    return correct, int(expected.numel())


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
    if isinstance(targets, tuple):
        targets = targets[0]
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


def _supervision_for_source(text: str, language: str, record: AnnotationRecord | None):
    if record is None:
        return (
            _role_targets(text, language),
            _occurrence_targets(text, language),
            None,
            None,
            "weak-bootstrap-v1",
        )
    role_targets, role_coverage = _annotated_role_targets(text, record)
    occurrence, kinds, links, definitions = _annotated_symbol_targets(text, record)
    resolved_links = sum(1 for link in links if link.get("definitionId") in definitions)
    return (role_targets, (occurrence, kinds, links, definitions), role_coverage,
            resolved_links, record.teacher.get("version", record.teacher.get("name", "annotated")))


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
        "supervision": args.supervision_summary,
    }
    if task == "identifier-relation" and status != "SUPERVISED":
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


def _link_precision_counts(torch, model, data, links, definitions, sequence_bytes, device) -> tuple[int, int]:
    if not links or not definitions or not data:
        return 0, 0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(data), sequence_bytes):
            end = min(start + sequence_bytes, len(data))
            local_definitions = {
                definition_id: position for definition_id, position in definitions.items()
                if start <= position[0] < end
            }
            if not local_definitions:
                continue
            ids = torch.tensor([[BOS] + data[start:end]], dtype=torch.long, device=device)
            output = model.encode(ids)
            ordered = list(local_definitions.items())
            for link in links:
                usage_start = int(link["start"])
                target_id = link.get("definitionId")
                if not start <= usage_start < end or target_id not in local_definitions:
                    continue
                usage_vector = output[0, usage_start - start + 1]
                scores = []
                candidate_ids = []
                for definition_id, (definition_start, _definition_end) in ordered:
                    definition_vector = output[0, definition_start - start + 1]
                    features = _link_features(torch, usage_start, definition_start, len(data)).to(device)
                    scores.append(model.score_link(usage_vector, definition_vector, features))
                    candidate_ids.append(definition_id)
                null_features = _link_features(torch, usage_start, usage_start, len(data), null_candidate=True).to(device)
                scores.append(model.score_link(usage_vector, model.link_null, null_features))
                candidate_ids.append(None)
                predicted = candidate_ids[int(torch.argmax(torch.stack(scores)).detach().cpu())]
                correct += int(predicted == target_id)
                total += 1
    return correct, total


def _evaluate_sources(args, torch, functional, causal, roles, symbols, validation_specs, tasks, device, record_index) -> dict:
    counts = {
        "completion": [0, 0],
        "roles": [0, 0],
        "symbols": [0, 0],
        "links": [0, 0],
    }
    with SourceLedger(args.ledger) as ledger:
        for source in iter_sources(validation_specs, args.max_bytes, ledger, args.license_policy):
            language_name = normalize_language(source.spec.language or "text")
            language = language_index(language_name)
            data = _byte_ids(source.text)
            annotation = annotation_for(record_index, source.spec)
            if annotation is not None and annotation.content_sha256 and annotation.content_sha256 != source.content_sha256:
                ledger.record(source, "FAILED", reason="annotation contentSha256 does not match reacquired source",
                              labels={"phase": "validation"})
                del data, source
                continue
            role_targets, symbol_targets, _role_coverage, _link_count, _teacher = _supervision_for_source(
                source.text, language_name, annotation
            )
            labels = {"tasks": sorted(tasks), "phase": "validation", "bytes": len(data)}
            if "completion" in tasks:
                correct, total = _completion_precision_counts(torch, functional, causal, data, language,
                                                              args.sequence_bytes, device)
                counts["completion"][0] += correct
                counts["completion"][1] += total
            if "roles" in tasks:
                correct, total = _role_precision_counts(torch, roles, data, role_targets,
                                                        args.sequence_bytes, device)
                counts["roles"][0] += correct
                counts["roles"][1] += total
            if "symbols" in tasks:
                correct, total = _symbol_precision_counts(torch, symbols, data,
                                                          symbol_targets,
                                                          args.sequence_bytes, device)
                counts["symbols"][0] += correct
                counts["symbols"][1] += total
                if annotation is not None and symbol_targets[2] and "symbols" in tasks:
                    correct, total = _link_precision_counts(
                        torch, symbols, data, symbol_targets[2], symbol_targets[3], args.sequence_bytes, device
                    )
                    counts["links"][0] += correct
                    counts["links"][1] += total
            ledger.record(
                source,
                "USED",
                labels=labels,
                teacher_versions={"validation": "streamed-heldout-v1",
                                  "annotation": _teacher if annotation else None},
            )
            del data, source
    metrics = {}
    aggregate_correct = 0
    aggregate_total = 0
    for task, (correct, total) in counts.items():
        if task == "links":
            if "symbols" not in tasks or total == 0:
                continue
        elif task not in tasks:
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
    annotated_sources = 0
    annotated_roles = 0
    annotated_links = 0
    validation_history = []
    stop_reason = None
    with SourceLedger(args.ledger) as ledger:
        for epoch in range(args.epochs):
            for source in iter_sources(training_specs, args.max_bytes, ledger, args.license_policy):
                language_name = normalize_language(source.spec.language or "text")
                language = language_index(language_name)
                data = _byte_ids(source.text)
                annotation = annotation_for(args.annotation_index, source.spec)
                if annotation is not None and annotation.content_sha256 and annotation.content_sha256 != source.content_sha256:
                    ledger.record(source, "FAILED", reason="annotation contentSha256 does not match reacquired source",
                                  labels={"phase": "train", "epoch": epoch + 1})
                    del data, source
                    continue
                role_targets, symbol_targets, role_coverage, link_count, teacher = _supervision_for_source(
                    source.text, language_name, annotation
                )
                if annotation is not None:
                    annotated_sources += 1
                    annotated_roles += int(role_coverage or 0)
                    annotated_links += int(link_count or 0)
                labels = {"tasks": sorted(tasks), "phase": "train", "epoch": epoch + 1, "bytes": len(data),
                          "supervision": "gold" if annotation else "weak-bootstrap"}
                if roles_optimizer is not None and data:
                    value = _role_step(torch, functional, roles, roles_optimizer, data,
                                       role_targets, language,
                                       args.sequence_bytes, device)
                    losses["roles"].append(value)
                if symbols_optimizer is not None and data:
                    value = _symbol_step(torch, functional, symbols, symbols_optimizer, data,
                                         symbol_targets, language, args.sequence_bytes, device)
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
                        "roles": teacher if roles_optimizer and annotation else ("legacy-tokenizer-weak-v1" if roles_optimizer else None),
                        "symbols": teacher if symbols_optimizer and annotation else ("legacy-tokenizer-weak-v1" if symbols_optimizer else None),
                        "completion": "raw-byte-self-supervised-v1" if causal_optimizer else None,
                    },
                )
                source_count += 1
                language_counts[language_name] = language_counts.get(language_name, 0) + 1
                del data, source
            if validation_specs:
                metrics = _evaluate_sources(args, torch, functional, causal, roles, symbols,
                                            validation_specs, tasks, device, args.annotation_index)
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
    args.supervision_summary = {
        "annotationManifest": args.annotation_manifest,
        "annotatedSourceUses": annotated_sources,
        "annotatedRoleBytes": annotated_roles,
        "annotatedLinkExamples": annotated_links,
        "roleStatus": "SUPERVISED" if annotated_roles else "WEAKLY_SUPERVISED",
        "identifierStatus": "SUPERVISED" if annotated_links else "PARTIAL",
        "teacherVersions": sorted({
            str(record.teacher.get("version", record.teacher.get("name", "annotated")))
            for record in args.annotation_index.values()
        }),
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
        _export_model(roles.cpu(), path, task="token-role",
                      status="SUPERVISED" if annotated_roles else "WEAKLY_SUPERVISED", args=args,
                      losses={"mean": sum(losses["roles"]) / max(len(losses["roles"]), 1),
                              "steps": len(losses["roles"])})
        outputs["roles"] = str(path)
    if symbols_optimizer is not None:
        path = output_dir / "identifier-relation.model.bin"
        _export_model(symbols.cpu(), path, task="identifier-relation",
                      status="SUPERVISED" if annotated_links else "PARTIAL", args=args,
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
    parser.add_argument("--annotation-manifest", "--annotations", dest="annotation_manifest",
                        help="JSONL gold role/definition/usage annotation manifest")
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
    annotation_records = load_annotations(args.annotation_manifest)
    args.annotation_index = annotation_index(annotation_records)
    args.specs = source_specs(args.manifest, args.source, args.license_id)
    if not args.specs and annotation_records:
        args.specs = [record.spec for record in annotation_records]
    if args.epochs > 1 and any(item.uri == "-" for item in args.specs):
        raise SystemExit("stdin is one-shot; use --epochs 1 or a reacquirable source")
    print(json.dumps(train_sources(args), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
