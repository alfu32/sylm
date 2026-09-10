"""Small independent detectors and a ranker over their public output records."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .annotations import range_from, utf16_range_to_bytes
from .syl2_format import read_syl2, write_syl2
from .syl2_trainer import BOS, BYTE_VOCABULARY, SYMBOL_KIND_LABELS, _kind_index, _torch
from .registry import LANGUAGES, REGISTRY_HASH
from .semantic_context import SemanticContext

SCHEMA = "semantic-records-v1"
MAX_CANDIDATES = 512
BIO = ("none", "begin", "inside")
FEATURE_NAMES = ("null", "sameDocument", "sameName", "sameKind", "usageConfidence",
                 "definitionConfidence", "definitionBeforeUsage", "signedDistance") + tuple(
    f"{role}Kind:{kind}" for role in ("usage", "definition") for kind in SYMBOL_KIND_LABELS) + (
    "queryPresent", "usagePosition", "definitionPosition", "queryPosition",
    "usageToQuery", "absoluteUsageToQuery", "definitionToQuery", "absoluteDefinitionToQuery")


@dataclass(frozen=True)
class SymbolRecord:
    id: str
    document_id: str
    revision: str
    start: int
    end: int
    name: str
    kind: int
    confidence: float
    context_end: int | None = None
    context: SemanticContext | None = None

    def __post_init__(self):
        if not self.id or not self.document_id or not self.revision or not 0 <= self.start <= self.end:
            raise ValueError("invalid symbol identity or range")
        if not 0 <= self.kind < len(SYMBOL_KIND_LABELS) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid symbol kind/confidence")
        if self.context_end is not None and self.context_end < self.end:
            raise ValueError("symbol is outside the analyzed context")


def make_detector(seed=17):
    torch, nn, _ = _torch()
    torch.manual_seed(seed)

    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(BYTE_VOCABULARY, 64, padding_idx=258)
            self.encoder = nn.GRU(64, 64, num_layers=2, bidirectional=True, batch_first=True)
            self.occurrence_head = nn.Linear(128, 3)
            self.kind_head = nn.Linear(128, len(SYMBOL_KIND_LABELS))
            self.language_head = nn.Linear(128, len(LANGUAGES))

        def forward(self, ids):
            hidden, _ = self.encoder(self.embedding(ids))
            return self.occurrence_head(hidden), self.kind_head(hidden), self.language_head(hidden.mean(1))

    return Detector()


def make_intelligence(seed=19):
    torch, nn, _ = _torch()
    torch.manual_seed(seed)

    class Intelligence(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(len(FEATURE_NAMES), 64)
            self.score = nn.Linear(64, 1)

        def forward(self, features):
            return self.score(torch.tanh(self.hidden(features))).squeeze(-1)

    return Intelligence()


def boundaries(text):
    """(UTF-8 byte offset, UTF-16 code-unit offset) at every code-point boundary."""
    result = [(0, 0)]
    byte = unit = 0
    for char in text:
        byte += len(char.encode("utf-8"))
        unit += 2 if ord(char) > 0xFFFF else 1
        result.append((byte, unit))
    return result


def windows(text, sequence_bytes):
    if sequence_bytes < 4:
        raise ValueError("semantic windows must fit a complete UTF-8 code point (>=4 bytes)")
    points = boundaries(text)
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text) and points[end + 1][0] - points[start][0] <= sequence_bytes:
            end += 1
        yield start, end, points[start][0], points[end][0]
        start = end


def detector_targets(text, annotation, role):
    """Partial annotation never implies that all other source bytes are negatives."""
    size = len(text.encode("utf-8"))
    complete = annotation.teacher.get("occurrenceCoverage") == "complete"
    tags, kinds = [0 if complete else -100] * size, [-100] * size
    own = annotation.definitions if role == "definitions" else annotation.usages
    other = annotation.usages if role == "definitions" else annotation.definitions
    for item in other:
        a, b = range_from(item, "nameRange" if role == "usages" else "range")
        a, b = utf16_range_to_bytes(text, a, b)
        tags[a:b] = [0] * (b - a)
    for item in own:
        a, b = range_from(item, "nameRange" if role == "definitions" else "range")
        a, b = utf16_range_to_bytes(text, a, b)
        if a == b:
            raise ValueError("symbol annotations must have nonempty ranges")
        tags[a:b] = [2] * (b - a)
        tags[a] = 1
        kinds[a:b] = [_kind_index(item.get("kind"))] * (b - a)
    return tags, kinds


def detect(model, text, document_id, revision, role, sequence_bytes=512):
    torch, _, _ = _torch()
    device = next(model.parameters()).device
    points = boundaries(text)
    raw = text.encode("utf-8")
    # Per-character BIO decisions keep UTF-16 spans out of surrogate pairs.
    decisions, probabilities, kind_scores, language_scores = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for a, b, ba, bb in windows(text, sequence_bytes):
            tags, kinds, languages = model(torch.tensor([[BOS] + list(raw[ba:bb])], device=device))
            probs = tags[0, 1:].softmax(-1).cpu()
            kinds = kinds[0, 1:].cpu()
            language_scores.append((languages[0].softmax(-1).cpu(), b - a))
            for index in range(a, b):
                pos = points[index][0] - ba
                label = int(probs[pos].argmax())
                decisions.append(label)
                probabilities.append(float(probs[pos, label]))
                kind_scores.append(kinds[pos])
    records = []
    start = None
    for index in range(len(text) + 1):
        label = decisions[index] if index < len(text) else 0
        if start is not None and label != 2:
            a, b = points[start][1], points[index][1]
            records.append(SymbolRecord(f"{role}:{a}:{b}", document_id, revision, a, b,
                                        text[start:index], int(torch.stack(kind_scores[start:index]).mean(0).argmax()),
                                        sum(probabilities[start:index]) / (index - start), points[-1][1]))
            start = None
        if index < len(text) and label and start is None:
            start = index
    language = (sum(value * count for value, count in language_scores) / max(len(text), 1)).tolist() \
        if language_scores else [0.0] * len(LANGUAGES)
    return records, language


def pair_features(usage, definition, query_position=None):
    if query_position is not None and (type(query_position) is not int or query_position < 0):
        raise ValueError("query position must be a nonnegative UTF-16 offset")
    if definition is not None and usage.document_id == definition.document_id and usage.revision != definition.revision:
        raise ValueError("stale definition revision")
    same_doc = definition is not None and usage.document_id == definition.document_id
    distance = max(-1.0, min(1.0, (definition.start - usage.start) / 4096.0)) if same_doc else 0.0
    relative = lambda position: max(-1., min(1., (position - query_position) / 4096.)) if query_position is not None else 0.
    bounded = lambda position: position / (position + 4096.)
    u_distance = relative(usage.start)
    d_distance = relative(definition.start) if same_doc else 0.
    return [float(definition is None), float(same_doc),
            float(definition is not None and usage.name == definition.name),
            float(definition is not None and usage.kind == definition.kind), usage.confidence,
            definition.confidence if definition else 0.0,
            float(same_doc and definition.start <= usage.start), distance] + \
        [float(usage.kind == k) for k in range(len(SYMBOL_KIND_LABELS))] + \
        [float(definition is not None and definition.kind == k) for k in range(len(SYMBOL_KIND_LABELS))] + \
        [float(query_position is not None), bounded(usage.start), bounded(definition.start) if same_doc else 0.,
         bounded(query_position) if query_position is not None else 0.,
         u_distance, abs(u_distance), d_distance, abs(d_distance)]


def candidate_features(usage, definitions, query_position=None):
    if len(definitions) > MAX_CANDIDATES:
        raise ValueError("candidate context exceeds 512 definitions; narrow the query context")
    identities = [(d.document_id, d.revision, d.id) for d in definitions]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate definition candidate identity")
    return [pair_features(usage, definition, query_position) for definition in definitions] + [pair_features(usage, None, query_position)]


def rank(model, usage, definitions, threshold=0.8, query_position=None, completion_context=False):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if completion_context and (query_position is None or usage.end > query_position or
                               usage.context_end is None or usage.context_end > query_position or
                               any(d.document_id == usage.document_id and
                                   (d.end > query_position or d.context_end is None or d.context_end > query_position)
                                   for d in definitions)):
        raise ValueError("completion context contains records after the cursor")
    torch, _, _ = _torch()
    model.eval()
    with torch.inference_mode():
        scores = model(torch.tensor(candidate_features(usage, definitions, query_position), dtype=torch.float32,
                                    device=next(model.parameters()).device)).softmax(0).cpu()
    best = int(scores.argmax())
    confidence = float(scores[best])
    target = definitions[best] if best < len(definitions) and confidence >= threshold else None
    return {"usage": asdict(usage), "definition": asdict(target) if target else None,
            "confidence": confidence, "status": "resolved" if target else "unresolved",
            "candidateProbabilities": scores.tolist()}


def export(path, model, task, *, sequence_bytes=512, trained_steps=0, dependencies=None):
    write_syl2(path, {"schema": SCHEMA, "taskId": task, "registryHash": REGISTRY_HASH,
                     "languages": list(LANGUAGES), "sequenceBytes": sequence_bytes,
                     "bioLabels": list(BIO), "symbolKinds": list(SYMBOL_KIND_LABELS),
                     "featureNames": list(FEATURE_NAMES), "trainedSteps": trained_steps,
                     "status": "ANNOTATION_SUPERVISED" if trained_steps else "UNTRAINED",
                     "dependencies": dependencies or {},
                     "parameterCount": sum(p.numel() for p in model.parameters())}, model.state_dict())


def load(path, expected_task):
    if expected_task not in ("definitions", "usages", "code-intelligence"):
        raise ValueError("unknown semantic task")
    import numpy as np
    torch, _, _ = _torch()
    metadata, tensors = read_syl2(path)
    if metadata.get("schema") != SCHEMA or metadata.get("taskId") != expected_task:
        raise ValueError("incompatible semantic artifact")
    if metadata.get("registryHash") != REGISTRY_HASH or metadata.get("featureNames") != list(FEATURE_NAMES):
        raise ValueError("semantic registry or feature contract mismatch")
    model = make_intelligence() if expected_task == "code-intelligence" else make_detector()
    model.load_state_dict({name: torch.from_numpy(np.frombuffer(payload, dtype="<f4").copy().reshape(shape))
                           for name, (shape, payload) in tensors.items()}, strict=True)
    return model.eval(), metadata


def artifact_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
