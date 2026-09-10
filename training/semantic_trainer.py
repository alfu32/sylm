"""Train independent detectors, then train intelligence on their frozen predictions.

Source is reacquired through the shared bounded stream. Only model artifacts,
annotation references, metrics and provenance hashes are retained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .annotations import annotation_for, annotation_index, load_annotations, range_from
from .registry import language_index
from .semantic_models import (
    MAX_CANDIDATES, artifact_hash, boundaries, candidate_features, detect, detector_targets, export, load,
    make_detector, make_intelligence, windows,
)
from .streaming_sources import SourceLedger, iter_sources, source_specs
from .syl2_trainer import BOS, _split_specs, _torch


def detector_step(model, optimizer, source, annotation, role, sequence_bytes):
    torch, _, functional = _torch()
    model.train()
    tags, kinds = detector_targets(source.text, annotation, role)
    raw = source.text.encode("utf-8")
    losses = []
    for _, _, start, end in windows(source.text, sequence_bytes):
        target = torch.tensor([tags[start:end]])
        if not bool((target != -100).any()):
            continue
        kind_target = torch.tensor([kinds[start:end]])
        optimizer.zero_grad(set_to_none=True)
        logits, kind_logits, language = model(torch.tensor([[BOS] + list(raw[start:end])]))
        loss = functional.cross_entropy(logits[:, 1:].reshape(-1, 3), target.flatten(), ignore_index=-100)
        if bool((kind_target != -100).any()):
            loss += .2 * functional.cross_entropy(kind_logits[:, 1:].reshape(-1, 8), kind_target.flatten(), ignore_index=-100)
        loss += .1 * functional.cross_entropy(language, torch.tensor([language_index(source.spec.language or "unknown")]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


class LinkExamples:
    """Materialize only one usage's candidate features at a time."""
    def __init__(self, definitions):
        self.definitions = tuple(definitions)
        self.targets = []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        usage, target, query = self.targets[index]
        return candidate_features(usage, self.definitions, query), target


def link_examples(usages, definitions, annotation):
    """Inputs are predicted public records; annotations select targets only.

    Missing upstream predictions are excluded from loss and counted. In
    particular a gold target absent from the candidate set is NOT a null label.
    """
    by_range = {(u.start, u.end): u for u in usages}
    def_ranges = {str(d["id"]): range_from(d, "nameRange") for d in annotation.definitions}
    predicted = {(d.start, d.end): index for index, d in enumerate(definitions)}
    counts = {"annotatedUsages": 0, "missingUsage": 0, "missingDefinition": 0,
              "unsupportedTarget": 0, "candidateOverflow": 0, "eligible": 0}
    examples = LinkExamples(definitions)
    for item in annotation.usages:
        counts["annotatedUsages"] += 1
        if len(definitions) > MAX_CANDIDATES:
            counts["candidateOverflow"] += 1
            continue
        usage = by_range.get(range_from(item))
        if usage is None:
            counts["missingUsage"] += 1
            continue
        target_id = item.get("definitionId")
        if target_id is None and item.get("status") == "no_definition":
            index = len(definitions)
        elif str(target_id) in def_ranges:
            index = predicted.get(def_ranges[str(target_id)])
            if index is None:
                counts["missingDefinition"] += 1
                continue
        else:
            counts["unsupportedTarget"] += 1
            continue
        # Query offset is an input; labels are never fed into candidate features.
        examples.targets.append((usage, index, item.get("queryPosition", usage.start)))
        counts["eligible"] += 1
    return examples, counts


def intelligence_step(model, optimizer, examples):
    torch, _, functional = _torch()
    losses = []
    model.train()
    for features, target in examples:
        optimizer.zero_grad(set_to_none=True)
        logits = model(torch.tensor(features, dtype=torch.float32))
        loss = functional.cross_entropy(logits[None], torch.tensor([target]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def validate_annotation(text, annotation):
    """Validate offsets even during intelligence-only training and evaluation."""
    valid = {unit for _, unit in boundaries(text)}
    for role, items in (("definitions", annotation.definitions), ("usages", annotation.usages)):
        for item in items:
            start, end = range_from(item, "nameRange" if role == "definitions" else "range")
            if start not in valid or end not in valid or start >= end:
                raise ValueError("symbol range is empty or not on source UTF-16 boundaries")
            if "queryPosition" in item:
                query = item["queryPosition"]
                if type(query) is not int or query not in valid:
                    raise ValueError("queryPosition must be on a source UTF-16 boundary")


def checked_sources(specs, args, records, phase):
    with SourceLedger(args.ledger) as ledger:
        for source in iter_sources(specs, args.max_bytes, ledger):
            annotation = annotation_for(records, source.spec)
            if annotation is None:
                ledger.record(source, "SKIPPED", reason="semantic training requires annotations")
                continue
            if annotation.content_sha256 and annotation.content_sha256 != source.content_sha256:
                raise ValueError(f"annotation hash mismatch for {source.spec.uri}")
            validate_annotation(source.text, annotation)
            yield source, annotation
            ledger.record(source, "USED", labels={"phase": phase}, teacher_versions=annotation.teacher)


def evaluate(detectors, specs, args, records, intelligence=None):
    torch, _, _ = _torch()
    counts = {role: dict(tp=0, predicted=0, gold=0, annotatedMatched=0, annotatedGold=0,
                         completeSources=0, partialSources=0) for role in detectors}
    links = dict(eligible=0, correct=0, annotatedUsages=0, missingUsage=0,
                 missingDefinition=0, unsupportedTarget=0, candidateOverflow=0)
    for source, annotation in checked_sources(specs, args, records, "validation"):
        outputs = {}
        for role, model in detectors.items():
            pred, _ = detect(model, source.text, source.spec.uri, source.content_sha256, role, args.sequence_bytes)
            outputs[role] = pred
            own = annotation.definitions if role == "definitions" else annotation.usages
            gold = {range_from(item, "nameRange" if role == "definitions" else "range") for item in own}
            got = {(p.start, p.end) for p in pred}
            c = counts[role]
            c["annotatedMatched"] += len(gold & got)
            c["annotatedGold"] += len(gold)
            if annotation.teacher.get("occurrenceCoverage") == "complete":
                c["completeSources"] += 1
                c["tp"] += len(gold & got)
                c["predicted"] += len(got)
                c["gold"] += len(gold)
            else:
                c["partialSources"] += 1
        if intelligence is not None:
            examples, coverage = link_examples(outputs["usages"], outputs["definitions"], annotation)
            for key, value in coverage.items():
                links[key] += value
            intelligence.eval()
            with torch.inference_mode():
                links["correct"] += sum(int(intelligence(torch.tensor(f, dtype=torch.float32)).argmax()) == target
                                        for f, target in examples)
    for count in counts.values():
        count["precision"] = count["tp"] / count["predicted"] if count["predicted"] else None
        count["recall"] = count["tp"] / count["gold"] if count["gold"] else None
        count["annotatedRecall"] = count["annotatedMatched"] / count["annotatedGold"] if count["annotatedGold"] else None
    if intelligence is not None:
        links["candidateTop1Accuracy"] = links["correct"] / links["eligible"] if links["eligible"] else None
        links["candidateCoverage"] = links["eligible"] / links["annotatedUsages"] if links["annotatedUsages"] else None
        counts["intelligence"] = links
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-manifest", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--tasks", nargs="+", choices=("all", "definitions", "usages", "intelligence"), default=["all"])
    parser.add_argument("--definitions-model")
    parser.add_argument("--usages-model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--intelligence-epochs", type=int, default=10)
    parser.add_argument("--sequence-bytes", type=int, default=512)
    parser.add_argument("--max-bytes", type=int, default=262144)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.epochs, args.intelligence_epochs, args.cpu_threads, args.max_bytes) < 1 or args.sequence_bytes < 4 or args.learning_rate <= 0:
        parser.error("invalid training limits")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty; existing runs are preserved")
    annotations = load_annotations(args.annotation_manifest)
    records = annotation_index(annotations)
    specs = source_specs(args.manifest, [], None) if args.manifest else [record.spec for record in annotations]
    train_specs, val_specs = _split_specs(specs)
    if not train_specs:
        parser.error("no training sources")
    tasks = {"definitions", "usages", "intelligence"} if "all" in args.tasks else set(args.tasks)
    torch, _, _ = _torch()
    torch.set_num_threads(args.cpu_threads)
    detectors, paths, steps = {}, {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for role in ("definitions", "usages"):
        supplied = getattr(args, role + "_model")
        if role in tasks:
            detectors[role] = make_detector(17 if role == "definitions" else 18)
            paths[role] = str(output / (role + ".model.bin"))
            steps[role] = 0
        elif supplied:
            detectors[role], metadata = load(supplied, role)
            if "intelligence" in tasks and metadata["trainedSteps"] <= 0:
                parser.error(f"intelligence requires a trained {role} artifact")
            if metadata["sequenceBytes"] != args.sequence_bytes:
                parser.error("detector window differs from --sequence-bytes")
            paths[role] = supplied
            steps[role] = metadata["trainedSteps"]
        elif "intelligence" in tasks:
            parser.error(f"intelligence needs --{role}-model or training task {role}")
    optimizers = {role: torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
                  for role, model in detectors.items() if role in tasks}

    def progress(event):
        print(json.dumps(event), file=sys.stderr, flush=True)
        with (output / "epochs.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")

    for epoch in range(args.epochs if optimizers else 0):
        losses = {role: [] for role in optimizers}
        for source, annotation in checked_sources(train_specs, args, records, "detectors"):
            for role, optimizer in optimizers.items():
                values = detector_step(detectors[role], optimizer, source, annotation, role, args.sequence_bytes)
                steps[role] += len(values)
                losses[role].extend(values)
        for role in optimizers:
            # Epoch exports protect useful weights even if a later epoch is cancelled.
            export(paths[role], detectors[role], role, sequence_bytes=args.sequence_bytes, trained_steps=steps[role])
        progress({"phase": "detectors", "epoch": epoch + 1,
                  "losses": {k: sum(v) / len(v) if v else None for k, v in losses.items()},
                  "validation": evaluate(detectors, val_specs, args, records), "steps": dict(steps)})

    intelligence_steps = 0
    if "intelligence" in tasks:
        intelligence = make_intelligence()
        optimizer = torch.optim.AdamW(intelligence.parameters(), lr=args.learning_rate)
        for model in detectors.values():
            model.eval().requires_grad_(False)
        dependencies = {role: artifact_hash(path) for role, path in paths.items()}
        for epoch in range(args.intelligence_epochs):
            losses, coverage = [], {}
            for source, annotation in checked_sources(train_specs, args, records, "intelligence"):
                predictions = {role: detect(model, source.text, source.spec.uri, source.content_sha256,
                                            role, args.sequence_bytes)[0] for role, model in detectors.items()}
                examples, counts = link_examples(predictions["usages"], predictions["definitions"], annotation)
                for key, value in counts.items():
                    coverage[key] = coverage.get(key, 0) + value
                values = intelligence_step(intelligence, optimizer, examples)
                intelligence_steps += len(values)
                losses.extend(values)
            path = output / "code-intelligence.model.bin"
            export(path, intelligence, "code-intelligence", trained_steps=intelligence_steps, dependencies=dependencies)
            progress({"phase": "intelligence", "epoch": epoch + 1, "steps": intelligence_steps,
                      "loss": sum(losses) / len(losses) if losses else None, "coverage": coverage,
                      "validation": evaluate(detectors, val_specs, args, records, intelligence)})
        paths["intelligence"] = str(path)
    result = {"outputs": paths, "detectorSteps": steps, "intelligenceSteps": intelligence_steps,
              "intelligenceStatus": "ANNOTATION_SUPERVISED" if intelligence_steps else "UNTRAINED",
              "annotationManifest": args.annotation_manifest, "ledger": args.ledger,
              "validationSources": len(val_specs), "trainingSources": len(train_specs),
              "settings": vars(args), "seeds": {"definitions": 17, "usages": 18, "intelligence": 19}}
    (output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
