# Split semantic pipeline: implemented v1

Syntax highlighting is unchanged. The new implementation separates definitions from usages, then trains a small intelligence model on their actual outputs. Python/PyTorch is offline tooling only. Kotlin performs the same neural calculations from exported SYL2 tensor binaries.

```text
source snapshot ──┬── definition detector ── definition records ──┐
                 └── usage detector ─────── usage records ──────┤
query/cursor position ──────────────────────────────────────────┤
                                                              ▼
                                                    learned candidate ranker
                                                              │
                                                    definition link / abstain
```

This first intelligence stage resolves a supplied usage against supplied definition candidates. It does **not** generate the next word, model full syntax trees, infer scopes/types, or resolve imports like a compiler. Those are subsequent capabilities, not claims about the implemented ranker. No old completion model is resumed or resized by this change.

## Definitions and usages: independent contextual models

Both detectors have the following architecture, but separate weights, optimizers, losses, and files. Neither receives the other's labels or outputs.

| Layer | Dimensions | Purpose |
| --- | --- | --- |
| Input encoding | UTF-8 bytes, vocabulary 259 | Lossless multilingual input; special BOS/reset token. No language-specific tokenizer in the client. |
| Embedding | 259 × 64 | Learns a 64-dimensional representation of each input byte. |
| Bidirectional GRU, layer 1 | Input 64; hidden 64 per direction | Combines preceding and following bytes inside the analysis window. |
| Bidirectional GRU, layer 2 | Input 128; hidden 64 per direction | Builds contextual features from the first recurrent layer. Output width 128. |
| Occurrence head | Linear 128 → 3 | Predicts none/begin/inside for this detector's name spans. |
| Kind head | Linear 128 → 8 | Predicts unknown, variable, function, type, field, parameter, module, constant. |
| Language head | Mean pooling, linear 128 → registry size | Predicts source language from contextual activations; window probabilities are averaged by character count. |

Each detector currently has 148,474 parameters and exports approximately 598 KB of float32 weights and metadata. Default windows hold at most 512 source bytes plus BOS; recurrent state resets between windows. Context is bidirectional **within a window**, not whole-project context. Windows never split a Unicode code point. Span decoding can continue an inside tag across a window boundary but this does not extend recurrent context.

Training uses BIO cross-entropy + 0.2 × kind cross-entropy + 0.1 × language cross-entropy, with AdamW and gradient clipping. Unannotated bytes are masked unless the teacher explicitly guarantees complete occurrence coverage. Kind loss is applied only to annotated own-role spans.

The output record contains ID, document ID, revision, `[start,end)` UTF-16 range, name, kind, confidence, and analyzed `contextEnd`. These are name ranges, not full declaration/signature ranges. Predictions are decoded on complete Unicode characters, so returned ranges cannot bisect a surrogate pair. For a separately analyzed snippet, the caller must translate ranges and query coordinates by the snippet's UTF-16 origin and use a consistent document revision.

Language heads share the existing target registry, including supplemental JSON/XML/HTML/JSX/TSX/Svelte identities. That makes training multilingual-capable; it does not establish accuracy in languages absent from training. This version predicts document-level language probabilities, not embedded-language regions.

## Code intelligence: learned definition ranking

| Layer | Dimensions | Purpose |
| --- | --- | --- |
| Record features | 32 per usage/candidate pair | Encodes the actual detector outputs and the supplied query position. |
| Hidden linear + tanh | 32 → 64 | Learns interactions among proximity, names, kinds, confidence, and document identity. |
| Score projection | 64 → 1 | Produces one compatibility score for each definition and an explicit null candidate. |
| Candidate softmax | N + 1 | Normalizes scores over the supplied definitions and null. |
| Generic decoder | Configurable threshold, default 0.8 | Returns a definition only if it wins and exceeds the threshold; otherwise abstains. |

The ranker has 2,177 parameters, approximately 10.7 KB including metadata. Candidate count is capped at 512, with overflow rejected rather than silently dropping the correct target. Runtime ranking can accept cross-file records, but the current trainer supervises local same-file links only; cross-file quality is unvalidated.

### Position and distance features

The 32 features, in exact binary-contract order, are:

1. Null candidate, same document, same name, same kind, usage confidence, definition confidence, definition-before-usage, signed definition-to-usage distance.
2. Eight one-hot usage kinds, followed by eight one-hot definition kinds.
3. Query-present, usage position, definition position, query position, signed usage-to-query distance, absolute usage-to-query distance, signed definition-to-query distance, absolute definition-to-query distance.

All positions use original UTF-16 code units. Absolute positions are normalized with `p / (p + 4096)`. Signed distances use `clamp((position - reference) / 4096, -1, 1)`. Cross-document definition position/distance features are zero, with `sameDocument` distinguishing that case. Navigation defaults the query to the usage start; callers can explicitly supply the completion cursor.

Nearby usages/definitions often provide useful evidence, but distance is **not** a fixed inverse-proportional probability or a nearest-match rule. The network learns its contribution. Training needs counterexamples involving shadowing, forward references, imports, and same-named members. The current compact features cannot represent receiver types or scope nesting; increasing parameters alone cannot recover missing information. Distances beyond 4096 units saturate in this version.

The normalized [semantic context contract](training/semantic_context.py) defines optional scope IDs/chains, declared or inferred types, receiver types, import identity, stable symbol keys, and read/write/static flags for compiler/indexer teachers. These fields are validated and can travel with records, but are not yet included in the 32-feature trained ranker. A future feature-schema revision must update Python and Kotlin together and retrain the ranker.

Softmax confidence is not calibrated semantic certainty. It also depends on which candidates were supplied. Candidate accuracy must be reported alongside upstream detection recall and candidate coverage; high ranking accuracy on a small eligible subset can hide poor navigation coverage.

## Training on predictions, not teacher-perfect inputs

`training.semantic_trainer` is a new CLI; the old `training.syl2_trainer` and its combined `symbols` task remain unchanged for compatibility.

1. Train definition and usage detectors independently on source plus annotation targets. Export each completed epoch.
2. Freeze both detectors. Reacquire each source, run those exact models, and build ranker features from their predicted ranges, names, kinds, and confidence.
3. Use teacher links only to select the correct predicted candidate as the loss target. A missing predicted usage/definition is counted and excluded from ranker loss, never repaired with a gold record.
4. Train ranker cross-entropy over candidates plus null. Only explicit `status: "no_definition"` examples supervise null. Missing teacher data, unsupported external targets, and absent upstream predictions are not null labels.
5. Validate using the same prediction pipeline. Record detector exact-span precision/recall only on completely annotated sources; report annotated recall separately for partial sources. Report eligible link top-1 accuracy and candidate coverage, not byte accuracy as semantic precision.

The current stage-two training uses the detectors' predictions on their training sources. Production training should add out-of-fold or disjoint detector-prediction data to expose realistic upstream errors. Reserve project-disjoint validation/test data across all models. The CLI's file-level split fallback is not a substitute for project-level manifest construction. The tiny smoke fixture is solely an integration check.

Each usage annotation can supply `queryPosition`; it must be a valid UTF-16 boundary in the source. It defaults to the usage start. Current labels remain usage-to-definition navigation targets, even when a cursor feature is supplied. This does not create next-word supervision. Future completion training must generate prefix-only detector outputs and attach the held-out next complete code word as its target; full-document bidirectional outputs would leak future text.

Source is streamed one file at a time through the existing acquisition code. Raw source is not retained by this trainer; the ledger records URI, hash, license, split, teacher versions, and usage phase. Existing source/annotation manifests remain caller-owned. Epoch count bounds training. Completed epoch exports survive cancellation, but contain inference weights only, not Adam state or exact-resume checkpoints. No derivative-based automatic stopping is implemented in this new CLI yet.

### Commands

Use the project venv with `requirements.txt` installed. For new training:

```bash
.venv310/bin/python -m training.semantic_trainer \
  --annotation-manifest annotations.jsonl \
  --output-dir runs/semantic \
  --ledger runs/semantic-source-use.jsonl \
  --epochs 10 --intelligence-epochs 10 --sequence-bytes 512 --cpu-threads 1
```

To train only intelligence against fixed detectors:

```bash
.venv310/bin/python -m training.semantic_trainer \
  --tasks intelligence --annotation-manifest annotations.jsonl \
  --definitions-model runs/semantic/definitions.model.bin \
  --usages-model runs/semantic/usages.model.bin \
  --output-dir runs/semantic-ranker \
  --ledger runs/semantic-ranker-source-use.jsonl \
  --sequence-bytes 512 --intelligence-epochs 10
```

The window setting must match the supplied detector artifacts. Use an empty output directory to preserve existing runs. Each epoch emits JSON progress and appends `epochs.jsonl`; `training.json` records outputs, settings, seeds, and step counts. Zero-step artifacts are marked `UNTRAINED`; the Kotlin provider refuses to serve them.

## Kotlin client

Compile `src/main/kotlin/syntaxlm/SemanticModels.kt` into the app. Only the Kotlin/JVM standard library and exported binaries are needed; Python, PyTorch, parsers, and LSP servers are not runtime dependencies.

```kotlin
import java.io.File
import syntaxlm.SemanticModelProvider

val models = SemanticModelProvider(
    File("definitions.model.bin"),
    File("usages.model.bin"),
    File("code-intelligence.model.bin")
)
val snapshot = models.analyze(source, documentId = "editor://main.kt")
// models.languages gives labels in the order of snapshot.*Languages probabilities.
val usage = snapshot.usages.firstOrNull { caret >= it.start && caret < it.end }
val target = usage?.let { models.resolve(it, snapshot.definitions).definition }
// target.documentId and [target.start, target.end) identify the predicted name.
// Navigate only while the document revision still matches target.revision.
```

For a completion-context query, detect from the prefix, not the entire document:

```kotlin
val prefix = models.analyze(source, "editor://main.kt", cursor = caret)
val previousUsage = prefix.usages.lastOrNull()
val resolution = previousUsage?.let {
    models.resolve(it, prefix.definitions, cursor = caret, completionContext = true)
}
```

This resolves a previous usage in prefix context; it is **not** a next-word completion API. `analyze(cursor=...)` truncates the neural input before bidirectional inference. The full document hash is used only as a revision identity. `resolve(completionContext=true)` rejects local records whose analyzed context extends past the cursor, even if their name ranges precede it. Cross-file index snapshot freshness remains the caller's responsibility.

The provider verifies SYL2 checksums, shapes, finite weights, schema, registry consistency, and the exact detector SHA-256 hashes embedded in the intelligence artifact. Retraining either detector therefore requires retraining the ranker for a matching bundle. The client does generic byte encoding, tensor math, Unicode mapping, and decoding; it does not contain a language-specific name resolver. It does not yet provide incremental caching or an automatic project candidate index.

## Verification and limits

`tests/test_semantic_models.py` checks masked labels, prediction-only ranker inputs, source/cursor ranges, stale revisions, and future-context rejection. Its Kotlin probe compares GRU logits, language outputs, Unicode spans, pair features, and link probabilities against Python. Corrupted artifacts are rejected.

`training.smoke_semantics` trains on two authored tiny Python fixtures and validates on a third. It checks that both detectors train and that their frozen predictions yield real ranker optimizer steps. This is not multilingual training, a gold corpus, or a 99% precision result. See the smoke run's report for measured timings and fixture-only metrics.
