# local-lang-models

Small local language models for source-code understanding, designed so a Kotlin editor client can consume exported binary artifacts without invoking Python.

The project is currently in bootstrap state. It has a working legacy SYLM1 syntax highlighter with a Kotlin matrix runner, plus streaming trainers for SYLM1 and SYL2 artifacts. The SYL2 Python trainer now accepts gold role and local definition-link annotations, but SYL2 Kotlin inference and broad semantic coverage are not complete yet.

## Current Scope

Implemented today:

- legacy `SYLM` v1 syntax model: averaged perceptron, dependency-free Python trainer, portable matrix export;
- Kotlin `SYLM` v1 provider/runner: consumes only the exported matrix binary;
- bounded streaming source acquisition: reads one source at a time, uses it in memory, writes a metadata-only ledger;
- multilingual language registry for the TIOBE top-40 target set plus `unknown`;
- Tree-sitter grammar prefetch and coverage report for the target registry;
- first SYL2 neural trainer/exporter:
  - completion model trained self-supervised from source bytes;
  - token-role model trained from gold annotations when supplied, otherwise weak bootstrap labels;
  - identifier occurrence/kind and same-window definition-link heads trained from gold annotations when supplied;
  - SYL2 tensor container writer;
  - validation precision history and early stopping.

Not complete yet:

- Kotlin SYL2 neural runtime;
- compiler/indexer teacher adapters and broad cross-file definition-to-usage coverage;
- validated 99% precision across languages/tasks;
- automatic public-source discovery/crawling with license policy beyond manifest-driven raw file acquisition;
- semantic navigation quality for imports, overloads, members, shadowing, and cross-file symbols.

The detailed architecture is in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md). Training-data and retention rules are in [TRAINING_DATA.md](TRAINING_DATA.md).

## Model Plan

The target system has three local models:

1. `token-role`: classifies source ranges as keyword, identifier, punctuation, literal, comment, region, and related syntax roles while preserving positions.
2. `identifier-relation`: detects definitions/usages and links each usage to its definition where enough source/index context is available.
3. `next-word`: predicts the next word/token from the whole previous code prefix.

The Kotlin client should gather model results, validate ranges, cache document snapshots, and expose editor actions. It should not depend on Python code or offline annotation tools.

## Requirements

For the legacy CLI and SYLM1 trainer:

```bash
python3
```

For SYL2 neural training:

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt` currently contains `torch` and `numpy` for SYL2. The legacy path remains standard-library only.

For parser teachers, the environment also includes `tree-sitter` and
`tree-sitter-language-pack`. Prefetched grammars are stored separately in the
project-local `.treesitter-cache`; they are parser assets, not training source.

## Legacy SYLM1 Quick Start

Train and highlight with the dependency-free CLI:

```bash
python3 syntaxlm.py train --output syntaxlm.json
printf 'def greet(name):\n    return "hello " + name\n' \
  | python3 syntaxlm.py highlight --language python --model syntaxlm.json
```

Export the portable matrix binary for Kotlin:

```bash
python3 syntaxlm.py train --output syntaxlm.matrix.bin --format bin
kotlinc src/main/kotlin/syntaxlm/SyntaxLm.kt -d syntaxlm-kotlin.jar
printf 'fun twice(x: Int) = x * 2\n' \
  | kotlin -classpath syntaxlm-kotlin.jar syntaxlm.SyntaxLmRunner syntaxlm.matrix.bin kotlin
```

In an app, place `syntaxlm.matrix.bin` in assets/resources and use:

```kotlin
val provider = SyntaxHighlightProvider(context.assets.open("syntaxlm.matrix.bin"))
val spans = provider.highlight(source, "kotlin")
```

The `SYLM` v1 binary is a UTF-8 feature vocabulary followed by little-endian float32 rows. It is not the SYL2 neural container.

## Streaming Training

Streaming trainers take source references from direct `--source` values or a JSONL manifest. Each manifest line may include:

```json
{"uri":"https://example.invalid/project/main.py","language":"python","licenseId":"MIT","sourceId":"project/main.py","split":"train"}
```

Supported metadata fields include `uri`, `language`, `licenseId`, `sourceId`, `repositoryCommit`, `relativePath`, `split`, and `providerId`.

The trainer reads a bounded source into memory, trains/evaluates on it, discards it, and writes only model artifacts plus a ledger. The ledger contains source metadata and hashes, not source text, tokens, or annotations. `licenseId` is required by default; use `--license-policy allow` only for source you are authorized to use.

Train the legacy streaming bootstrap model:

```bash
python3 -m training.sylm1_trainer --manifest sources.jsonl \
  --output sylm1.matrix.bin --ledger source-use-sylm1.jsonl
```

Train the first SYL2 artifacts (self-/weakly supervised when no annotation manifest is supplied):

```bash
python3 -m training.syl2_trainer --manifest sources.jsonl \
  --output-dir syl2 --ledger source-use-syl2.jsonl --tasks all
```

SYL2 outputs:

- `syl2/next-word.model.bin`: self-supervised causal byte model;
- `syl2/token-role.model.bin`: weakly supervised token-role model;
- `syl2/identifier-relation.model.bin`: partial identifier occurrence model; link head is exported but untrained.

## Supervised Training

Gold supervision is supplied separately from source acquisition. The annotation
manifest contains source identity, UTF-16 ranges, labels, and teacher
provenance; it does not contain source text. The trainer reacquires the source
through the normal bounded stream, validates the coordinates against that
source, trains, and discards the source.

One JSONL annotation record can look like this:

```json
{"uri":"/data/project/main.py","language":"python","licenseId":"MIT","split":"train","teacher":{"name":"scip-tree-sitter","version":"2026.09"},"roleSpans":[{"start":0,"end":3,"role":"keyword"},{"start":4,"end":7,"role":"identifier"}],"definitions":[{"id":"d1","nameRange":{"start":4,"end":7},"kind":"function"}],"usages":[{"start":35,"end":38,"definitionId":"d1","kind":"function"}]}
```

`start` is inclusive and `end` exclusive in UTF-16 units. `roleSpans` may use
the shared roles (`keyword`, `identifier`, `punctuation`, `string_literal`,
`number_literal`, `comment`, `array_literal`, and `object_literal`). Definition
IDs are local to one source record. A usage whose target is genuinely external
or unresolved may omit `definitionId` and set an explicit `status`.

Run supervised training by pairing the source manifest with the annotation
manifest:

```bash
python3 -m training.syl2_trainer --manifest sources.jsonl \
  --annotation-manifest annotations.jsonl --output-dir syl2-supervised \
  --ledger source-use-syl2-supervised.jsonl --tasks roles symbols \
  --epochs 100 --target-precision 0.99 --plateau-precision-gate 0.80 \
  --min-epochs 5 --early-stop-patience 3
```

The role model receives masked gold bytes, so unlabeled bytes do not become
false negatives. The identifier model trains occurrence and symbol-kind heads,
plus the dynamic usage-to-definition candidate link head for definitions in the
same training window. Cross-window links are reported as unevaluated coverage,
not as null links. The completion model remains source-byte self-supervised.
Artifacts are marked `SUPERVISED` only when matching gold labels were actually
used; otherwise their metadata remains `WEAKLY_SUPERVISED` or `PARTIAL`.

For a source-only annotation manifest, omit `--manifest`; its source metadata is
used as the acquisition manifest. A license ID is still required by default.

## Parser and Semantic Teachers

Generate annotations through the no-retention teacher runner:

```bash
python3 -m training.annotate_stream --manifest sources.jsonl \
  --teacher-config teacher-config.json \
  --output annotations.jsonl --ledger teacher-source-use.jsonl
```

The built-in `python-ast` teacher covers Python lexical spans, classes,
functions, imports, exception/resource constructs, comprehensions, lambdas,
definitions, and partial local usages. Its semantic result is explicitly
marked `partial`; it is not a replacement for a compiler/indexer.

External teachers receive one JSON request on stdin:

```json
{"language":"rust","sourceId":"src/lib.rs","uri":"...","contentSha256":"...","source":"..."}
```

They must return only annotation JSON (`roleSpans`, `definitions`, `usages`,
`constructs`, `relations`, and `teacher`) on stdout. They must not return source
payloads. This protocol can wrap Tree-sitter for syntax/construct regions,
compiler frontends for types, imports, inheritance and overrides, and SCIP or
language-native indexers for definitions, calls, references, aliases and
cross-file links. Commands and versions are pinned in
[teacher-config.example.json](teacher-config.example.json). The Python runner
uses stdin/stdout and bounded output, so no checkout or parser workspace is
required on disk.

A language is skipped when no authoritative teacher is configured. Parser-only
results may train syntax heads, but semantic model status remains partial until
definition/link annotations are supplied by a compiler or indexer teacher.

Prefetch the target grammar set and write its exact/fallback/unsupported
coverage report:

```bash
./.venv/bin/python -m training.prefetch_grammars \
  --cache-dir .treesitter-cache --report grammar-coverage.json
```

The current prefetch retrieved 33 distinct grammars. `scratch`, `gml`, and
`abap` need dedicated grammars or compiler-specific teachers; `unknown` is an
intentional abstention class. Family/dialect fallbacks are recorded in
[grammar-coverage.json](grammar-coverage.json) and are not counted as exact
language support.

## Early Stopping

For long SYL2 runs, include validation entries in the manifest with `"split":"validation"` and use bounded stopping:

```bash
python3 -m training.syl2_trainer --manifest sources.jsonl \
  --output-dir syl2 --ledger source-use-syl2.jsonl --tasks all \
  --epochs 100 --target-precision 0.99 --plateau-precision-gate 0.80 \
  --min-epochs 5 --early-stop-patience 3
```

Training stops when held-out precision reaches `--target-precision`. It can also stop on a flat validation precision curve, but only after precision reaches `--plateau-precision-gate`. Use `0.70` for exploratory runs, `0.80` as the default practical gate, and `0.90` when you want the run to keep going until it has reached a stronger baseline before plateau stopping is allowed.

If no validation split is present, early stopping cannot evaluate precision and the trainer runs for the requested epoch count. The exported SYL2 metadata records the precision history, stop reason, thresholds, source counts, language counts, and ledger path.

## Annotation Training

The legacy `syntaxlm.py train` command also accepts inline JSONL examples:

```json
{"language":"python","code":"def add(a, b): return a + b","spans":[{"start":0,"end":3,"kind":"keyword"},{"start":4,"end":7,"kind":"function"}]}
```

Train with:

```bash
python3 syntaxlm.py train --data labels.jsonl --output my-model.json --epochs 8
```

When annotations are absent, the legacy trainer generates bootstrap labels with a small built-in lexer. These labels are useful for smoke tests and initial learning, not for final editor-quality claims.

## Development Checks

Run the dependency-free test suite:

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile syntaxlm.py training/*.py tests/*.py
git diff --check
```

An actual SYL2 training smoke test requires installing `requirements.txt`.

## Quality Bar

The intended release target is 99% precision where the task supports that metric, measured on held-out data by language and task. A single aggregate score is not enough for release. Report per-language coverage, precision, unknown/unsupported cases, and whether labels are weak bootstrap labels or semantic teacher labels.

For model 2, weak occurrence labels do not prove navigation correctness. True usage-to-definition precision requires compiler/indexer annotations or reviewed semantic ground truth.
