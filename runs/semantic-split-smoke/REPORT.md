# Split semantic pipeline smoke run

This is an integration fixture, **not a production model bundle**. Its percentages do not estimate multilingual quality. Source/annotation fixtures were temporary and have been removed; `training/smoke_semantics.py` recreates them. The provenance ledger is `../semantic-split-smoke-source-use.jsonl`.

## Training

Executed on CPU with one PyTorch thread: two authored training files and one held-out authored Python file. Each contains two assignments followed by an expression using both variables. Teacher `authored-fixture` version 2 labels all four occurrences. This validates pipeline mechanics, not generalization across projects.

40 detector epochs, followed by 10 intelligence epochs; 32-byte windows, learning rate 0.003. Both detectors performed 80 optimizer steps. The ranker performed 40 steps against the frozen detectors' actual predicted records. Full settings, seeds, counts and paths are in `training.json`; each epoch is in `epochs.jsonl`.

| Artifact | Parameters | File bytes |
| --- | ---: | ---: |
| definitions.model.bin | 148,474 | 597,679 |
| usages.model.bin | 148,474 | 597,674 |
| code-intelligence.model.bin | 2,177 | 10,667 |

The final held-out fixture has only two definitions and two usages:

- Definitions: 2 correct of 2 predicted; 2 of 2 annotated definitions found.
- Usages: 1 correct of 1 predicted; 1 of 2 annotated usages found.
- Links: 1 correct top-1 candidate of 1 eligible query; candidate coverage is **1/2 (50%)**.

Top-1 ranking here is measured before the runtime confidence threshold. It is not resolved precision at 0.8. One missing upstream usage cannot be rescued by a high score on the other query. Nothing in this run establishes real-world 99% precision.

## Kotlin CPU timing

Measured on Intel Xeon W-11955M, Kotlin/JVM 2.2.0 and JDK 21, using scalar Kotlin operations. The detector input is the authored `tests/semantic_benchmark_source.py` (288 UTF-16 units, ASCII). Model windows are 32 bytes for these smoke artifacts, not the default 512. Measurements exclude file loading and JVM startup. There is no incremental document cache.

| Operation | Warm median | Warm p95 |
| --- | ---: | ---: |
| Definitions, full 288-unit snippet | 21.41 ms | 22.59 ms |
| Usages, full 288-unit snippet | 23.22 ms | 40.35 ms |
| Rank one usage against 100 definitions | 0.38 ms | 0.72 ms |

Each detector had five warmup passes followed by 30 measured passes. Ranking had ten warmups and 30 measured passes; its candidates are synthetic and all share a name. Tests were also active during this timing session, so this is a development-machine spot check, not an isolated benchmark or latency guarantee. A whole-document analysis runs both detectors; these are not generated-word latencies or timings for the old role/symbol models.

Reproduce with:

```bash
kotlinc src/main/kotlin/syntaxlm/SemanticModels.kt tests/SemanticCheck.kt -d /tmp/semantic-benchmark.jar
kotlin -classpath /tmp/semantic-benchmark.jar syntaxlm.SemanticCheck \
  runs/semantic-split-smoke tests/semantic_benchmark_source.py
```

Use larger representative documents, incomplete editor states, cold-load timing, peak memory, and separately measured production weights before enforcing the 10–150 ms editor budget.
