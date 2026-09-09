# Completion capacity experiment — 2026-09-09

The larger completion model reached **55.67% next-byte top-1 accuracy** after
10 CPU epochs. A fresh baseline control reached **49.43%** on the same source
hashes and held-out split: **+6.24 percentage points**. Increasing capacity
helped this experiment; neither run established a convergence ceiling.

| Configuration | Small control | Larger completion |
| --- | ---: | ---: |
| Byte embedding width | 96 | 160 |
| GRU hidden width | 192 | 384 |
| GRU layers | 2 | 3 |
| Parameters | 473,298 | 2,562,322 |
| Epochs | 10 | 10 |
| Correct held-out bytes / 29,438 | 14,551 | 16,388 |
| Final accuracy | 49.43% | 55.67% |

Each recurrent layer updates a learned summary of preceding bytes. The final
layer feeds next-byte, boundary, and causal language heads. State is carried
between chunks; gradients are detached at chunk boundaries. Larger dimensions
are encoded in the binary's `completionConfig` and tensor shapes. The artifact
requires a SYL2 neural runtime; the existing legacy Kotlin SYLM1 runner does
not yet implement it.

## Learning curves

| Epoch | Small accuracy | Large accuracy |
| --- | ---: | ---: |
| 1 | 22.24% | 31.42% |
| 2 | 34.50% | 37.71% |
| 3 | 37.52% | 41.55% |
| 4 | 40.73% | 46.14% |
| 5 | 43.18% | 48.77% |
| 6 | 44.91% | 51.21% |
| 7 | 46.90% | 52.84% |
| 8 | 47.53% | 53.82% |
| 9 | 48.52% | 54.92% |
| 10 | 49.43% | 55.67% |

The legacy JSON field `precision` reports byte classification accuracy here.
It does not measure whole-word completion usefulness or acceptance. Both runs
ended at the requested epoch cap, below the 80% plateau gate and 99% target.
No architectural plateau was demonstrated. The original 48.54% result used
older versions of two training files; this new control removes that confound.

## Data and reproduction

Four local documents were trained: `training/syl2_trainer.py`,
`src/main/kotlin/syntaxlm/SyntaxLm.kt`, `teacher-config.json`, and
`MODEL_ARCHITECTURE.md`. `syntaxlm.py` alone was held out by the existing
automatic source split. This is a small mixed Python/Kotlin/JSON/prose corpus,
with Python-only validation, not a representative multilingual completion
benchmark. There are no downloaded MLCPD sources in these runs. Only one seed
was used, and the comparison is at equal epochs rather than equal CPU time.

```bash
.venv310/bin/python -u -m training.syl2_trainer \
  --source syntaxlm.py --source training/syl2_trainer.py \
  --source src/main/kotlin/syntaxlm/SyntaxLm.kt \
  --source teacher-config.json --source MODEL_ARCHITECTURE.md \
  --license-id project-internal --tasks completion --epochs 10 \
  --max-bytes 262144 --sequence-bytes 512 --device cpu --cpu-threads 1 \
  --completion-embedding 160 --completion-hidden 384 --completion-layers 3 \
  --target-precision 0.99 --plateau-precision-gate 0.80 \
  --output-dir runs/completion-large-repeat \
  --ledger runs/completion-large-repeat-source-use.jsonl
```

Omit the three `--completion-*` flags for the original-size control and use a
different output directory/ledger. Both experiments use seed 17 and AdamW with
learning rate 0.0003. Exact source hashes appear in the source-use ledgers.
Changing those files changes the corpus for a subsequent run.

Artifacts: [larger binary](next-word.model.bin), [epoch log](epochs.jsonl),
[control binary](../completion-baseline-control/next-word.model.bin),
[large source ledger](../completion-large-cpu-source-use.jsonl), and
[control source ledger](../completion-baseline-control-source-use.jsonl).

The roles and symbols artifacts were not retrained. Next steps for judging
generalization are a much broader code corpus, repository-separated validation
across languages, and word-level completion metrics. Continuing these small
local experiments alone cannot establish editor-quality multilingual support.

Verification: all 16 tests passed, including causal chunk-state equivalence
and variable-size tensor export. Both saved binaries passed checksum checks,
contained finite tensors, loaded strictly into models built from their embedded
configuration, and reproduced their exact final validation counts. The larger
binary is 10,254,516 bytes; the small control is 1,898,052 bytes.
