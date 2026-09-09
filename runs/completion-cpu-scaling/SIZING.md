# Completion model sizing

The application budget is 10–150 ms for one complete code word or identifier.
The model generates bytes, so the benchmark measures 8, 16, and 32 generated
bytes as practical identifier lengths. Measurements used PyTorch float32,
batch size one, greedy decoding, four CPU threads, and warmed model state on an
Intel Xeon W-11955M. Prefix replay is a separate cost; the application should
cache recurrent state while the document changes.

| Model | Parameters | Float32 weights | 8 bytes | 16 bytes | 32 bytes | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.47M | 1.9 MB | 1.0 ms | 2.0 ms | 3.9 ms | Plenty of latency headroom; lower capacity |
| Large | 2.56M | 10.2 MB | 1.6 ms | 3.2 ms | 6.4 ms | Existing trained model |
| 7M | 6.77M | 27.1 MB | 4.4 ms | 8.9 ms | 18.0 ms | Very safe |
| 17M | 16.91M | 67.7 MB | 20.0 ms | 40.9 ms | 80.4 ms | Recommended |
| 31M | 31.34M | 125.4 MB | 42.4 ms | 85.8 ms | 172.5 ms | Borderline |
| 65M | 65.22M | 260.9 MB | 93.9 ms | 189.7 ms | 377.2 ms | Too large |

The 17M configuration is the largest tested size that keeps a 32-byte
identifier within 150 ms. The 31M configuration can fit 16-byte identifiers,
but has little room for scheduling, Kotlin runtime overhead, or beam search.
The 65M configuration fails the 16-byte target. Four CPU threads were used for
the table; single-thread results are slower.

These are model forward timings, not Kotlin timings. Java/Kotlin must implement
the same GRU and should use a cached recurrent state, float32 or quantized
weights, and greedy decoding for this budget. Float32 storage is approximately
four bytes per parameter: 64M parameters therefore require about 256 MB before
runtime memory. Float16 would halve storage, but requires compatible Kotlin
math and does not automatically halve CPU latency.

Run the benchmark with:

```bash
.venv310/bin/python -m training.benchmark_completion \
  --sizes baseline large 7m 17m 32m 65m --threads 1 4 \
  --steps 256 --repeats 5 --prefix-bytes 256 1024 \
  --output runs/completion-cpu-scaling/benchmark-new.json
```
