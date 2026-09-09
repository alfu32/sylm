"""Measure warmed CPU decoding and prefix replay for configurable GRU models.

Untrained configurations test computational cost, not prediction quality. The
benchmark uses dense float32 PyTorch kernels, batch size one and a single beam.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .syl2_format import read_syl2
from .syl2_trainer import BOS, _make_models


CONFIGURATIONS = {
    "baseline": (96, 192, 2),
    "large": (160, 384, 3),
    "7m": (192, 640, 3),
    "17m": (256, 1024, 3),
    "32m": (256, 1408, 3),
    "65m": (256, 2048, 3),
}


def summarize(milliseconds: list[float]) -> dict:
    ordered = sorted(milliseconds)
    return {"samples": len(ordered), "medianMs": statistics.median(ordered),
            "p95Ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "meanMs": statistics.mean(ordered), "maxMs": ordered[-1]}


def build_model(config: tuple[int, int, int], artifact: str | None):
    if artifact:
        metadata, tensors = read_syl2(artifact)
        if metadata["taskId"] != "next-word":
            raise ValueError("benchmark requires a completion artifact")
        info = metadata["completionConfig"]
        config = (info["embeddingSize"], info["hiddenSize"], info["layers"])
    torch, _, model, _, _ = _make_models(
        17, completion_embedding=config[0], completion_hidden=config[1],
        completion_layers=config[2])
    if artifact:
        import numpy as np
        model.load_state_dict({
            name: torch.from_numpy(np.frombuffer(payload, dtype="<f4").copy().reshape(shape))
            for name, (shape, payload) in tensors.items()
        }, strict=True)
    return torch, model.eval(), config


def measure(model, torch, *, steps: int, repeats: int, prefix_sizes: list[int]) -> dict:
    # Input allocations, imports and model loading are outside the timed region.
    # This is warm steady-state latency, including all heads and greedy selection.
    token = torch.tensor([[BOS]], dtype=torch.long)
    hidden = None
    with torch.inference_mode():
        for _ in range(32):
            logits, _, _, hidden = model(token, hidden)
            token = logits[:, -1, :256].argmax(-1).unsqueeze(1)
        durations = []
        start_token, start_hidden = token, hidden
        for _ in range(steps):
            started = time.perf_counter()
            logits, _, _, hidden = model(token, hidden)
            # Force a fixed-length byte stream: no stop on EOS or boundary.
            token = logits[:, -1, :256].argmax(-1).unsqueeze(1)
            durations.append((time.perf_counter() - started) * 1000)
        decode = summarize(durations)
        decode["bytesPerSecond"] = 1000 / decode["meanMs"]
        spans = {}
        for length in (8, 16, 32):
            timings = []
            for _ in range(repeats):
                token, hidden = start_token, start_hidden
                started = time.perf_counter()
                for _ in range(length):
                    logits, _, _, hidden = model(token, hidden)
                    token = logits[:, -1, :256].argmax(-1).unsqueeze(1)
                timings.append((time.perf_counter() - started) * 1000)
            spans[str(length)] = summarize(timings)
        prefix = {}
        sample = b"def example(value):\n    return value + 1\n"
        for length in prefix_sizes:
            data = (sample * (length // len(sample) + 1))[:length]
            ids = torch.tensor([[BOS] + list(data)], dtype=torch.long)
            timings = []
            for _ in range(repeats):
                hidden = None
                started = time.perf_counter()
                for offset in range(0, ids.shape[1], 256):
                    _, _, _, hidden = model(ids[:, offset:offset + 256], hidden)
                timings.append((time.perf_counter() - started) * 1000)
            prefix[str(length)] = summarize(timings)
    return {"decodeByte": decode, "generateBytes": spans, "prefixReplayBytes": prefix}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefix-bytes", nargs="+", type=int, default=[256, 1024])
    parser.add_argument("--model", help="benchmark this trained artifact instead of the size sweep")
    parser.add_argument("--output", required=True, help="new report path (refuses overwrite)")
    args = parser.parse_args()
    if min(args.threads + args.prefix_bytes + [args.steps, args.repeats]) < 1:
        parser.error("thread counts, sample counts and lengths must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new report path")
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "cpu": cpu,
              "platform": platform.platform(), "python": platform.python_version(),
              "runtime": "PyTorch CPU float32; NOT Kotlin", "beamWidth": 1,
              "timing": "warm wall-clock; allocations for input, model load and disk IO excluded",
              "unit": "one byte; words/identifiers require multiple steps",
              "prefixInput": "synthetic ASCII code; BOS included",
              "weights": "trained artifact" if args.model else "seed-17 random initialization",
              "artifact": args.model, "results": []}
    names = ["artifact"] if args.model else args.sizes
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        for threads in args.threads:
            print(json.dumps({"event": "benchmark_start", "size": name, "threads": threads}),
                  file=sys.stderr, flush=True)
            torch, model, config = build_model(CONFIGURATIONS.get(name, CONFIGURATIONS["large"]), args.model)
            torch.set_num_threads(threads)
            report["torchVersion"] = torch.__version__
            count = sum(p.numel() for p in model.parameters())
            metrics = measure(model, torch, steps=args.steps, repeats=args.repeats,
                              prefix_sizes=args.prefix_bytes)
            record = {"size": name, "embedding": config[0], "hidden": config[1], "layers": config[2],
                      "parameters": count, "float32WeightMB": count * 4 / 1_000_000,
                      "threads": threads, **metrics}
            report["results"].append(record)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"event": "benchmark_complete", **record}), file=sys.stderr, flush=True)
            del model
            gc.collect()
    print(json.dumps({"report": str(output), "configurationsMeasured": len(report["results"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
