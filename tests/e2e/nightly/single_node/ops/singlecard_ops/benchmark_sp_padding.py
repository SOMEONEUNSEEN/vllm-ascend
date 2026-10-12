# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual NPU pad/cat comparison; see docs/sp_padding_performance_plan.md.

Measures synchronized batch wall time, including allocation and dispatch.
This is an isolated expression benchmark, not a custom-op or graph benchmark.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


def run_variant(x, tp_size, rank, variant, scope):
    padding = (-x.shape[0]) % tp_size
    value = x.dtype == torch.bool
    if padding:
        if variant == "pad":
            x = F.pad(x, (0, 0) * (x.ndim - 1) + (0, padding), value=value)
        else:
            pad_shape = list(x.shape)
            pad_shape[0] = padding
            x = torch.cat([x, x.new_ones(pad_shape) if value else x.new_zeros(pad_shape)], dim=0)
    if scope == "padding":
        return x
    chunk = x.shape[0] // tp_size
    out = x[rank * chunk : (rank + 1) * chunk]
    return out.clone() if padding == 0 else out


def measure(fn, iterations):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        output = fn()
        del output
    torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / iterations / 1000


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs="+", default=[129, 7168])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32", "bool"], default="bfloat16")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--scope", choices=["padding", "shard"], default="shard")
    parser.add_argument("--noncontiguous", action="store_true")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tp_size < 1 or not 0 <= args.rank < args.tp_size:
        parser.error("require tp-size > 0 and 0 <= rank < tp-size")
    if min(args.shape) < 0 or min(args.warmup, args.iterations, args.rounds) < 1:
        parser.error("shape dimensions must be nonnegative and iteration counts positive")
    if (args.dtype == "bool" and len(args.shape) != 1) or (args.dtype != "bool" and len(args.shape) not in (2, 3)):
        parser.error("use a 1D bool mask or 2D/3D hidden states")
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    storage_shape = list(args.shape)
    if args.noncontiguous:
        storage_shape[-1] *= 2
    x = torch.randint(0, 2, storage_shape, device=args.device).to(getattr(torch, args.dtype))
    if args.noncontiguous:
        x = x[..., ::2]
    original = x.clone()
    variants = {
        name: (lambda name=name: run_variant(x, args.tp_size, args.rank, name, args.scope)) for name in ("pad", "cat")
    }
    torch.testing.assert_close(variants["cat"](), variants["pad"](), rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    for fn in variants.values():
        for _ in range(args.warmup):
            fn()
    torch.npu.synchronize()
    rng = random.Random(args.seed)
    records = []
    for round_index in range(args.rounds):
        order = list(variants)
        rng.shuffle(order)
        record = {"round": round_index, "order": order}
        for name in order:
            record[name] = measure(variants[name], args.iterations)
        records.append(record)
    result = {
        "config": {**vars(args), "output": str(args.output)},
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(),
        "stride": list(x.stride()),
        "padding": (-x.shape[0]) % args.tp_size,
        "metric": "synchronized batch wall microseconds per call (not kernel-only time)",
        "median_us": {name: statistics.median(row[name] for row in records) for name in variants},
        "median_paired_pad_over_cat": statistics.median(row["pad"] / row["cat"] for row in records),
        "records": records,
    }
    # Keep existing experiment results intact.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
