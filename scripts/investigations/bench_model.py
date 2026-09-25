"""Measure what this network costs per step, so the training budget is chosen and not guessed.

Reports peak memory and throughput in points per second for a grid of crop sizes
and batch sizes, on whatever device is present. Throughput is the number that
decides the configuration: a bigger crop that halves the step rate buys nothing.

Also compares the two samplers. Farthest point sampling is the textbook choice
and runs one sequential kernel per sample. Random sampling is defensible here
because crops are voxel downsampled to a uniform density before they reach the
network, and this measures what the swap actually saves.

Run:
    python scripts/investigations/bench_model.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.model import SurreyNet, parameter_count  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "artifacts" / "bench_model.json"
CONFIGS = [(8192, 16), (16384, 4), (16384, 8), (32768, 4), (32768, 6), (65536, 2)]


def one(npts: int, bs: int, sampler: str, device: str, iters: int = 4) -> dict:
    torch.manual_seed(0)
    net = SurreyNet(15, width=128, sampler=sampler).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=amp)
    x = torch.randn(bs, npts, 4, device=device)
    x[..., 2] = x[..., 2].abs() * 10.0
    y = torch.randint(0, 15, (bs, npts), device=device)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dt = float("nan")
    for i in range(iters):
        t0 = time.perf_counter()
        with torch.amp.autocast(device, dtype=torch.float16, enabled=amp):
            loss = torch.nn.functional.cross_entropy(net(x).reshape(-1, 15), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if device == "cuda":
            torch.cuda.synchronize()
        if i:  # discard the first, it carries allocation and autotune
            dt = time.perf_counter() - t0 if i == 1 else min(dt, time.perf_counter() - t0)

    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    return {
        "points_per_crop": npts,
        "batch": bs,
        "sampler": sampler,
        "step_ms": round(dt * 1000, 1),
        "peak_gb": round(peak, 3),
        "points_per_second": round(npts * bs / dt),
        "fits": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    dev = args.device
    name = torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor()
    total = (
        torch.cuda.get_device_properties(0).total_memory / 1e9 if dev == "cuda" else float("nan")
    )
    print(f"device {dev}  {name}  {total:.2f} GB")
    print(f"params {parameter_count(SurreyNet(15, width=128)):,}\n")

    rows = []
    for npts, bs in CONFIGS:
        for sampler in ("random", "fps"):
            if sampler == "fps" and (npts, bs) != (32768, 4):
                continue  # one comparison point is enough, it is slow
            try:
                r = one(npts, bs, sampler, dev)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                r = {
                    "points_per_crop": npts,
                    "batch": bs,
                    "sampler": sampler,
                    "step_ms": None,
                    "peak_gb": None,
                    "points_per_second": None,
                    "fits": False,
                }
                print(f"{npts:>6} x {bs:<3} {sampler:7s}  OOM")
            else:
                print(
                    f"{npts:>6} x {bs:<3} {sampler:7s}  {r['step_ms']:>7.1f} ms  "
                    f"{r['peak_gb']:>5.2f} GB  {r['points_per_second']:>9,} pts/s"
                )
            rows.append(r)

    ok = [r for r in rows if r["fits"] and r["sampler"] == "random"]
    best = max(ok, key=lambda r: r["points_per_second"]) if ok else None

    result = {
        "device": dev,
        "device_name": name,
        "device_total_gb": round(total, 3) if total == total else None,
        "torch": torch.__version__,
        "amp": dev == "cuda",
        "parameters": parameter_count(SurreyNet(15, width=128)),
        "best_by_throughput": {
            k: best[k]
            for k in ("points_per_crop", "batch", "step_ms", "peak_gb", "points_per_second")
        }
        if best
        else None,
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    if best:
        print(
            f"\nbest throughput: {best['points_per_crop']} x {best['batch']} "
            f"at {best['points_per_second']:,} pts/s, {best['peak_gb']:.2f} GB"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
