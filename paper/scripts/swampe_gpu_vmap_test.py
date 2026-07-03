#!/usr/bin/env python3
"""GPU batched-throughput sweep for my_swamp via jax.vmap.

CLI port of paper/scripts/swampe_gpu_vmap_test.ipynb -- same regime, same
knobs, same doubling/plateau-detection algorithm, same RESULTS block, so
output from either is directly comparable/quotable. The notebook is the
original, interactive Colab artifact (`Runtime > Change runtime type > GPU`
-> `Run all`); this script is the same methodology for a GPU you reach over
SSH without a notebook server.

`jax.vmap` runs a batch of trajectories (a sweep of DPhieq) in one compiled
call. The point is to find the batch size that uses the GPU *efficiently*, not
the biggest that merely fits in memory: starting at N=1, batch size doubles
until a doubling adds less than PLATEAU_TOL throughput (the GPU has saturated
and batch time is scaling linearly with N), or MAX_BATCH/an OOM is hit. The
"efficient knee" is then the smallest N within 10% of the peak throughput
reached, and a detailed diagnostic run is repeated there.

Run on a GPU. It will still run on CPU for a quick correctness check, but the
headline throughput numbers will not be representative.

Usage
-----
    python paper/scripts/swampe_gpu_vmap_test.py
    python paper/scripts/swampe_gpu_vmap_test.py --precision float64 --sweep-days 10
    python paper/scripts/swampe_gpu_vmap_test.py --out paper/benchmark_data/gpu_vmap_sweep.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _device_memory_mb(device) -> float | None:
    """Best-effort peak GPU memory via JAX's memory_stats(); None if unsupported."""
    try:
        stats = device.memory_stats() or {}
    except Exception:
        return None
    peak = stats.get("peak_bytes_in_use")
    return None if peak is None else peak / 1.0e6


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _is_oom(exc: Exception) -> bool:
    s = str(exc).upper()
    return ("RESOURCE_EXHAUSTED" in s) or ("OUT OF MEMORY" in s) or ("OOM" in s)


def _timeit(fn, warmups: int, repeats: int) -> dict:
    for _ in range(warmups):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "mean": statistics.mean(times), "median": statistics.median(times), "min": min(times),
        "std": statistics.pstdev(times) if len(times) > 1 else 0.0, "raw": times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--precision", choices=("float64", "float32"), default="float64",
                         help="float64 matches the CPU parity/speed tests; float32 needs a fresh process.")
    parser.add_argument("--sweep-days", type=float, default=10.0,
                         help="Run length per throughput measurement (matches the CPU speed test window).")
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs averaged at each batch size.")
    parser.add_argument("--max-batch", type=int, default=8192, help="Hard cap / safety stop for the doubling.")
    parser.add_argument("--plateau-tol", type=float, default=0.10,
                         help="Stop doubling once a doubling adds less than this fraction of throughput.")
    parser.add_argument("--min-n", type=int, default=4, help="Don't declare a plateau before reaching this N.")
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "benchmark_data" / "gpu_vmap_sweep.json")
    args = parser.parse_args()

    use_x64 = args.precision == "float64"
    os.environ["SWAMPE_JAX_ENABLE_X64"] = "1" if use_x64 else "0"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", use_x64)

    try:
        from my_swamp.model import run_model_scan_final as _my_final

        def my_run(**kw):
            return _my_final(**kw)
    except Exception:
        from my_swamp.model import run_model_scan as _my_scan

        def my_run(**kw):
            return _my_scan(return_history=False, **kw)

    # Forced, synchronously rotating benchmark regime (matches the parity,
    # sensitivity, and gradient benchmark figures -- see
    # paper/scripts/benchmark_gradient.py).
    config = dict(
        M=42, dt=120.0, Phibar=3.0e5, omega=3.2e-5, a=8.2e7, g=9.8,
        test=None, forcflag=True, taurad=10.0 * 3600.0, taudrag=6.0 * 3600.0,
        DPhieq=1.0e6, diffflag=True, modalflag=True, alpha=0.01, expflag=False, K6=1.24e33,
    )
    base = {k: v for k, v in config.items() if k != "DPhieq"}
    dt = config["dt"]

    def tmax_for_days(days: float, dt: float) -> tuple[int, int]:
        n = int(round(days * 86400.0 / dt))
        return max(3, n + 1), n

    gpu = next((d for d in jax.devices() if d.platform == "gpu"), None)

    def dphieq_batch(n: int):
        return jnp.linspace(0.5e6, 2.0e6, n)

    def make_batched(tmax: int):
        def one(dpe):
            return my_run(tmax=tmax, DPhieq=dpe, jit_scan=True, diagnostics=False, **base)["last_state"].Phi_curr
        return jax.jit(jax.vmap(one))

    def run_batched(fn, xb):
        with jax.default_device(gpu) if gpu is not None else _no_device_ctx():
            out = fn(xb)
            out.block_until_ready()
            return out

    # build_static does host-side numpy on the Legendre basis, so it must run
    # CONCRETELY once (under jax.vmap it would np.asarray a tracer). One plain
    # call populates an lru_cache the vmap'd version reuses (CLAUDE.md S5).
    my_run(tmax=8, **config)["last_state"].Phi_curr.block_until_ready()

    print(f"jax {jax.__version__}  precision={args.precision}  x64={jax.config.read('jax_enable_x64')}")
    print(f"devices = {jax.devices()}")
    print(f"GPU = {_gpu_name() if gpu else 'NONE -- benchmarking CPU; numbers will not be representative'}")

    tmax, _ = tmax_for_days(args.sweep_days, dt)
    steps = tmax - 2
    print(f"sweep: M={config['M']} dt={dt:.0f}s  {args.sweep_days}-day runs ({steps} steps), "
          f"doubling N up to {args.max_batch}")
    print(f"{'N':>6} {'batch_s':>9} {'traj/s':>10} {'ms/traj':>9} {'ms/step/traj':>13} {'thr gain':>9} {'mem MB':>9}")

    rows = []
    n, prev_thr = 1, None
    while n <= args.max_batch:
        fn = make_batched(tmax)
        xb = dphieq_batch(n)
        try:
            run_batched(fn, xb)  # compile (excluded from timing)
            stats = _timeit(lambda: run_batched(fn, xb), warmups=0, repeats=args.repeats)
        except Exception as exc:
            if _is_oom(exc):
                print(f"  N={n}: OOM -- stopping (well past the efficient point anyway)")
                jax.clear_caches()
                break
            raise
        total = stats["mean"]
        throughput = n / total
        ms_per_traj = total / n * 1e3
        ms_per_step_per_traj = total / n / steps * 1e3
        gain = (throughput / prev_thr - 1.0) if prev_thr is not None else None
        peak_mem = _device_memory_mb(gpu) if gpu is not None else None
        rows.append(dict(N=n, T=total, thr=throughput, ptt=ms_per_traj, pspt=ms_per_step_per_traj,
                          std=stats["std"], gain=gain, mem=peak_mem, raw=stats["raw"]))
        print(f"{n:6d} {total:9.4f} {throughput:10.1f} {ms_per_traj:9.3f} {ms_per_step_per_traj:13.5f} "
              f"{'   -' if gain is None else f'{gain * 100:7.1f}%'} "
              f"{('-' if peak_mem is None else f'{peak_mem:,.0f}'):>9}")
        if gain is not None and gain < args.plateau_tol and n >= args.min_n:
            print(f"  -> throughput plateaued (last doubling added only {gain * 100:.1f}% < {args.plateau_tol * 100:.0f}%)")
            break
        prev_thr = throughput
        n *= 2

    peak = max(rows, key=lambda r: r["thr"])
    knee = next(r for r in rows if r["thr"] >= 0.90 * peak["thr"])
    print(f"\npeak throughput : N={peak['N']}  {peak['thr']:,.1f} traj/s  ({peak['ptt']:.3f} ms/traj)")
    print(f"efficient knee  : N={knee['N']}  {knee['thr']:,.1f} traj/s  ({knee['ptt']:.3f} ms/traj)  "
          f"<- smallest batch within 10% of peak")

    # Detailed diagnostics at the efficient batch.
    nf = knee["N"]
    fn = make_batched(tmax)
    xb = dphieq_batch(nf)
    run_batched(fn, xb)
    stats = _timeit(lambda: run_batched(fn, xb), warmups=0, repeats=max(args.repeats, 5))
    out = run_batched(fn, xb)
    arr = np.asarray(out)

    single = rows[0]
    total = stats["mean"]
    throughput = nf / total
    ms_per_traj = total / nf * 1e3
    ms_per_step_per_traj = total / nf / steps * 1e3
    vmap_eff = single["ptt"] / ms_per_traj
    finite_frac = float(np.isfinite(arr).mean())
    dphi = np.asarray(xb)
    per_member_mean = arr.reshape(arr.shape[0], -1).mean(axis=1)
    peak_mem = _device_memory_mb(gpu) if gpu is not None else None

    print("\n===== RESULTS (copy everything below into paper/speed_benchmark.md) =====")
    print(f"precision               = {args.precision}")
    print(f"gpu                     = {_gpu_name()}")
    print(f"M                       = {config['M']}")
    print(f"dt_seconds              = {dt:.0f}")
    print(f"sweep_days              = {args.sweep_days}")
    print(f"steps                   = {steps}")
    print(f"efficient_batch_N       = {nf}")
    print(f"peak_throughput_N       = {peak['N']}")
    print(f"throughput_traj_per_s   = {throughput:.1f}")
    print(f"peak_throughput_traj_s  = {peak['thr']:.1f}")
    print(f"ms_per_trajectory       = {ms_per_traj:.4f}")
    print(f"ms_per_step_per_traj    = {ms_per_step_per_traj:.5f}")
    print(f"single_traj_ms_per_traj = {single['ptt']:.4f}")
    print(f"vmap_efficiency_x       = {vmap_eff:.2f}")
    print(f"gpu_mem_peak_MB         = {('NA' if peak_mem is None else round(peak_mem))}")
    print(f"finite_frac             = {finite_frac:.6f}")
    print("sweep_N_thr =", [(r["N"], round(r["thr"], 1)) for r in rows])
    print("===== END RESULTS =====")

    payload = {
        "script": "paper/scripts/swampe_gpu_vmap_test.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "precision": args.precision,
        "gpu": _gpu_name(),
        "jax_version": jax.__version__,
        "regime": {**config, "sweep_days": args.sweep_days, "tmax": tmax, "steps": steps,
                   "dphieq_lo": 0.5e6, "dphieq_hi": 2.0e6},
        "plateau_tol": args.plateau_tol, "min_n": args.min_n, "max_batch": args.max_batch,
        "repeats": args.repeats,
        "sweep_rows": [{k: v for k, v in r.items()} for r in rows],
        "peak_n": peak["N"], "efficient_knee_n": nf,
        "efficient_batch_diagnostics": {
            "throughput_traj_per_s": throughput, "ms_per_trajectory": ms_per_traj,
            "ms_per_step_per_traj": ms_per_step_per_traj, "vmap_efficiency_x": vmap_eff,
            "gpu_mem_peak_MB": peak_mem, "finite_frac": finite_frac,
            "per_member_dphieq": dphi.tolist(), "per_member_mean_phi": per_member_mean.tolist(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


class _no_device_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
