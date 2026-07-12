#!/usr/bin/env python3
# ruff: noqa: E741
"""Quantify the opt-in numerics modes (readme section 9) against the locked defaults.

Stages (run one at a time; each prints a compact report):

  stability  Max stable dt: explicit modified-Euler vs semi-implicit,
             WASP-43b regime (Phibar=4e6), 5-day runs.
  accuracy   Terminal-state agreement of the semi-implicit scheme at growing dt
             against the explicit production reference (dt=120, K6=5e33), 10 days,
             including retrieval-relevant metrics (day-night amplitude, hot-spot offset).
  corner     The documented killer prior corner (tau_rad=48h, tau_drag=48h,
             Phibar=8e6): explicit production settings vs semi-implicit, 20 days.
  speed      Wall-clock, 20-day WASP-43b forward + 10-day forward-mode gradient:
             explicit dt=120 vs semi-implicit dt=1200.
  raw        RAW filter benefit on the exact steady zonal jet (test 2) under the
             semi-implicit leapfrog, where the time filter actually acts.

Usage:
  JAX_PLATFORMS=cpu MY_SWAMPE_ENABLE_X64=1 python scripts/benchmark_new_numerics.py --stage stability
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from my_swampe.model import run_model_scan_final, assert_finite_state

# WASP-43b-like regime (retrieval/wasp_43b_test/config/wasp43b_pilot_gpu.json):
# gravity waves sqrt(Phibar)~2000 m/s; production explicit solver runs dt=120 s
# with K6 rescaled to 5e33.
WASP43B = dict(
    M=42,
    Phibar=4.0e6,
    DPhieq=3.5e6,
    omega=8.939689388812098e-05,
    a=71920952.0,
    g=49.66,
    taurad=10.0 * 3600.0,
    taudrag=6.0 * 3600.0,
    test=None,
)
K6_PROD = 5.0e33     # production explicit value for this regime
K6_DEFAULT = 1.24e33  # package default (synthetic Phibar=3e5 regime)

DAY = 86400.0


def steps(days: float, dt: float) -> int:
    return 2 + int(round(days * DAY / dt))


def run(days: float, dt: float, *, K6: float, semi_implicit: bool = False,
        raw_filter: bool = False, williams_alpha: float = 0.53,
        params: dict = WASP43B, **kw):
    """Terminal-state run; returns (finite, seconds, Phi_terminal)."""
    t0 = time.perf_counter()
    out = run_model_scan_final(
        dt=dt, tmax=steps(days, dt), K6=K6,
        semi_implicit=semi_implicit, raw_filter=raw_filter,
        williams_alpha=williams_alpha, diagnostics=False, **params, **kw,
    )
    jax.block_until_ready(out["last_state"].Phi_curr)
    dt_wall = time.perf_counter() - t0
    finite = assert_finite_state(out["last_state"], raise_on_nan=False)
    return finite, dt_wall, np.asarray(out["last_state"].Phi_curr)


def equatorial_metrics(Phi: np.ndarray, Phibar: float):
    """Day-night amplitude + hot-spot longitude offset of the equatorial T proxy."""
    J, I = Phi.shape
    T = (Phibar + Phi) / 3.78e3  # geopotential temperature proxy (K)
    eq = 0.5 * (T[J // 2 - 1] + T[J // 2])
    lons = np.linspace(-180.0, 180.0, I, endpoint=False)
    return float(eq.max() - eq.min()), float(lons[int(np.argmax(eq))])


def stage_stability():
    print("== stability: 5-day WASP-43b runs, terminal state finite? ==")
    print("-- explicit modified-Euler, production K6=5e33 --")
    for dt in (120.0, 180.0, 240.0, 360.0):
        finite, wall, _ = run(5.0, dt, K6=K6_PROD)
        print(f"  modEuler dt={dt:6.0f}s  steps={steps(5.0, dt):5d}  finite={finite}  wall={wall:6.1f}s")
    print("-- semi-implicit, DEFAULT K6=1.24e33 --")
    for dt in (120.0, 300.0, 600.0, 1200.0, 1800.0, 2400.0, 3600.0):
        finite, wall, _ = run(5.0, dt, K6=K6_DEFAULT, semi_implicit=True)
        print(f"  semi-imp dt={dt:6.0f}s  steps={steps(5.0, dt):5d}  finite={finite}  wall={wall:6.1f}s")


def stage_accuracy():
    days = 10.0
    print(f"== accuracy: {days:.0f}-day WASP-43b terminal Phi vs explicit reference (dt=120, K6=5e33) ==")
    _, _, ref = run(days, 120.0, K6=K6_PROD)
    amp_ref, off_ref = equatorial_metrics(ref, WASP43B["Phibar"])
    sd = float(np.std(ref))
    print(f"  reference: day-night amplitude={amp_ref:7.1f} K  hotspot offset={off_ref:+6.1f} deg  std(Phi)={sd:.3e}")
    rows = [("modEuler", 120.0, K6_PROD, False)] + [
        ("semi-imp", d, K6_DEFAULT, True) for d in (120.0, 300.0, 600.0, 1200.0, 2400.0)
    ]
    for name, dt, K6, si in rows:
        finite, _, phi = run(days, dt, K6=K6, semi_implicit=si)
        if not finite:
            print(f"  {name} dt={dt:6.0f}s  NOT FINITE")
            continue
        amp, off = equatorial_metrics(phi, WASP43B["Phibar"])
        rms = float(np.sqrt(np.mean((phi - ref) ** 2))) / sd
        print(
            f"  {name} dt={dt:6.0f}s  rms(dPhi)/std={rms:8.2%}  "
            f"amplitude={amp:7.1f} K (d={amp - amp_ref:+6.1f})  offset={off:+6.1f} deg (d={off - off_ref:+5.1f})"
        )


def stage_corner():
    print("== corner: tau_rad=48h, tau_drag=48h, Phibar=8e6 (the 2/16 killer corner), 20 days ==")
    params = dict(WASP43B, Phibar=8.0e6, taurad=48.0 * 3600.0, taudrag=48.0 * 3600.0)
    finite, wall, _ = run(20.0, 120.0, K6=K6_PROD, params=params)
    print(f"  modEuler dt=120s K6=5e33      finite={finite}  wall={wall:6.1f}s")
    for dt in (600.0, 1200.0):
        finite, wall, _ = run(20.0, dt, K6=K6_DEFAULT, semi_implicit=True, params=params)
        print(f"  semi-imp dt={dt:4.0f}s K6=1.24e33  finite={finite}  wall={wall:6.1f}s")


def stage_corners():
    """All 16 corners of the WASP-43b retrieval prior box, 20-day stability.

    Box (wasp43b_pilot_gpu.json priors): tau_rad, tau_drag in {0.5, 48} h;
    Phibar in {2e6, 8e6}; DPhieq in {5e5, 5e6}. The production explicit solver
    (dt=120, K6=5e33) is documented 14/16 stable. The semi-implicit runs use
    the RAW filter at alpha=0.05 (the recommended robust setting) and default K6.
    """
    print("== corners: 16 prior-box corners, 20 days, finite? ==")
    corners = [
        (tr, td, pb, dp)
        for tr in (0.5, 48.0)
        for td in (0.5, 48.0)
        for pb in (2.0e6, 8.0e6)
        for dp in (5.0e5, 5.0e6)
    ]
    configs = [
        ("modEuler dt=120  K6=5e33   ", dict(dt=120.0, K6=K6_PROD)),
        ("semi-imp dt=600  K6=default", dict(dt=600.0, K6=K6_DEFAULT, semi_implicit=True,
                                             raw_filter=True, williams_alpha=0.53, alpha=0.05)),
        ("semi-imp dt=1200 K6=default", dict(dt=1200.0, K6=K6_DEFAULT, semi_implicit=True,
                                             raw_filter=True, williams_alpha=0.53, alpha=0.05)),
    ]
    for name, kw in configs:
        t0 = time.perf_counter()
        n_ok, failures = 0, []
        for tr, td, pb, dp in corners:
            params = dict(WASP43B, Phibar=pb, DPhieq=dp, taurad=tr * 3600.0, taudrag=td * 3600.0)
            dt = kw["dt"]
            finite, _, _ = run(20.0, params=params, **{k: v for k, v in kw.items() if k != "dt"}, dt=dt)
            if finite:
                n_ok += 1
            else:
                failures.append((tr, td, pb, dp))
        print(f"  {name}: {n_ok}/16 stable  (total {time.perf_counter() - t0:6.1f}s)")
        for tr, td, pb, dp in failures:
            print(f"      FAILED corner: tau_rad={tr}h tau_drag={td}h Phibar={pb:.0e} DPhieq={dp:.0e}"
                  f"  (contrast={dp / pb:.2f})")


def stage_speed():
    print("== speed: 20-day WASP-43b forward (production-equivalent) ==")
    results = {}
    for name, dt, K6, si in (("modEuler dt=120 ", 120.0, K6_PROD, False),
                             ("semi-imp dt=1200", 1200.0, K6_DEFAULT, True)):
        run(20.0, dt, K6=K6, semi_implicit=si)  # warm compile
        _, wall, _ = run(20.0, dt, K6=K6, semi_implicit=si)
        n = steps(20.0, dt)
        results[name] = wall
        print(f"  {name}  steps={n:6d}  wall={wall:7.2f}s  ({1e3 * wall / n:6.2f} ms/step)")
    print(f"  forward speedup: x{results['modEuler dt=120 '] / results['semi-imp dt=1200']:.1f}")

    print("-- 10-day forward-mode gradient d mean(Phi^2) / d tau_rad (jvp, like the retrieval) --")
    for name, dt, K6, si in (("modEuler dt=120 ", 120.0, K6_PROD, False),
                             ("semi-imp dt=1200", 1200.0, K6_DEFAULT, True)):
        def loss(taurad, dt=dt, K6=K6, si=si):
            p = dict(WASP43B)
            p["taurad"] = taurad
            out = run_model_scan_final(dt=dt, tmax=steps(10.0, dt), K6=K6,
                                       semi_implicit=si, diagnostics=False, **p)
            return jnp.mean(out["last_state"].Phi_curr ** 2)

        x = jnp.asarray(WASP43B["taurad"])
        jax.block_until_ready(jax.jvp(loss, (x,), (jnp.ones_like(x),)))  # warm
        t0 = time.perf_counter()
        val, grad = jax.jvp(loss, (x,), (jnp.ones_like(x),))
        jax.block_until_ready(grad)
        print(f"  {name}  value-and-grad wall={time.perf_counter() - t0:7.2f}s  grad={float(grad):.3e}")


def stage_raw():
    days = 10.0
    print(f"== raw: filter-induced damage at strong filtering, WASP-43b regime, {days:.0f} days ==")
    print("   (in the default modified-Euler scheme the classic RA filter never feeds back into")
    print("    the trajectory - the two-level scheme reads only the unfiltered current carries -")
    print("    so the filter matters where it acts: the semi-implicit leapfrog. Robust corner")
    print("    settings need alpha~0.05-0.1; RAW is what keeps that from degrading the physics.)")

    # Reference: weakly filtered small-dt semi-implicit solution.
    _, _, ref = run(days, 120.0, K6=K6_DEFAULT, semi_implicit=True, alpha=0.01)
    amp_ref, off_ref = equatorial_metrics(ref, WASP43B["Phibar"])
    sd = float(np.std(ref))
    print(f"  reference (SI dt=120, alpha=0.01): amplitude={amp_ref:7.1f} K  offset={off_ref:+6.1f} deg")

    for label, kw in (
        ("classic RA  alpha=0.10, dt=1200", dict(alpha=0.10)),
        ("RAW w=0.53  alpha=0.10, dt=1200", dict(alpha=0.10, raw_filter=True, williams_alpha=0.53)),
        ("classic RA  alpha=0.05, dt=1200", dict(alpha=0.05)),
        ("RAW w=0.53  alpha=0.05, dt=1200", dict(alpha=0.05, raw_filter=True, williams_alpha=0.53)),
    ):
        finite, _, phi = run(days, 1200.0, K6=K6_DEFAULT, semi_implicit=True, **kw)
        if not finite:
            print(f"  {label}  NOT FINITE")
            continue
        amp, off = equatorial_metrics(phi, WASP43B["Phibar"])
        rms = float(np.sqrt(np.mean((phi - ref) ** 2))) / sd
        print(f"  {label}  rms-vs-ref/std={rms:7.3%}  amplitude={amp:7.1f} K (d={amp - amp_ref:+6.1f})"
              f"  offset={off:+6.1f} deg (d={off - off_ref:+5.1f})")


STAGES = {
    "stability": stage_stability,
    "accuracy": stage_accuracy,
    "corner": stage_corner,
    "corners": stage_corners,
    "speed": stage_speed,
    "raw": stage_raw,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=sorted(STAGES) + ["all"], required=True)
    args = ap.parse_args()
    for name in (sorted(STAGES) if args.stage == "all" else [args.stage]):
        STAGES[name]()
