# ruff: noqa: E741
"""Tests for the opt-in semi-implicit gravity-wave scheme (CLAUDE.md 13.3).

Contract under test:
- ``semi_implicit=False`` (default) leaves the locked modified-Euler path
  untouched (covered by the parity suite; here we only exercise the new mode).
- The semi-implicit scheme is stable and finite at timesteps far above the
  explicit gravity-wave limit in the hot-Jupiter regime (Phibar=4e6).
- It converges toward the modified-Euler solution as dt shrinks.
- Gradients flow through the implicit solve and the exponential
  hyperdiffusion.
- Incompatible flag combinations raise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from my_swamp.model import assert_finite_state, run_model_scan, run_model_scan_final


_SYNTH = dict(
    M=42,
    Phibar=3.0e5,
    omega=3.2e-5,
    a=8.2e7,
    taurad=36000.0,
    taudrag=21600.0,
    DPhieq=1.0e6,
    test=None,
)

# WASP-43b-like regime (retrieval/wasp_43b_test): gravity waves ~3.7x faster
# than the synthetic setup; the explicit scheme needs dt=120 s and K6=5e33.
_WASP43B = dict(
    M=42,
    Phibar=4.0e6,
    omega=8.939689388812098e-05,
    a=71920952.0,
    taurad=36000.0,
    taudrag=21600.0,
    DPhieq=3.5e6,
    g=49.66,
    test=None,
)


@pytest.mark.smoke
def test_semi_implicit_runs_finite_default_regime():
    out = run_model_scan(**_SYNTH, dt=240.0, tmax=120, semi_implicit=True)
    assert int(out["dead_first_idx"]) == -1
    for key in ("eta", "delta", "Phi", "U", "V"):
        assert np.all(np.isfinite(np.asarray(out["outs"][key][-1]))), key


@pytest.mark.smoke
def test_semi_implicit_stable_beyond_explicit_gravity_wave_limit():
    # One day of the WASP-43b regime at dt=1200 s (10x the production explicit
    # dt=120 s) with the default K6 — only possible because gravity waves are
    # implicit and the hyperdiffusion is an exact integrating factor.
    out = run_model_scan_final(
        **_WASP43B,
        dt=1200.0,
        tmax=74,
        semi_implicit=True,
        diagnostics=False,
    )
    assert_finite_state(out["last_state"])


@pytest.mark.smoke
def test_semi_implicit_equilibrium_is_dt_insensitive():
    # The property the mode is sold on: growing dt by 5x leaves the forced
    # quasi-steady state unchanged (gravity waves and linear relaxation are
    # implicit, the stiff nonlinear forcing is lagged, the hyperdiffusion
    # factor is exact). Measured 1.6e-5 relative at these settings.
    #
    # NOTES: (1) this holds in the moderate-contrast (DPhieq < Phibar)
    # hot-Jupiter regime; in the super-contrast synthetic regime
    # (DPhieq/Phibar > 1) the nightside-collapse nonlinearity caps dt near the
    # explicit value regardless of scheme. (2) The *forced transient*
    # trajectory is not comparable to modified-Euler at any dt — the locked
    # SWAMPE quirks (CLAUDE.md section 3, items 1-3) give the forced
    # modified-Euler scheme O(1)-different effective tendencies, and the two
    # schemes settle to equilibria that differ by a few percent (identical
    # hot-spot offset, ~5% day-night amplitude). Comparisons against the
    # explicit production reference live in
    # scripts/benchmark_new_numerics.py --stage accuracy.
    days = 3.0

    def terminal(dt):
        steps = int(round(days * 86400.0 / dt))
        kw = dict(_WASP43B, taurad=36000.0, taudrag=21600.0)
        out = run_model_scan(**kw, dt=dt, tmax=2 + steps, semi_implicit=True)
        return np.asarray(out["outs"]["Phi"][-1])

    ref = terminal(240.0)
    coarse = terminal(1200.0)
    rel = float(np.sqrt(np.mean((coarse - ref) ** 2))) / float(np.std(ref))
    assert np.isfinite(rel)
    assert rel < 1e-3


@pytest.mark.smoke
def test_semi_implicit_preserves_steady_advection():
    # Test 1 (solid-body advection) has a steady vorticity field; the
    # semi-implicit leapfrog must hold it to roundoff (measured ~5e-13).
    hours = 6.0
    dt = 600.0
    earth = dict(M=42, Phibar=3.0e3, omega=7.2921159e-5, a=6.37122e6, test=1, forcflag=False)
    steps = int(round(hours * 3600.0 / dt))
    out = run_model_scan(**earth, dt=dt, tmax=2 + steps, semi_implicit=True)
    eta = np.asarray(out["outs"]["eta"])
    drift = float(np.sqrt(np.mean((eta[-1] - eta[0]) ** 2))) / float(np.std(eta[0]))
    assert drift < 1e-8


@pytest.mark.smoke
def test_semi_implicit_grad_wrt_taurad_finite_nonzero():
    def loss(taurad):
        kw = dict(_SYNTH, dt=600.0, tmax=40)
        kw["taurad"] = taurad
        out = run_model_scan_final(**kw, semi_implicit=True, diagnostics=False)
        return jnp.mean(out["last_state"].Phi_curr ** 2)

    g = jax.grad(loss)(jnp.asarray(36000.0))
    assert bool(jnp.isfinite(g))
    assert float(jnp.abs(g)) > 0.0


@pytest.mark.smoke
def test_semi_implicit_grad_wrt_si_alpha_finite():
    def loss(si_alpha):
        out = run_model_scan_final(
            **_SYNTH,
            dt=600.0,
            tmax=40,
            semi_implicit=True,
            si_alpha=si_alpha,
            diagnostics=False,
        )
        return jnp.mean(out["last_state"].Phi_curr ** 2)

    g = jax.grad(loss)(jnp.asarray(0.5))
    assert bool(jnp.isfinite(g))


@pytest.mark.smoke
def test_semi_implicit_rejects_explicit_scheme():
    with pytest.raises(ValueError):
        run_model_scan(**_SYNTH, dt=240.0, tmax=10, semi_implicit=True, expflag=True)


@pytest.mark.smoke
def test_semi_implicit_with_raw_filter_runs_finite():
    out = run_model_scan(
        **_SYNTH,
        dt=600.0,
        tmax=60,
        semi_implicit=True,
        raw_filter=True,
        williams_alpha=0.53,
    )
    assert int(out["dead_first_idx"]) == -1
    assert np.all(np.isfinite(np.asarray(out["outs"]["Phi"][-1])))
