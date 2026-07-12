# ruff: noqa: E741
"""Tests for the opt-in Robert–Asselin–Williams (RAW) filter (CLAUDE.md 13.2).

Contract under test:
- ``raw_filter=False`` (default) is the locked classic-RA behavior.
- ``raw_filter=True, williams_alpha=1.0`` reproduces the default bit-for-bit
  (the Williams adjustment of the new level vanishes and the modified-Euler
  scheme never reads the Fourier prev carry, which is the only other change).
- ``williams_alpha != 1`` actually changes the trajectory.
- The filter parameters stay differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from my_swampe.model import run_model_scan, run_model_scan_final


_COMMON = dict(
    M=42,
    dt=240.0,
    tmax=40,
    Phibar=3.0e5,
    omega=3.2e-5,
    a=8.2e7,
    taurad=36000.0,
    taudrag=21600.0,
    DPhieq=1.0e6,
    test=None,
)


@pytest.mark.smoke
def test_raw_alpha_one_is_bit_identical_to_classic_ra():
    base = run_model_scan(**_COMMON)
    raw = run_model_scan(**_COMMON, raw_filter=True, williams_alpha=1.0)

    for key in ("eta", "delta", "Phi", "U", "V"):
        a = np.asarray(base["outs"][key])
        b = np.asarray(raw["outs"][key])
        np.testing.assert_array_equal(a, b, err_msg=f"RAW(williams_alpha=1) changed {key}")


@pytest.mark.smoke
def test_raw_williams_optimum_changes_trajectory():
    base = run_model_scan(**_COMMON)
    raw = run_model_scan(**_COMMON, raw_filter=True, williams_alpha=0.53)

    dphi = np.max(np.abs(np.asarray(base["outs"]["Phi"][-1]) - np.asarray(raw["outs"]["Phi"][-1])))
    assert np.isfinite(dphi)
    assert dphi > 0.0


@pytest.mark.smoke
def test_raw_filter_rejects_explicit_scheme():
    with pytest.raises(ValueError):
        run_model_scan(**_COMMON, expflag=True, raw_filter=True)


@pytest.mark.smoke
def test_grad_wrt_williams_alpha_is_finite():
    def loss(w_alpha):
        out = run_model_scan_final(
            **_COMMON,
            raw_filter=True,
            williams_alpha=w_alpha,
            diagnostics=False,
        )
        return jnp.mean(out["last_state"].Phi_curr ** 2)

    g = jax.grad(loss)(jnp.asarray(0.53))
    assert bool(jnp.isfinite(g))
