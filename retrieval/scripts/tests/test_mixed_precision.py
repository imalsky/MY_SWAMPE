"""Mixed-precision (f32 dynamics, f64 light curve) regression tests.

The JAX x64 flag is process-global and the main suite runs float32, so these
tests only run under an x64 invocation:

    MY_SWAMPE_ENABLE_X64=1 JAX_ENABLE_X64=1 conda run -n MY_SWAMP \
        python -m pytest retrieval/scripts/tests/test_mixed_precision.py -q

Contract: ``mixed_precision=False`` (default) is bit-identical pure float64;
``mixed_precision=True`` matches the pure-f64 forward to sub-ppm and its
custom fwd-JVP likelihood gradient to <1e-3 relative.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pipeline as P

pytestmark = pytest.mark.skipif(
    not bool(jax.config.jax_enable_x64),
    reason="mixed-precision tests need process-global x64 (MY_SWAMPE_ENABLE_X64=1)",
)


def _configs():
    common = dict(
        use_x64=True,
        M=42,
        dt_seconds=240.0,
        model_days=0.25,
        n_times=60,
        taurad_true_hours=10.0,
        taudrag_true_hours=6.0,
        diagnostics=False,
    )
    return P.Config(**common), P.Config(**common, mixed_precision=True)


def test_mixed_precision_requires_x64_config():
    with pytest.raises(ValueError):
        P.validate_config(P.Config(use_x64=False, mixed_precision=True))


def test_mixed_forward_matches_f64_sub_ppm():
    cfg64, cfgmx = _configs()
    pipe64 = P.build_pipeline(cfg64)
    pipemx = P.build_pipeline(cfgmx)
    f64 = np.asarray(pipe64.phase_curve_model_jit(pipe64.theta_truth))
    fmx = np.asarray(pipemx.phase_curve_model_jit(pipemx.theta_truth))
    assert fmx.dtype == np.float64  # cast back up for the light-curve stage
    assert np.max(np.abs(f64 - fmx)) * 1e6 < 0.5  # ppm of stellar flux


def test_mixed_gradient_matches_f64():
    cfg64, cfgmx = _configs()
    pipe64 = P.build_pipeline(cfg64)
    pipemx = P.build_pipeline(cfgmx)
    obs = P.generate_observations(pipe64, seed=7)
    pipemx.set_observations(obs["flux_obs"], obs_sigma=obs["obs_sigma"])

    u0 = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
    g64 = np.asarray(jax.grad(pipe64.loglikelihood_for_blackjax)(u0))
    gmx = np.asarray(jax.grad(pipemx.loglikelihood_for_blackjax)(u0))
    assert np.all(np.isfinite(gmx))
    rel = np.abs(gmx - g64) / np.maximum(np.abs(g64), 1e-12)
    assert np.max(rel) < 1e-3
