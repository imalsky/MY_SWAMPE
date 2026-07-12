"""Tests for the 2026-07 production-run upgrades.

Covers: the per-stage-adaptive preconditioned MALA mutation kernel (statistical
correctness on an anisotropic Gaussian + end-to-end SMC with adaptation), the
weighted-scale preconditioner helper, the new Config validation rules, and the
semi-implicit / RAW flag pass-through into my_swampe RunFlags.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import blackjax

import pipeline as P


# ---------------------------------------------------------------------------
# _weighted_scale_diag
# ---------------------------------------------------------------------------


def test_weighted_scale_diag_unit_geometric_mean_and_shape():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(200, 3)) * np.array([0.1, 1.0, 10.0])
    w = np.full(200, 1.0 / 200)
    scale = P._weighted_scale_diag(p, w, clip=100.0)
    assert scale.shape == (3,)
    assert np.isclose(np.exp(np.mean(np.log(scale))), 1.0, rtol=1e-12)
    # ordering reflects the anisotropy
    assert scale[0] < scale[1] < scale[2]


def test_weighted_scale_diag_clip_and_degenerate_weights():
    rng = np.random.default_rng(1)
    p = rng.normal(size=(50, 2)) * np.array([1e-8, 1.0])
    w = np.full(50, 1.0 / 50)
    scale = P._weighted_scale_diag(p, w, clip=5.0)
    assert scale.min() >= 1.0 / 5.0 - 1e-12 and scale.max() <= 5.0 + 1e-12
    # non-finite / zero weights fall back to ones
    bad = P._weighted_scale_diag(p, np.zeros(50), clip=5.0)
    np.testing.assert_allclose(bad, np.ones(2))
    nan_w = np.full(50, np.nan)
    np.testing.assert_allclose(P._weighted_scale_diag(p, nan_w, clip=5.0), np.ones(2))


# ---------------------------------------------------------------------------
# Preconditioned MALA kernel: statistical correctness
# ---------------------------------------------------------------------------


def _evolve_chains(kernel, logdensity, x0s, step_size, scale_diag, n_steps, seed=0):
    """Advance many chains in parallel; returns (final positions, mean acceptance)."""
    states = jax.vmap(lambda x: blackjax.mala.init(x, logdensity))(jnp.asarray(x0s))
    scale = jnp.asarray(scale_diag)
    n_chains = int(np.asarray(x0s).shape[0])

    @jax.jit
    def sweep(sts, key):
        keys = jax.random.split(key, n_chains)
        sts, infos = jax.vmap(lambda k, s: kernel(k, s, logdensity, step_size, scale))(keys, sts)
        return sts, jnp.mean(infos.acceptance_rate)

    keys = jax.random.split(jax.random.PRNGKey(seed), n_steps)
    states, accs = jax.lax.scan(sweep, states, keys)
    return np.asarray(states.position), float(np.mean(np.asarray(accs)))


@pytest.mark.parametrize("scale_diag,step_size", [((1.0, 1.0), 0.001), ((0.05, 2.0), 0.05)])
def test_preconditioned_mala_preserves_anisotropic_gaussian(scale_diag, step_size):
    """The kernel must leave the target invariant for any preconditioner.

    512 chains start as EXACT draws from the target (Gaussian, sigma=(0.05, 2.0));
    after 1500 kernel steps the cross-chain marginals must still match the target.
    A wrong MH ratio (the classic preconditioned-MALA bug) shows up as drift of
    the cross-chain std regardless of mixing speed.
    """
    sigma = np.array([0.05, 2.0])
    mu = np.array([0.3, -1.0])

    def logdensity(x):
        return -0.5 * jnp.sum(((x - jnp.asarray(mu)) / jnp.asarray(sigma)) ** 2)

    rng = np.random.default_rng(42)
    n_chains = 512
    x0s = mu[None, :] + sigma[None, :] * rng.standard_normal((n_chains, 2))

    kernel = P._build_preconditioned_mala_kernel()
    final, acc = _evolve_chains(kernel, logdensity, x0s, step_size=step_size,
                                scale_diag=np.asarray(scale_diag), n_steps=1500, seed=3)
    assert 0.05 < acc <= 1.0
    # cross-chain moments: se(mean) = sigma/sqrt(512), se(std) ~ sigma/sqrt(2*512)
    for j in range(2):
        assert abs(final[:, j].mean() - mu[j]) < 4.0 * sigma[j] / math.sqrt(n_chains)
        assert abs(final[:, j].std() - sigma[j]) < 5.0 * sigma[j] / math.sqrt(2 * n_chains)


def test_preconditioned_mala_rejects_nonfinite_proposal_density():
    """A proposal landing in a -inf region must be rejected, not poison the chain."""

    def logdensity(x):
        # hard wall: -inf outside |x| < 1
        inside = jnp.all(jnp.abs(x) < 1.0)
        return jnp.where(inside, -0.5 * jnp.sum(x * x), -jnp.inf)

    kernel = P._build_preconditioned_mala_kernel()
    x0s = np.zeros((32, 2))
    positions, _ = _evolve_chains(kernel, logdensity, x0s,
                                  step_size=0.5, scale_diag=np.ones(2), n_steps=500, seed=0)
    assert np.all(np.isfinite(positions))
    assert np.all(np.abs(positions) < 1.0)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_validate_config_stage_adapt_requires_mala():
    with pytest.raises(ValueError):
        P.validate_config(P.Config(mcmc_stage_adapt=True, smc_mcmc_kernel="hmc"))
    P.validate_config(P.Config(mcmc_stage_adapt=True, smc_mcmc_kernel="mala"))


def test_validate_config_scale_clip_bounds():
    with pytest.raises(ValueError):
        P.validate_config(P.Config(mcmc_stage_adapt=True, mcmc_scale_clip=0.5))


def test_validate_config_semi_implicit_rejects_expflag():
    with pytest.raises(ValueError):
        P.validate_config(P.Config(semi_implicit=True, expflag=True))
    P.validate_config(P.Config(semi_implicit=True))


# ---------------------------------------------------------------------------
# End-to-end SMC with per-stage adaptation (cheap Gaussian likelihood)
# ---------------------------------------------------------------------------


def _gaussian_toy_pipe(n_particles=48, num_mcmc_steps=6):
    """A Pipeline stub with the exact attribute contract run_smc_loop needs.

    Identity theta_from_u; logistic prior (as in the real pipeline); sharp
    anisotropic Gaussian likelihood so the tempered posterior concentrates and
    the fixed-step kernel would collapse without per-stage adaptation.
    """
    cfg = P.fast_cpu_config(
        smc_num_particles=n_particles, smc_num_mcmc_steps=num_mcmc_steps,
        smc_max_steps=60, smc_target_ess_frac=0.6,
        mcmc_stage_adapt=True, mcmc_stage_adapt_gain=1.0, mcmc_scale_clip=20.0,
        mcmc_target_accept_mala=0.574, mala_step_size=0.1,
        mcmc_step_size_min=1e-6, mcmc_step_size_max=5.0,
        mcmc_auto_tune=False, num_samples=64, num_chains=2,
    )
    dtype = P.float_dtype()
    mu = jnp.asarray([0.5, -0.3], dtype=dtype)
    sig = jnp.asarray([0.01, 0.5], dtype=dtype)  # 50:1 anisotropy

    def log_likelihood(u):
        return -0.5 * jnp.sum(((u - mu) / sig) ** 2)

    def log_prior_u(u):
        return jnp.sum(jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))

    def sample_prior_u(key, n):
        z = jax.random.uniform(key, shape=(n, 2), minval=1e-6, maxval=1 - 1e-6)
        return jnp.log(z) - jnp.log1p(-z)

    return P.Pipeline(cfg=cfg, dtype=dtype, n_dim=2,
                      log_prior_u=log_prior_u, sample_prior_u=sample_prior_u,
                      loglikelihood_for_blackjax=log_likelihood,
                      theta_from_u=lambda u: u), np.asarray(mu), np.asarray(sig)


@pytest.mark.slow
def test_run_smc_loop_stage_adapt_recovers_gaussian_and_keeps_diversity():
    pipe, mu, sig = _gaussian_toy_pipe()
    res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(0), progress=False)

    assert res["reached_beta1"]
    draws = res["theta_draws"].reshape(-1, 2)
    # posterior ~ likelihood here (prior is ~flat at the likelihood's scale)
    for j in range(2):
        assert abs(draws[:, j].mean() - mu[j]) < 0.5 * sig[j]
        assert 0.5 * sig[j] < draws[:, j].std() < 2.0 * sig[j]

    # the whole point of the fix: mutation must not freeze as beta -> 1
    assert res["acceptance_rate"][-1] > 0.1
    n = int(pipe.cfg.smc_num_particles)
    assert res["unique_particles"][-1] > 0.5 * n

    # the step size actually adapted (non-constant history)...
    hist = res["step_size_history"]
    assert hist.size >= 3 and (hist.max() / hist.min()) > 1.5
    # ...and the preconditioner learned the 50:1 anisotropy (direction, not magnitude)
    scale = res["scale_diag_final"]
    assert scale[1] / scale[0] > 3.0


@pytest.mark.slow
def test_run_smc_loop_fixed_step_path_unchanged():
    """Back-compat: mcmc_stage_adapt=False keeps the historical fixed-step path."""
    pipe, mu, sig = _gaussian_toy_pipe()
    pipe.cfg = P.fast_cpu_config(
        smc_num_particles=32, smc_num_mcmc_steps=6, smc_max_steps=60,
        smc_target_ess_frac=0.6, mcmc_stage_adapt=False, mcmc_auto_tune=False,
        mala_step_size=0.05, num_samples=32, num_chains=2,
    )
    res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(1), progress=False)
    assert res["reached_beta1"]
    assert res["step_size_history"].size > 0
    # fixed path: constant step size across stages
    assert float(res["step_size_history"].max()) == float(res["step_size_history"].min())


# ---------------------------------------------------------------------------
# Semi-implicit / RAW wiring into my_swampe RunFlags
# ---------------------------------------------------------------------------


def test_semi_implicit_raw_flags_reach_runflags_and_run():
    cfg = P.fast_cpu_config(model_days=0.5, n_times=24, dt_seconds=600.0,
                            semi_implicit=True, si_alpha=0.5,
                            raw_filter=True, williams_alpha=0.53, alpha=0.05)
    pipe = P.build_pipeline(cfg)
    assert bool(pipe.flags.semi_implicit) is True
    assert bool(pipe.flags.raw_filter) is True
    assert np.isclose(float(pipe.flags.williams_alpha), 0.53)
    assert np.isclose(float(pipe.flags.si_alpha), 0.5)
    # forward model runs and stays finite under the new scheme
    flux = np.asarray(pipe.phase_curve_model_jit(pipe.theta_truth))
    assert flux.shape == (24,)
    assert np.all(np.isfinite(flux))


def test_default_config_keeps_locked_scheme_flags_off():
    cfg = P.fast_cpu_config(model_days=0.5, n_times=8)
    pipe = P.build_pipeline(cfg)
    assert bool(pipe.flags.semi_implicit) is False
    assert bool(pipe.flags.raw_filter) is False
