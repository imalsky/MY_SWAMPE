"""End-to-end wiring tests for the WASP-43b real-data pilot (the PBS path).

Checks the actual pilot config JSON + prepared observations, then runs the
pipeline exactly as `run_nas_wasp43b.pbs` -> `run_smc.py` will -- in float64,
via a subprocess, because x64 is process-global and this suite runs float32.
(The pilot config declares use_x64=true; the float32 path is NOT supported for
this config -- eclipse-contact derivatives at b=0.689 lose precision.)

Skipped when the (gitignored) observations.npz has not been prepared locally.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

SUITE = Path(__file__).resolve().parents[2] / "wasp_43b_test"
CFG_JSON = SUITE / "config" / "wasp43b_pilot_gpu.json"
OBS_NPZ = SUITE / "outputs" / "observations.npz"
SCRIPTS_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (CFG_JSON.exists() and OBS_NPZ.exists()),
    reason="WASP-43b pilot config or prepared observations.npz not present",
)


def test_pilot_config_json_sane():
    cfg = {k: v for k, v in json.loads(CFG_JSON.read_text()).items() if not k.startswith("_")}
    assert cfg["use_x64"] is True
    assert cfg["generate_synthetic_data"] is False
    assert cfg["likelihood_baseline_mode"] == "linear_time"
    assert cfg["emission_model"] == "planck"
    # a_planet_m is the SW sphere radius (a planet radius!), not the orbital
    # semi-major axis -- the config regression that motivated this test.
    assert 5.0e7 < cfg["a_planet_m"] < 1.0e8
    assert cfg["dt_seconds"] == 120.0 and cfg["K6"] == 5.0e33
    assert cfg["Phibar"] == 4.0e6 and cfg["DPhieq"] == 3.5e6  # init/center values
    for flag in ("infer_tau_rad", "infer_tau_drag", "infer_planet_fpfs",
                 "infer_Phibar", "infer_DPhieq", "infer_noise_inflation"):
        assert cfg[flag] is True
    assert cfg["prior_Phibar_min"] == 2.0e6 and cfg["prior_Phibar_max"] == 8.0e6
    assert cfg["prior_DPhieq_min"] == 5.0e5 and cfg["prior_DPhieq_max"] == 5.0e6
    assert cfg["orbital_period_override_days"] == pytest.approx(0.813474037)
    assert cfg["omega_rad_s"] == pytest.approx(2 * np.pi / (0.813474037 * 86400.0))


def test_pilot_observations_npz_sane():
    obs = np.load(OBS_NPZ)
    for key in ("times_days", "flux_obs", "obs_sigma_vec", "band_wavelengths_um", "band_weights"):
        assert key in obs.files, f"missing {key}"
    assert obs["times_days"].shape == (320,)
    sig = np.asarray(obs["obs_sigma_vec"])
    assert np.all(np.isfinite(sig)) and np.all(sig > 0.0)
    wl = np.asarray(obs["band_wavelengths_um"])
    w = np.asarray(obs["band_weights"])
    assert wl.shape == (11,) and w.shape == (11,)
    assert wl.min() == pytest.approx(5.25) and wl.max() == pytest.approx(10.25)
    assert np.all(w > 0.0) and np.sum(w) == pytest.approx(1.0, rel=1e-6)


_X64_CHECK = textwrap.dedent(
    """
    import json, math, sys
    from dataclasses import replace
    from pathlib import Path
    import numpy as np

    sys.path.insert(0, sys.argv[1])          # retrieval/scripts
    SUITE = Path(sys.argv[2])                 # retrieval/wasp_43b_test

    import jax
    import jax.numpy as jnp
    import pipeline as P

    assert jax.config.jax_enable_x64, "subprocess must run in x64"

    ov = {k: v for k, v in json.loads((SUITE / "config/wasp43b_pilot_gpu.json").read_text()).items()
          if not k.startswith("_")}
    ov.pop("out_dir", None)
    ov.update(model_days=1.5)                 # short spin-up; wiring not science
    cfg = P.gpu_config(**ov)

    obs = np.load(SUITE / "outputs/observations.npz")
    cfg = replace(cfg,
                  observation_times_days=tuple(float(x) for x in obs["times_days"]),
                  n_times=int(obs["times_days"].size),
                  planck_band_wavelengths_m=tuple(float(x) * 1e-6 for x in obs["band_wavelengths_um"]),
                  planck_band_weights=tuple(float(x) for x in obs["band_weights"]))
    P.validate_config(cfg)
    pipe = P.build_pipeline(cfg)
    pipe.set_observations(np.asarray(obs["flux_obs"]), obs_sigma=np.asarray(obs["obs_sigma_vec"]))

    assert pipe.param_names == ["tau_rad_hours", "tau_drag_hours", "planet_fpfs",
                                "Phibar", "DPhieq", "noise_inflation"]
    assert pipe.use_custom_grads and np.asarray(pipe.obs_sigma_jax).shape == (320,)
    assert pipe._fast_path_ok is False   # Phibar/DPhieq inference rebuilds static per eval

    flux = np.asarray(pipe.phase_curve_model_jit(pipe.theta_truth))
    assert flux.shape == (320,) and np.isfinite(flux).all(), "forward not finite"
    t = np.asarray(pipe.times_days)
    Porb = float(cfg.orbital_period_override_days)
    for ecl in (-0.5 * Porb, 0.5 * Porb):
        i = int(np.argmin(np.abs(t - ecl)))
        assert abs(flux[i]) < 1e-6, "planet flux not ~0 in eclipse"
    # orientation on mirrored pairs around the +P/2 eclipse (within data coverage)
    ecl = 0.5 * Porb
    asym = []
    for d in np.linspace(0.06, 0.11, 6) * Porb:
        ib = int(np.argmin(np.abs(t - (ecl - d))))
        ia = int(np.argmin(np.abs(t - (ecl + d))))
        asym.append(flux[ib] - flux[ia])
    assert np.mean(asym) > 0.0, f"phase curve not brighter before eclipse: {np.mean(asym)}"

    lo, hi = pipe.param_prior_lo, pipe.param_prior_hi
    def u_for(theta):
        z = np.empty(len(pipe.specs))
        for i, spec in enumerate(pipe.specs):
            if spec.prior_type == "uniform":
                z[i] = (theta[i] - lo[i]) / (hi[i] - lo[i])
            else:
                z[i] = (np.log10(theta[i]) - np.log10(lo[i])) / (np.log10(hi[i]) - np.log10(lo[i]))
        z = np.clip(z, 1e-9, 1 - 1e-9)
        return jnp.asarray(np.log(z) - np.log1p(-z), pipe.dtype)

    #        tau_rad tau_drag fpfs   Phibar DPhieq noise_k
    th1 = np.array([10.0, 6.0, 0.003, 4.0e6, 3.5e6, 1.0])
    th2 = np.array([10.0, 6.0, 0.003, 4.0e6, 3.5e6, 2.0])
    ll1 = float(pipe.log_likelihood_u(u_for(th1))); ll2 = float(pipe.log_likelihood_u(u_for(th2)))
    assert math.isfinite(ll1) and math.isfinite(ll2), "likelihood not finite"
    mu = np.asarray(pipe.observed_flux_model_jit(jnp.asarray(th1, pipe.dtype)))
    sig = np.asarray(pipe.obs_sigma_jax)
    chi2 = float(np.sum(((np.asarray(pipe.flux_obs) - mu) / sig) ** 2))
    expected = -0.5 * chi2 * (1.0 / 4.0 - 1.0) - mu.size * math.log(2.0)
    assert abs((ll2 - ll1) - expected) < 1e-4 * abs(expected), "noise-inflation identity failed"

    # sampled Phibar must reach the emission map, not just the dynamics: shifting
    # Phibar by +1e6 shifts the mean brightness temperature by ~ +1e6/R_d
    th3 = th1.copy(); th3[3] = 5.0e6
    T1 = pipe.compute_maps_for_theta(jnp.asarray(th1, pipe.dtype))["T"]
    T3 = pipe.compute_maps_for_theta(jnp.asarray(th3, pipe.dtype))["T"]
    dT = float(np.mean(T3) - np.mean(T1))
    assert abs(dT - 1.0e6 / 3.78e3) < 0.25 * (1.0e6 / 3.78e3), \
        f"Phibar not threaded into the temperature map (mean dT={dT:.0f} K, expected ~265 K)"

    g = np.asarray(jax.grad(pipe.loglikelihood_for_blackjax)(u_for(th1)))
    assert g.shape == (6,) and np.isfinite(g).all() and np.all(np.abs(g) > 0.0), f"bad grad {g}"
    print("X64 REAL-DATA PATH OK")
    """
)


def test_pilot_realdata_path_x64_subprocess():
    """Run the pilot wiring checks in float64 (the precision the PBS job uses)."""
    env = dict(os.environ)
    env.update(MY_SWAMPE_ENABLE_X64="1", JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    res = subprocess.run(
        [sys.executable, "-c", _X64_CHECK, str(SCRIPTS_DIR), str(SUITE)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr[-3000:]}"
    assert "X64 REAL-DATA PATH OK" in res.stdout
