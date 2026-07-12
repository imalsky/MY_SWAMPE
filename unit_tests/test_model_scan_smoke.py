from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.smoke
def test_run_model_scan_smoke() -> None:
    """Verify that `run_model` completes a minimal scan-based rollout."""
    import jax  # noqa: F401  # pylint: disable=unused-import
    from my_swampe.model import run_model_scan

    # Minimal run: one scan step (t starts at 2 by SWAMPE convention).
    res = run_model_scan(
        M=42,
        dt=30.0,
        tmax=3,
        Phibar=3.0e3,
        omega=7.2921159e-5,
        a=6.37122e6,
        test=1,
        g=9.8,
        forcflag=False,
        diffflag=False,
        modalflag=True,
        expflag=False,
        jit_scan=False,
    )

    assert set(res.keys()) >= {"static", "t_seq", "outs", "last_state", "starttime"}

    static = res["static"]
    t_seq = np.asarray(res["t_seq"])
    outs = res["outs"]

    assert t_seq.ndim == 1
    assert t_seq.size == 1
    assert int(t_seq[0]) == 2  # default start time when not using continuation

    # Core field outputs (time, lat, lon)
    for key in ("eta", "delta", "Phi", "U", "V"):
        arr = np.asarray(outs[key])
        assert arr.shape == (t_seq.size, static.J, static.I)
        assert np.isfinite(arr).all(), f"{key} contains NaN/Inf"

    # Scalar diagnostics (time,)
    for key in ("rms", "spin_min", "phi_min", "phi_max"):
        arr = np.asarray(outs[key])
        assert arr.shape == (t_seq.size,)
        assert np.isfinite(arr).all(), f"{key} contains NaN/Inf"


@pytest.mark.smoke
@pytest.mark.parametrize("semi_implicit", [False, True])
def test_run_model_scan_test2_smoke(semi_implicit: bool) -> None:
    """Test 2 (Williamson balanced zonal flow) runs and stays finite.

    Regression guard for the velocity_init test=2 shape bug: the meridional
    wind value is latitude-independent, and the vectorized port dropped the
    reference's preallocated (J, I) array, so Vic came out (1, I) and the
    initial-state stack crashed on any test=2 run.
    """
    from my_swampe.model import run_model_scan

    res = run_model_scan(
        M=42,
        dt=600.0,
        tmax=12,
        Phibar=3.0e3,
        omega=7.2921159e-5,
        a=6.37122e6,
        test=2,
        forcflag=False,
        semi_implicit=semi_implicit,
    )
    static = res["static"]
    for key in ("eta", "delta", "Phi", "U", "V"):
        arr = np.asarray(res["outs"][key])
        assert arr.shape[1:] == (static.J, static.I)
        assert np.isfinite(arr).all(), f"{key} contains NaN/Inf"
