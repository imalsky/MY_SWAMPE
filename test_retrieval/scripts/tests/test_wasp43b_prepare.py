"""Tests for WASP-43 b real-data preparation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py required for the WASP-43b preparation tests")

_SUITE_SCRIPTS = Path(__file__).resolve().parents[2] / "wasp_43b_test" / "scripts"
if str(_SUITE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SUITE_SCRIPTS))

import prepare_wasp43b_observations as prep  # noqa: E402


def write_tiny_light_curve(path: Path) -> None:
    """Write a tiny Eureka-like HDF5 light-curve product."""
    n_time = 16
    n_chan = 3
    time = prep.TRANSIT_EPOCH_MJD + 10.0 * prep.PERIOD_DAYS + np.linspace(0.12, 0.70, n_time)
    flux = np.ones((n_chan, n_time), dtype=np.float64)
    flux[0] += np.linspace(0.0, 1.0e-3, n_time)
    flux[1] += np.linspace(2.0e-4, 1.2e-3, n_time)
    flux[2] += 5.0e-3
    err = np.full((n_chan, n_time), 2.0e-4, dtype=np.float64)
    mask = np.zeros((n_chan, n_time), dtype=np.int8)
    mask[0, 0] = 1
    flux[1, 1] = np.nan

    with h5py.File(path, "w") as handle:
        handle["time"] = time
        handle["flux"] = flux
        handle["err"] = err
        handle["mask"] = mask
        handle["wavelength"] = np.asarray([5.25, 6.25, 11.25])
        handle["wave_low"] = np.asarray([5.0, 6.0, 11.0])
        handle["wave_hi"] = np.asarray([5.5, 6.5, 11.5])


def test_prepare_tiny_hdf5_bins_positive_errors(tmp_path: Path):
    h5_path = tmp_path / "tiny_eureka.h5"
    write_tiny_light_curve(h5_path)

    observations, provenance = prep.load_and_prepare(
        h5_path,
        target_bins=4,
        ramp_integrations=0,
        error_inflation=1.25,
        primary_transit_half_width_days=0.0,
    )

    times = observations["times_days"]
    flux = observations["flux_obs"]
    sigma = observations["obs_sigma_vec"]

    assert times.shape == (4,)
    assert flux.shape == (4,)
    assert sigma.shape == (4,)
    assert np.all(np.isfinite(times))
    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)
    assert np.all(np.diff(times) > 0.0)
    assert provenance["selected_channels"] == 2
    assert provenance["n_binned"] == 4
