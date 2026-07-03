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


# ---------------------------------------------------------------------------
# Secondary-eclipse ingress/egress masking (2026-07 production run)
# ---------------------------------------------------------------------------


def test_eclipse_contact_half_durations_match_esposito_geometry():
    """Contact half-durations from the Esposito et al. 2017 geometry.

    Cross-checked by hand (Seager & Mallen-Ornelas 2003 eq. 3):
    T14/2 ~ 35.5 min, T23/2 ~ 18.3 min, ingress/egress duration ~ 17 min.
    """
    h14, h23 = prep.eclipse_contact_half_durations_days()
    assert 0.0 < h23 < h14 < 0.05
    assert abs(h14 * 1440.0 - 35.5) < 1.5
    assert abs(h23 * 1440.0 - 18.3) < 1.5


def test_eclipse_edge_mask_windows():
    half14, half23 = prep.eclipse_contact_half_durations_days()
    center = 0.5 * prep.PERIOD_DAYS
    pad = 0.0
    probe = np.array([
        0.0,                                # mid-orbit: keep
        center,                             # eclipse center (full occultation): keep
        center - 0.5 * (half14 + half23),   # mid-ingress: mask
        -(center - 0.5 * (half14 + half23)),  # mid-ingress, other wrap sign: mask
        center - half14 - 0.01,             # just outside contact 1: keep
        center - half23 + 0.005,            # inside full eclipse: keep
    ])
    # fold to the phase convention of centered_orbital_phase_days
    phase = (probe + 0.5 * prep.PERIOD_DAYS) % prep.PERIOD_DAYS - 0.5 * prep.PERIOD_DAYS
    mask = prep.eclipse_edge_mask(phase, pad_days=pad)
    assert mask.tolist() == [False, False, True, True, False, False]


def test_eclipse_edge_mask_pad_widens_window():
    half14, half23 = prep.eclipse_contact_half_durations_days()
    just_outside = 0.5 * prep.PERIOD_DAYS - half14 - 0.001
    phase = np.array([(just_outside + 0.5 * prep.PERIOD_DAYS) % prep.PERIOD_DAYS - 0.5 * prep.PERIOD_DAYS])
    assert not prep.eclipse_edge_mask(phase, pad_days=0.0)[0]
    assert prep.eclipse_edge_mask(phase, pad_days=0.002)[0]


def test_prepare_masks_eclipse_edges_and_records_provenance(tmp_path: Path):
    """Points seeded in the contact windows are dropped by default and the
    provenance records the mask; mask_eclipse_edges=False keeps them."""
    h5_path = tmp_path / "tiny_eureka_eclipse.h5"

    half14, half23 = prep.eclipse_contact_half_durations_days()
    mid_edge = 0.5 * prep.PERIOD_DAYS - 0.5 * (half14 + half23)
    base = prep.TRANSIT_EPOCH_MJD + 10.0 * prep.PERIOD_DAYS
    # 12 clean points + 2 mid-ingress/egress points
    clean = np.linspace(0.05, 0.30, 12)
    times = np.concatenate([clean, [mid_edge, prep.PERIOD_DAYS - mid_edge]])

    n_time = times.size
    with h5py.File(h5_path, "w") as handle:
        handle["time"] = base + times
        handle["flux"] = np.ones((2, n_time), dtype=np.float64)
        handle["err"] = np.full((2, n_time), 2.0e-4, dtype=np.float64)
        handle["mask"] = np.zeros((2, n_time), dtype=np.int8)
        handle["wavelength"] = np.asarray([5.25, 6.25])
        handle["wave_low"] = np.asarray([5.0, 6.0])
        handle["wave_hi"] = np.asarray([5.5, 6.5])

    kwargs = dict(target_bins=4, ramp_integrations=0, error_inflation=1.0,
                  primary_transit_half_width_days=0.0)

    _, prov_masked = prep.load_and_prepare(h5_path, **kwargs)
    assert prov_masked["eclipse_edge_mask_applied"] is True
    assert prov_masked["n_masked_eclipse_edges"] == 2
    assert prov_masked["n_retained_unbinned"] == 12
    assert prov_masked["eclipse_contact_half_durations_days"]["t14_half"] > \
        prov_masked["eclipse_contact_half_durations_days"]["t23_half"]

    _, prov_unmasked = prep.load_and_prepare(h5_path, mask_eclipse_edges=False, **kwargs)
    assert prov_unmasked["eclipse_edge_mask_applied"] is False
    assert prov_unmasked["n_masked_eclipse_edges"] == 0
    assert prov_unmasked["n_retained_unbinned"] == 14


def test_binning_never_straddles_a_masked_gap(tmp_path: Path):
    """Points on both sides of a masked contact window must land in different
    bins: averaging across the gap would mix out-of-eclipse and in-eclipse flux
    (~the eclipse depth apart) and recreate the artifact the mask removes."""
    h5_path = tmp_path / "tiny_eureka_straddle.h5"

    half14, half23 = prep.eclipse_contact_half_durations_days()
    center = 0.5 * prep.PERIOD_DAYS
    base = prep.TRANSIT_EPOCH_MJD + 10.0 * prep.PERIOD_DAYS
    pre = np.linspace(center - half14 - 0.05, center - half14 - 0.005, 8)   # out of eclipse
    edge = np.array([center - 0.5 * (half14 + half23)])                     # mid-ingress (masked)
    inside = np.linspace(center - half23 + 0.005, center + half23 - 0.005, 6)  # full eclipse, clear of the pad
    times = np.concatenate([pre, edge, inside])

    flux_1d = np.where(times < center - half14, 1.003, 1.000)  # a fake eclipse depth
    n_time = times.size
    with h5py.File(h5_path, "w") as handle:
        handle["time"] = base + times
        handle["flux"] = np.tile(flux_1d, (2, 1))
        handle["err"] = np.full((2, n_time), 2.0e-4, dtype=np.float64)
        handle["mask"] = np.zeros((2, n_time), dtype=np.int8)
        handle["wavelength"] = np.asarray([5.25, 6.25])
        handle["wave_low"] = np.asarray([5.0, 6.0])
        handle["wave_hi"] = np.asarray([5.5, 6.5])

    observations, prov = prep.load_and_prepare(
        h5_path, target_bins=4, ramp_integrations=0, error_inflation=1.0,
        primary_transit_half_width_days=0.0,
    )
    assert prov["n_masked_eclipse_edges"] == 1
    assert prov["n_bin_segments"] == 2
    assert sum(prov["bins_per_segment"]) == prov["n_binned"]
    # every binned flux is purely one side or the other -- no mixture values
    f = observations["flux_obs"]
    assert np.all((np.abs(f - 1.003) < 1e-9) | (np.abs(f - 1.000) < 1e-9))
    # and no binned time sits inside a contact window
    ph = prep.centered_orbital_phase_days(observations["times_days"])
    assert not prep.eclipse_edge_mask(ph, pad_days=prov["eclipse_edge_pad_days"]).any()
