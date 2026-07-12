#!/usr/bin/env python3
"""Prepare WASP-43 b JWST/MIRI reduced light curves for MY_SWAMPE retrieval."""

from __future__ import annotations

import argparse
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple
from zipfile import ZipFile

import h5py
import numpy as np

SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = SUITE_ROOT / "data" / "raw" / "WASP43b_MIRI_Data.zip"
DEFAULT_H5_IN_ARCHIVE = "WASP43b_MIRI_Data/1_Light_Curves/eureka_v1.h5"
DEFAULT_OUTPUT = SUITE_ROOT / "outputs" / "observations.npz"
PROVENANCE_DIR = SUITE_ROOT / "data" / "provenance"

# Linear ephemeris from Ivshina & Winn 2022 (ApJS 259, 62), BJD_TDB. Predicted
# transit during the JWST visit: BJD_TDB 2459915.12067 (ExoClock agrees to 16 s).
# Do NOT use the Hellier et al. 2011 discovery ephemeris (P=0.813475,
# T0=2455528.86774): propagated to the JWST epoch it lands ~6 min late
# (~1.9 deg of orbital phase), and its epoch is tabulated in HJD, not BJD_TDB.
PERIOD_DAYS = 0.813474037
TRANSIT_EPOCH_BJD_TDB = 2457423.449697
TRANSIT_EPOCH_MJD = TRANSIT_EPOCH_BJD_TDB - 2400000.5
WAVELENGTH_MIN_UM = 5.0
WAVELENGTH_MAX_UM = 10.5
RAMP_INTEGRATIONS = 779
# Bell et al. 2024's broadband fits found scatter ~1.25x the estimated photon
# noise ("scatter_multi"); we inflate the binned errors by the same factor. The
# retrieval additionally infers a free noise-inflation parameter on top of this.
ERROR_INFLATION = 1.25
PRIMARY_TRANSIT_HALF_WIDTH_DAYS = 0.06
TARGET_BINS = 320
# --- secondary-eclipse ingress/egress masking -------------------------------
# The retrieval's eclipse model (jaxoplanet, fixed geometry, no timing freedom)
# mismatches the sharp observed contacts by a few minutes, producing -1000 to
# -1500 ppm (~15-19 sigma) residual spikes confined to ingress/egress in the
# 2026-07 pilot. Those few points dominate the chi^2 and inflate the fitted
# noise-inflation parameter, so we mask the contact windows (standard practice
# for phase-curve retrievals with a fixed eclipse model). In-eclipse and
# out-of-eclipse points are all kept. Geometry: Esposito et al. 2017 (A&A 601,
# A53) - the same values fixed in the retrieval config; e=0 (Bell et al. 2024).
STAR_MASS_MSUN = 0.688
STAR_RADIUS_RSUN = 0.6506
PLANET_RADIUS_RJUP = 1.006
IMPACT_PARAM = 0.689
ECLIPSE_EDGE_PAD_DAYS = 0.002  # ~ one 320-bin spacing on each side of a contact
GM_SUN = 1.32712440018e20  # m^3/s^2
R_SUN_M = 6.957e8
R_JUP_M = 7.1492e7  # IAU nominal equatorial (matches the retrieval's RJUP_TO_RSUN)
# Stellar effective temperature for the per-channel stellar-Planck correction of
# the band weights (Bonomo et al. 2017, as adopted by Bell et al. 2024).
T_STAR_K = 4400.0
H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
K_BOLTZ = 1.380649e-23


def read_h5_bytes(input_path: Path, member: str = DEFAULT_H5_IN_ARCHIVE) -> bytes:
    """Read an HDF5 file from a direct path or from the Zenodo zip archive."""
    if input_path.suffix.lower() in {".h5", ".hdf5"}:
        return input_path.read_bytes()
    with ZipFile(input_path) as archive:
        return archive.read(member)


def nearest_transit_time_mjd(times_mjd: np.ndarray) -> float:
    """Return the transit epoch nearest the median observation time."""
    epoch = np.round((float(np.nanmedian(times_mjd)) - TRANSIT_EPOCH_MJD) / PERIOD_DAYS)
    return TRANSIT_EPOCH_MJD + epoch * PERIOD_DAYS


def centered_orbital_phase_days(times_days: np.ndarray) -> np.ndarray:
    """Return time from nearest primary transit in days, in [-P/2, P/2)."""
    return (times_days + 0.5 * PERIOD_DAYS) % PERIOD_DAYS - 0.5 * PERIOD_DAYS


def eclipse_contact_half_durations_days() -> Tuple[float, float]:
    """Half-durations (days) of the secondary eclipse: (contact1-4, contact2-3).

    Circular orbit, occultation geometry identical to the transit's (e=0), with
    the semi-major axis from Kepler's third law (planet mass negligible):
    T14 = P/pi * asin( sqrt((1+k)^2 - b^2) / (a/R* sin i) ), T23 likewise with
    (1-k)^2. Seager & Mallen-Ornelas 2003, eq. 3.
    """
    period_s = PERIOD_DAYS * 86400.0
    a_m = (GM_SUN * STAR_MASS_MSUN * period_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
    a_over_rstar = a_m / (STAR_RADIUS_RSUN * R_SUN_M)
    k = (PLANET_RADIUS_RJUP * R_JUP_M) / (STAR_RADIUS_RSUN * R_SUN_M)
    b = IMPACT_PARAM
    cos_i = b / a_over_rstar
    sin_i = math.sqrt(1.0 - cos_i**2)
    x14 = math.sqrt((1.0 + k) ** 2 - b**2) / (a_over_rstar * sin_i)
    x23 = math.sqrt((1.0 - k) ** 2 - b**2) / (a_over_rstar * sin_i)
    t14 = PERIOD_DAYS / math.pi * math.asin(min(x14, 1.0))
    t23 = PERIOD_DAYS / math.pi * math.asin(min(x23, 1.0))
    return 0.5 * t14, 0.5 * t23


def eclipse_edge_mask(phase_days: np.ndarray, *, pad_days: float) -> np.ndarray:
    """True for points inside the eclipse ingress/egress contact windows (+/- pad).

    ``phase_days`` is time from primary transit folded to [-P/2, P/2); the
    eclipse center sits at the +/-P/2 wrap point, so the distance from eclipse
    center is d = P/2 - |phase|. Ingress/egress is T23/2 <= d <= T14/2.
    """
    half14, half23 = eclipse_contact_half_durations_days()
    d = 0.5 * PERIOD_DAYS - np.abs(np.asarray(phase_days, dtype=np.float64))
    return (d >= half23 - float(pad_days)) & (d <= half14 + float(pad_days))


def combine_spectral_channels(
    flux: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    wavelength: np.ndarray,
    wave_low: np.ndarray,
    wave_hi: np.ndarray,
    *,
    wavelength_min_um: float,
    wavelength_max_um: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine selected spectroscopic channels by inverse-variance weighting."""
    channel_sel = (
        np.isfinite(wavelength)
        & np.isfinite(wave_low)
        & np.isfinite(wave_hi)
        & (wave_low >= wavelength_min_um)
        & (wave_hi <= wavelength_max_um)
    )
    if not np.any(channel_sel):
        raise ValueError("No spectral channels selected for the requested wavelength range.")

    flux_sel = np.asarray(flux[channel_sel], dtype=np.float64)
    err_sel = np.asarray(err[channel_sel], dtype=np.float64)
    mask_sel = np.asarray(mask[channel_sel], dtype=bool)

    valid = np.isfinite(flux_sel) & np.isfinite(err_sel) & (err_sel > 0.0) & (~mask_sel)
    weights = np.where(valid, 1.0 / np.square(err_sel), 0.0)
    sum_weights = np.sum(weights, axis=0)
    combined_flux = np.divide(
        np.sum(weights * np.where(valid, flux_sel, 0.0), axis=0),
        sum_weights,
        out=np.full(sum_weights.shape, np.nan, dtype=np.float64),
        where=sum_weights > 0.0,
    )
    combined_err = np.divide(
        1.0,
        np.sqrt(sum_weights),
        out=np.full(sum_weights.shape, np.nan, dtype=np.float64),
        where=sum_weights > 0.0,
    )
    return combined_flux, combined_err, channel_sel


def band_model_weights(
    err: np.ndarray,
    mask: np.ndarray,
    wavelength_um: np.ndarray,
    channel_sel: np.ndarray,
    *,
    t_star_k: float = T_STAR_K,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Per-channel weights for the retrieval's band-integrated Planck model.

    The combined light curve is an inverse-variance-weighted mean of per-channel
    *relative* (median-normalized) fluxes, so its planet signal is
    sum_c w_c * Fp_c/Fs_c with w_c the data weights. Since
    Fp_c/Fs_c ∝ expm1(x_c(T_star)) / expm1(x_c(T_planet)) (the lambda^-5 Planck
    prefactors cancel in the ratio), the model must weight channel c by
    w_c * expm1(h c / (lambda_c k_B T_star)). Returns (wavelengths_um,
    normalized model weights, effective wavelength of the data weights).
    """
    wl_sel = np.asarray(wavelength_um[channel_sel], dtype=np.float64)
    err_sel = np.asarray(err[channel_sel], dtype=np.float64)
    mask_sel = np.asarray(mask[channel_sel], dtype=bool)
    valid = np.isfinite(err_sel) & (err_sel > 0.0) & (~mask_sel)
    inv_var = np.where(valid, 1.0 / np.square(err_sel), np.nan)
    w_data = np.nanmedian(inv_var, axis=1)
    if not (np.all(np.isfinite(w_data)) and np.all(w_data > 0.0)):
        raise ValueError("Per-channel data weights are not all finite and positive.")
    lam_m = wl_sel * 1.0e-6
    x_star = (H_PLANCK * C_LIGHT) / (lam_m * K_BOLTZ * float(t_star_k))
    w_model = w_data * np.expm1(x_star)
    w_model = w_model / np.sum(w_model)
    lambda_eff_um = float(np.sum(w_data * wl_sel) / np.sum(w_data))
    return wl_sel, w_model, lambda_eff_um


def _bin_sorted(
    times_sorted: np.ndarray,
    flux_sorted: np.ndarray,
    err_sorted: np.ndarray,
    n_bins: int,
) -> Tuple[list, list, list]:
    """Inverse-variance-bin an already-sorted segment into ``n_bins`` bins (>= 1)."""
    time_out, flux_out, err_out = [], [], []
    for group in np.array_split(np.arange(times_sorted.size), int(n_bins)):
        sigma = err_sorted[group]
        weights = 1.0 / np.square(sigma)
        sum_weights = np.sum(weights)
        time_out.append(float(np.sum(weights * times_sorted[group]) / sum_weights))
        flux_out.append(float(np.sum(weights * flux_sorted[group]) / sum_weights))
        err_out.append(float(math.sqrt(1.0 / sum_weights)))
    return time_out, flux_out, err_out


def inverse_variance_bin(
    times_days: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    *,
    target_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a time series into approximately equal-count inverse-variance bins."""
    order = np.argsort(times_days)
    n_bins = min(int(target_bins), times_days.size)
    if n_bins < 2:
        raise ValueError("Need at least two valid points after masking.")
    time_out, flux_out, err_out = _bin_sorted(times_days[order], flux[order], err[order], n_bins)
    return np.asarray(time_out), np.asarray(flux_out), np.asarray(err_out)


def segmented_inverse_variance_bin(
    times_days: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    keep_sorted_positions: np.ndarray,
    *,
    target_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Inverse-variance binning that never lets a bin straddle a masked gap.

    ``times_days``/``flux``/``err`` are the RETAINED points in time order;
    ``keep_sorted_positions`` gives each retained point's index in the full
    time-sorted (pre-mask) series, so a jump > 1 marks a masked gap (ramp,
    primary transit, eclipse contact window). Points are split into contiguous
    segments at those gaps and each segment is binned separately — an
    equal-count bin averaging flux across an eclipse-contact gap would mix
    out-of-eclipse and in-eclipse flux (~the eclipse depth apart), recreating
    exactly the contact-window artifact the mask removes.

    ``target_bins`` is allocated across segments proportionally to their point
    counts (largest-remainder rounding, >= 1 bin and <= n_points per segment).
    Returns (times, flux, err, segment_sizes_in_bins).
    """
    pos = np.asarray(keep_sorted_positions, dtype=np.int64)
    n = pos.size
    if n < 2:
        raise ValueError("Need at least two valid points after masking.")
    breaks = np.nonzero(np.diff(pos) > 1)[0] + 1
    segments = np.split(np.arange(n), breaks)

    total = float(n)
    raw = [int(target_bins) * len(s) / total for s in segments]
    n_bins = [min(len(s), max(1, int(math.floor(r)))) for r, s in zip(raw, segments)]
    # largest-remainder distribution of the shortfall
    frac_order = sorted(range(len(segments)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    k = 0
    while sum(n_bins) < min(int(target_bins), int(total)) and k < 10 * len(segments):
        i = frac_order[k % len(segments)]
        if n_bins[i] < len(segments[i]):
            n_bins[i] += 1
        k += 1

    time_out, flux_out, err_out, seg_bins = [], [], [], []
    for seg, nb in zip(segments, n_bins):
        t, f, e = _bin_sorted(times_days[seg], flux[seg], err[seg], nb)
        time_out += t
        flux_out += f
        err_out += e
        seg_bins.append(int(nb))
    return np.asarray(time_out), np.asarray(flux_out), np.asarray(err_out), seg_bins


def load_and_prepare(
    input_path: Path,
    *,
    member: str = DEFAULT_H5_IN_ARCHIVE,
    target_bins: int = TARGET_BINS,
    ramp_integrations: int = RAMP_INTEGRATIONS,
    error_inflation: float = ERROR_INFLATION,
    primary_transit_half_width_days: float = PRIMARY_TRANSIT_HALF_WIDTH_DAYS,
    wavelength_min_um: float = WAVELENGTH_MIN_UM,
    wavelength_max_um: float = WAVELENGTH_MAX_UM,
    mask_eclipse_edges: bool = True,
    eclipse_edge_pad_days: float = ECLIPSE_EDGE_PAD_DAYS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load the reduced MIRI light curve and return MY_SWAMPE-ready observations."""
    h5_bytes = read_h5_bytes(input_path, member)
    with h5py.File(BytesIO(h5_bytes), "r") as handle:
        time_mjd = np.asarray(handle["time"], dtype=np.float64)
        flux = np.asarray(handle["flux"], dtype=np.float64)
        err = np.asarray(handle["err"], dtype=np.float64)
        mask = np.asarray(handle["mask"], dtype=np.int8)
        wavelength = np.asarray(handle["wavelength"], dtype=np.float64)
        wave_low = np.asarray(handle["wave_low"], dtype=np.float64)
        wave_hi = np.asarray(handle["wave_hi"], dtype=np.float64)

    flux_white, err_white, channel_sel = combine_spectral_channels(
        flux,
        err,
        mask,
        wavelength,
        wave_low,
        wave_hi,
        wavelength_min_um=wavelength_min_um,
        wavelength_max_um=wavelength_max_um,
    )
    band_wl_um, band_weights, lambda_eff_um = band_model_weights(err, mask, wavelength, channel_sel)

    transit_mjd = nearest_transit_time_mjd(time_mjd)
    times_days = time_mjd - transit_mjd
    phase_days = centered_orbital_phase_days(times_days)

    finite = np.isfinite(times_days) & np.isfinite(flux_white) & np.isfinite(err_white) & (err_white > 0.0)
    ramp_mask = np.arange(times_days.size) < int(ramp_integrations)
    primary_mask = np.abs(phase_days) < float(primary_transit_half_width_days)
    if mask_eclipse_edges:
        edge_mask = eclipse_edge_mask(phase_days, pad_days=eclipse_edge_pad_days)
    else:
        edge_mask = np.zeros_like(primary_mask)
    keep = finite & (~ramp_mask) & (~primary_mask) & (~edge_mask)

    # Segment-aware binning: bins are split at every masked gap so no bin ever
    # averages flux across the primary transit or an eclipse contact window.
    order = np.argsort(times_days)
    keep_sorted = keep[order]
    sorted_positions = np.nonzero(keep_sorted)[0]
    kept_idx = order[keep_sorted]
    binned_time, binned_flux, binned_err, segment_bins = segmented_inverse_variance_bin(
        times_days[kept_idx],
        flux_white[kept_idx],
        err_white[kept_idx],
        sorted_positions,
        target_bins=target_bins,
    )
    binned_err = binned_err * float(error_inflation)

    observations = {
        "times_days": binned_time.astype(np.float64),
        "flux_obs": binned_flux.astype(np.float64),
        "obs_sigma": np.asarray(float(np.mean(binned_err)), dtype=np.float64),
        "obs_sigma_vec": binned_err.astype(np.float64),
        "orbital_period_days": np.asarray(PERIOD_DAYS, dtype=np.float64),
        "time_transit_days": np.asarray(0.0, dtype=np.float64),
        "source_time_mjd": time_mjd.astype(np.float64),
        "band_wavelengths_um": band_wl_um.astype(np.float64),
        "band_weights": band_weights.astype(np.float64),
    }
    provenance = {
        "target": "WASP-43 b",
        "input_path": str(input_path),
        "archive_member": member if input_path.suffix.lower() not in {".h5", ".hdf5"} else None,
        "data_product": "Zenodo 10.5281/zenodo.10525170, Eureka v1 light curves",
        "selected_wavelength_um": [float(wavelength_min_um), float(wavelength_max_um)],
        "selected_channels": int(np.sum(channel_sel)),
        "n_integrations": int(time_mjd.size),
        "n_finite_after_channel_combine": int(np.sum(finite)),
        "n_masked_ramp": int(np.sum(ramp_mask)),
        "n_masked_ramp_finite": int(np.sum(ramp_mask & finite)),
        "n_masked_primary_transit": int(np.sum(primary_mask & finite & (~ramp_mask))),
        "eclipse_edge_mask_applied": bool(mask_eclipse_edges),
        "n_masked_eclipse_edges": int(np.sum(edge_mask & finite & (~ramp_mask) & (~primary_mask))),
        "eclipse_edge_pad_days": float(eclipse_edge_pad_days),
        "eclipse_contact_half_durations_days": {
            "t14_half": eclipse_contact_half_durations_days()[0],
            "t23_half": eclipse_contact_half_durations_days()[1],
        },
        "eclipse_edge_geometry": {
            "star_mass_msun": STAR_MASS_MSUN,
            "star_radius_rsun": STAR_RADIUS_RSUN,
            "planet_radius_rjup": PLANET_RADIUS_RJUP,
            "impact_param": IMPACT_PARAM,
            "source": "Esposito et al. 2017 (A&A 601, A53); e=0",
            "rationale": "fixed-geometry eclipse model mismatches observed contacts by a few min; "
                         "masking the ingress/egress windows removes ~15-19 sigma residual spikes "
                         "(2026-07 pilot) that otherwise dominate chi^2 and inflate noise_inflation",
        },
        "n_retained_unbinned": int(np.sum(keep)),
        "n_binned": int(binned_time.size),
        "n_bin_segments": len(segment_bins),
        "bins_per_segment": [int(x) for x in segment_bins],
        "target_bins": int(target_bins),
        "error_inflation": float(error_inflation),
        "primary_transit_half_width_days": float(primary_transit_half_width_days),
        "transit_epoch_mjd_used": float(transit_mjd),
        "period_days": float(PERIOD_DAYS),
        "ephemeris_source": "Ivshina & Winn 2022 (ApJS 259, 62), BJD_TDB",
        "flux_units": "relative system flux",
        "error_units": "relative system flux",
        "band_wavelengths_um": [float(x) for x in band_wl_um],
        "band_weights_model": [float(x) for x in band_weights],
        "band_effective_wavelength_um": float(lambda_eff_um),
        "band_t_star_k": float(T_STAR_K),
    }
    return observations, provenance


def save_observations(output_path: Path, observations: Dict[str, np.ndarray]) -> None:
    """Save the prepared observation bundle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **observations)


def write_preparation_provenance(provenance: Dict[str, Any]) -> None:
    """Write preparation metadata for auditability."""
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROVENANCE_DIR / "wasp43b_preparation.json"
    path.write_text(json.dumps(provenance, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ARCHIVE, help="Zenodo zip archive or direct HDF5 path.")
    parser.add_argument("--member", default=DEFAULT_H5_IN_ARCHIVE, help="HDF5 member inside the zip archive.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output observations.npz path.")
    parser.add_argument("--target-bins", type=int, default=TARGET_BINS, help="Number of inverse-variance time bins.")
    parser.add_argument("--ramp-integrations", type=int, default=RAMP_INTEGRATIONS, help="Initial integrations to mask.")
    parser.add_argument("--error-inflation", type=float, default=ERROR_INFLATION, help="Multiplicative error inflation.")
    parser.add_argument(
        "--primary-transit-half-width-days",
        type=float,
        default=PRIMARY_TRANSIT_HALF_WIDTH_DAYS,
        help="Mask half-width around primary transits.",
    )
    parser.add_argument(
        "--no-eclipse-edge-mask",
        action="store_true",
        help="Keep the secondary-eclipse ingress/egress points (masked by default).",
    )
    parser.add_argument(
        "--eclipse-edge-pad-days",
        type=float,
        default=ECLIPSE_EDGE_PAD_DAYS,
        help="Padding added on each side of the eclipse contact windows before masking.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare and save the real-data observation bundle."""
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run fetch_wasp43b_data.py first.")

    observations, provenance = load_and_prepare(
        args.input,
        member=args.member,
        target_bins=args.target_bins,
        ramp_integrations=args.ramp_integrations,
        error_inflation=args.error_inflation,
        primary_transit_half_width_days=args.primary_transit_half_width_days,
        mask_eclipse_edges=not args.no_eclipse_edge_mask,
        eclipse_edge_pad_days=args.eclipse_edge_pad_days,
    )
    provenance["output_path"] = str(args.output)
    save_observations(args.output, observations)
    write_preparation_provenance(provenance)
    print(f"[wrote {args.output}]")
    print(
        "[prepared "
        f"{provenance['n_binned']} bins from {provenance['n_retained_unbinned']} retained integrations; "
        f"{provenance['selected_channels']} channels]"
    )


if __name__ == "__main__":
    main()
