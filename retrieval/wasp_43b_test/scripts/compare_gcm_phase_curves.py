#!/usr/bin/env python3
"""Compare the prepared WASP-43b observations and the SWAMP forward model
against the GCM phase-curve predictions archived with Bell et al. 2024
(Zenodo 10.5281/zenodo.10525170, 3_Models/GCMs/gcm_phase_curves.nc; 31
white-light 5-10.5 um curves from 5 GCM groups, in ppm vs orbital phase).

Outputs
-------
outputs/gcm_comparison.png   overlay figure
stdout                       per-curve peak lead / amplitude / day / night table

The observed planet flux is estimated as (binned relative flux) - (in-eclipse
level), which removes the star + baseline to first order. SWAMP curves are the
pipeline forward model at a few (tau_rad, tau_drag) values with the pilot
config's fpfs placeholder; they illustrate the model family, not a fit.

Run inside the MY_SWAMP conda env (needs jax + jaxoplanet for the model
overlay; pass --no-model to skip it and only use numpy/h5py).
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

SUITE_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = SUITE_ROOT / "data" / "raw" / "WASP43b_MIRI_Data.zip"
GCM_MEMBER = "WASP43b_MIRI_Data/3_Models/GCMs/gcm_phase_curves.nc"
OBS_PATH = SUITE_ROOT / "outputs" / "observations.npz"
OUT_PNG = SUITE_ROOT / "outputs" / "gcm_comparison.png"

PERIOD_DAYS = 0.813474037
ECLIPSE_PHASE_DEG = 180.0

# representative subset for the figure (one per GCM group, clear + cloudy)
FIGURE_LABELS = [
    "Mendonca_2018_W43b_1xsolar_Cloudfree_WLC",
    "Mendonca_2018_W43b_1xsolar_Greycld_WLC",
    "carone_cf_phase_wlc_1xsolar",
    "RMGCM_W43b_WLC_DATA_WASP-43b-clear",
    "Teinturier_cloudless_MIRI_LRS",
    "Teinturier_silicate_1um_MIRI_LRS",
    "Tan_cf_wlc",
    "Tan_MnS+Na2S+MgSiO3_2micron_wlc",
]


def load_gcm_curves():
    """Return (labels, phase_deg, Fp_ppm) from the archived netCDF (HDF5) file."""
    import h5py

    with ZipFile(ARCHIVE) as archive:
        raw = archive.read(GCM_MEMBER)
    with h5py.File(io.BytesIO(raw), "r") as handle:
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["gcm_labels"][:]]
        phase = np.asarray(handle["orbital_phase_deg"], dtype=np.float64)
        fp = np.asarray(handle["Fp"], dtype=np.float64)
    return labels, phase, fp


def curve_stats(phase_deg: np.ndarray, fp_ppm: np.ndarray):
    """Peak lead (deg before eclipse at 180), amplitude, day/night flux."""
    ph = np.mod(phase_deg, 360.0)
    order = np.argsort(ph)
    ph, fp = ph[order], fp_ppm[order]
    good = np.isfinite(fp)
    ph, fp = ph[good], fp[good]
    i_pk = int(np.nanargmax(fp))
    peak_lead = ECLIPSE_PHASE_DEG - ph[i_pk]
    return dict(
        peak_lead_deg=float(peak_lead),
        amplitude_ppm=float(np.nanmax(fp) - np.nanmin(fp)),
        day_ppm=float(np.nanmax(fp)),
        night_ppm=float(np.nanmin(fp)),
    )


def observed_planet_ppm():
    """Approximate observed planet flux: binned relative flux minus the
    in-eclipse level (in eclipse the planet is hidden, so the level is the
    star + instrument baseline)."""
    obs = np.load(OBS_PATH)
    t = np.asarray(obs["times_days"], dtype=np.float64)
    f = np.asarray(obs["flux_obs"], dtype=np.float64)
    phase_deg = np.mod(t / PERIOD_DAYS * 360.0, 360.0)
    # fully-in-eclipse window: T23/2 ~ 0.33 h = 0.014 d -> +/- 6 deg around 180
    in_ecl = np.abs(phase_deg - ECLIPSE_PHASE_DEG) < 6.0
    if in_ecl.sum() < 3:
        raise RuntimeError("Too few in-eclipse points to set the stellar level.")
    level = float(np.mean(f[in_ecl]))
    return phase_deg, (f - level) * 1.0e6, int(in_ecl.sum())


def swamp_model_curves(tau_pairs):
    """Pipeline forward curves (planet flux, ppm) at the pilot config, for a
    few (tau_rad_h, tau_drag_h) values. Imports jax lazily."""
    os.environ.setdefault("SWAMP_RETRIEVAL_PRESET", "gpu")
    os.environ.setdefault("SWAMP_RETRIEVAL_USE_X64", "1")
    os.environ.setdefault(
        "SWAMP_RETRIEVAL_OVERRIDES_FILE", str(SUITE_ROOT / "config" / "wasp43b_pilot_gpu.json")
    )
    scripts_dir = SUITE_ROOT.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import jax.numpy as jnp  # noqa: F401
    import run_smc as R
    import pipeline as P

    cfg = R.preload_real_observation_times(R.make_config())
    P.validate_config(cfg)
    pipe = P.build_pipeline(cfg)
    t = np.asarray(pipe.times_days)
    phase_deg = np.mod(t / PERIOD_DAYS * 360.0, 360.0)
    out = []
    for tr, td in tau_pairs:
        theta = np.asarray(pipe.theta_truth, dtype=np.float64).copy()
        theta[pipe.param_names.index("tau_rad_hours")] = tr
        theta[pipe.param_names.index("tau_drag_hours")] = td
        flux = np.asarray(pipe.phase_curve_model_jit(jnp.asarray(theta, pipe.dtype)))
        out.append((tr, td, phase_deg, flux * 1.0e6))
        s = curve_stats(phase_deg, flux * 1.0e6)
        print(f"  SWAMP tau_rad={tr:5.1f}h tau_drag={td:5.1f}h: "
              f"peak lead {s['peak_lead_deg']:+6.1f} deg, day {s['day_ppm']:6.0f} ppm "
              f"(fpfs placeholder {float(cfg.planet_fpfs):.4f}, scale is fitted in the retrieval)",
              flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-model", action="store_true", help="skip the SWAMP forward overlay")
    args = parser.parse_args()

    labels, gphase, gfp = load_gcm_curves()
    print(f"[{len(labels)} GCM curves loaded]")
    print(f"{'GCM':45s} {'peak lead':>10s} {'amp':>7s} {'day':>7s} {'night':>7s}")
    for i, lab in enumerate(labels):
        s = curve_stats(gphase, gfp[i])
        print(f"{lab:45s} {s['peak_lead_deg']:+9.1f}d {s['amplitude_ppm']:6.0f} "
              f"{s['day_ppm']:6.0f} {s['night_ppm']:6.0f}")

    obs_phase, obs_ppm, n_ecl = observed_planet_ppm()
    print(f"[observed planet flux: eclipse level from {n_ecl} in-eclipse bins; "
          f"max {np.nanmax(obs_ppm):.0f} ppm]")

    model_curves = []
    if not args.no_model:
        print("[running SWAMP forward overlays]")
        model_curves = swamp_model_curves([(3.0, 3.0), (10.0, 6.0), (30.0, 30.0)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = SUITE_ROOT.parent / "scripts" / "science.mplstyle"
    if style.exists():
        plt.style.use(str(style))

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(obs_phase, obs_ppm, ".", ms=4, color="0.25", alpha=0.6, zorder=3,
            label="prepared MIRI data (- eclipse level)")
    for i, lab in enumerate(labels):
        if lab not in FIGURE_LABELS:
            continue
        ph = np.mod(gphase, 360.0)
        order = np.argsort(ph)
        ax.plot(ph[order], gfp[i][order], lw=1.2, alpha=0.8, label=lab.replace("_", " ")[:38])
    for tr, td, ph, fp in model_curves:
        order = np.argsort(ph)
        ax.plot(ph[order], fp[order], lw=2.0, ls="--",
                label=f"SWAMP tau_rad={tr:g}h tau_drag={td:g}h")
    ax.axvline(ECLIPSE_PHASE_DEG, color="0.6", ls=":", lw=1)
    ax.text(ECLIPSE_PHASE_DEG, ax.get_ylim()[1], " eclipse", va="top", fontsize=8, color="0.4")
    ax.set_xlabel("orbital phase [deg, transit = 0]")
    ax.set_ylabel("planet flux [ppm]")
    ax.set_title("WASP-43b 5-10.5 um: data vs Bell et al. 2024 GCMs vs SWAMP forward model")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    print(f"[wrote {OUT_PNG}]")


if __name__ == "__main__":
    main()
