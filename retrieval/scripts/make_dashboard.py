#!/usr/bin/env python3
"""make_dashboard.py — one consolidated results figure for a finished retrieval.

Reads the .npz outputs (no JAX, no re-run) and assembles a single
``results_dashboard.png`` with the panels that tell the whole story:
  (a) phase-curve fit (data + truth + posterior median + PPC band)
  (b) joint posterior with truth crosshair + correlation (the degeneracy)
  (c) SMC convergence (tempering schedule + ESS)
  (d,e) 1-D marginals with truth + prior range
  (f) terminal brightness-temperature map (truth)

    python make_dashboard.py [OUT_DIR]   # default ./swamp_jaxoplanet_retrieval_outputs
"""
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
try:
    import corner as corner_lib
except Exception:
    corner_lib = None
try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None

_SCRIPTS_DIR = Path(__file__).resolve().parent
_RETRIEVAL_ROOT = _SCRIPTS_DIR.parent
_STYLE_FILE = _SCRIPTS_DIR / "science.mplstyle"
if _STYLE_FILE.exists():
    plt.style.use(str(_STYLE_FILE))

# data read from retrieval/data/, figure written to retrieval/plots/
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("SWAMP_PLOT_OUT_DIR", str(_RETRIEVAL_ROOT / "data")))
PLOTS_DIR = Path(os.environ.get("SWAMP_PLOTS_DIR", str(_RETRIEVAL_ROOT / "plots")))

# Okabe-Ito colorblind-safe qualitative palette (Okabe & Ito 2002; Wong, Nature
# Methods 2011) -- same role -> color mapping used in plot_smc.py.
COLOR_TRUTH = "#D55E00"       # vermillion
COLOR_POSTERIOR = "#0072B2"  # blue
COLOR_BAND = "#56B4E9"       # sky blue (shaded bands / PPC)
COLOR_DATA = "#000000"       # observed data points
COLOR_ACCENT = "#009E73"     # bluish green (secondary series, e.g. ESS)

POSTERIOR_VISIBLE_MASS = 0.99
POSTERIOR_RANGE_PAD_FRACTION = 0.08
POSTERIOR_HIST_BINS = 64
LOG_AXIS_MIN_VISIBLE_ORDERS = 1.0
CORNER_MIN_BINS = 16
CORNER_MAX_BINS = 32
CORNER_SMOOTH = 1.6


def load(name):
    p = OUT / name
    return np.load(p, allow_pickle=True) if p.exists() else None


def finite_1d(x: np.ndarray) -> np.ndarray:
    """Return finite values from an array as one dimension."""
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x)]


def posterior_visible_range(
    values: np.ndarray,
    *,
    use_log: bool,
    hard_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Return padded limits around the central posterior mass."""
    v = finite_1d(values)
    if use_log:
        v = v[v > 0.0]
    if v.size == 0:
        return hard_bounds if hard_bounds is not None else (-1.0, 1.0)

    tail = 0.5 * (1.0 - POSTERIOR_VISIBLE_MASS)
    work = np.log10(v) if use_log else v
    lo_w, hi_w = np.quantile(work, [tail, 1.0 - tail])
    if not (math.isfinite(float(lo_w)) and math.isfinite(float(hi_w))) or float(lo_w) == float(hi_w):
        lo_w = float(np.min(work))
        hi_w = float(np.max(work))

    span = float(hi_w - lo_w)
    if span <= 0.0:
        span = max(abs(float(lo_w)), 1.0) * 0.1
    lo_w = float(lo_w) - POSTERIOR_RANGE_PAD_FRACTION * span
    hi_w = float(hi_w) + POSTERIOR_RANGE_PAD_FRACTION * span

    lo = 10.0 ** lo_w if use_log else lo_w
    hi = 10.0 ** hi_w if use_log else hi_w
    if hard_bounds is not None:
        lo = max(float(lo), float(hard_bounds[0]))
        hi = min(float(hi), float(hard_bounds[1]))
    if not (math.isfinite(float(lo)) and math.isfinite(float(hi))) or float(lo) >= float(hi):
        lo, hi = float(np.min(v)), float(np.max(v))
        if lo >= hi:
            delta = max(abs(lo), 1.0) * 0.1
            lo -= delta
            hi += delta
    return float(lo), float(hi)


def orders_of_magnitude_span(lo: float, hi: float) -> float:
    """Return the log10 span for positive bounds."""
    if lo <= 0.0 or hi <= 0.0:
        return 0.0
    return float(np.log10(hi) - np.log10(lo))


def display_log_axis(bounds: Tuple[float, float]) -> bool:
    """Use log tick labels only when the visible range spans at least a decade."""
    lo, hi = bounds
    return orders_of_magnitude_span(float(lo), float(hi)) >= LOG_AXIS_MIN_VISIBLE_ORDERS


def adaptive_corner_bins(values: np.ndarray, bounds: Tuple[float, float], *, use_log: bool) -> int:
    """Choose a stable corner-plot bin count from visible samples."""
    lo, hi = bounds
    v = finite_1d(values)
    if use_log:
        v = v[v > 0.0]
        lo, hi = np.log10([lo, hi])
        v = np.log10(v)

    v = v[(v >= lo) & (v <= hi)]
    if v.size < 2:
        return CORNER_MIN_BINS

    q25, q75 = np.quantile(v, [0.25, 0.75])
    iqr = float(q75 - q25)
    span = float(hi - lo)
    if iqr <= 0.0 or span <= 0.0:
        raw_bins = int(np.ceil(np.sqrt(v.size)))
    else:
        width = 2.0 * iqr / np.cbrt(v.size)
        raw_bins = int(np.ceil(span / width)) if width > 0.0 else int(np.ceil(np.sqrt(v.size)))

    return int(np.clip(raw_bins, CORNER_MIN_BINS, CORNER_MAX_BINS))


def main():
    cfg = json.loads((OUT / "config.json").read_text())
    obs = load("observations.npz")
    samps = load("posterior_samples.npz")
    extra = load("mcmc_extra_fields.npz")
    ppc = load("posterior_predictive_quantiles.npz")
    maps = load("maps_truth_and_posterior_summary.npz")

    names = [str(x) for x in samps["param_names"].tolist()]
    labels = [str(x) for x in samps["param_labels"].tolist()] if "param_labels" in samps.files else names
    S = np.asarray(samps["samples"]).reshape(-1, len(names))
    truth = np.asarray(cfg.get("inferred_param_truth") or [np.nan] * len(names), float)
    plo = np.asarray(cfg.get("inferred_param_prior_lo"), float)
    phi_ = np.asarray(cfg.get("inferred_param_prior_hi"), float)
    ptypes = [str(x) for x in cfg.get("inferred_param_prior_types", [])]

    def is_log(i):
        # Log axis whenever that parameter's own prior is log-uniform (its native
        # sampling space) -- e.g. tau_rad/tau_drag here.
        return i < len(ptypes) and ptypes[i].strip().lower() == "log10_uniform"

    t = np.asarray(obs["times_days"])
    fobs = np.asarray(obs["flux_obs"]); sigma = float(obs["obs_sigma"])
    ftrue = np.asarray(obs["flux_true"]) if "flux_true" in obs.files else np.full_like(fobs, np.nan)
    has_flux_true = bool(np.isfinite(ftrue).any())
    display_offset = 0.0 if has_flux_true else float(np.nanmedian(fobs))

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    run_kind = "injection-recovery" if has_flux_true else "real-data pilot"
    fig.suptitle(f"Differentiable SWAMP -> phase-curve retrieval: {run_kind} "
                 f"(N={int(extra['smc_num_particles']) if extra is not None and 'smc_num_particles' in extra.files else '?'} "
                 f"particles, {cfg.get('model_days')}-day spin-up, "
                 f"{sigma*1e6:.0f} ppm noise, float{'64' if cfg.get('use_x64') else '32'})",
                 fontsize=14, fontweight="bold")

    # (a) phase-curve fit
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, (fobs - display_offset) * 1e6, ".", ms=3, color=COLOR_DATA, alpha=0.5, label="observed")
    if has_flux_true:
        ax.plot(t, ftrue * 1e6, "-", lw=2, color=COLOR_TRUTH, label="truth")
    if ppc is not None:
        ax.plot(t, (np.asarray(ppc["p50"]) - display_offset) * 1e6, "-", lw=1.5, color=COLOR_POSTERIOR, label="posterior median")
        ax.fill_between(t, (np.asarray(ppc["p05"]) - display_offset) * 1e6, (np.asarray(ppc["p95"]) - display_offset) * 1e6, alpha=0.3, color=COLOR_BAND, label="90% PPC")
    ax.set_xlabel("time [days]")
    ax.set_ylabel("planet flux [ppm]" if has_flux_true else "relative flux - median [ppm]")
    ax.set_title("(a) thermal phase-curve fit"); ax.legend(fontsize=8)

    # (b) joint posterior: KDE density contours + scatter + truth + correlation.
    # Log-log if the parameters' own priors are log-uniform (their native sampling space).
    ax = fig.add_subplot(gs[0, 1])
    if len(names) >= 2:
        x, y = S[:, 0], S[:, 1]
        log_x, log_y = is_log(0), is_log(1)
        xlim = posterior_visible_range(x, use_log=log_x, hard_bounds=(float(plo[0]), float(phi_[0])))
        ylim = posterior_visible_range(y, use_log=log_y, hard_bounds=(float(plo[1]), float(phi_[1])))
        display_log_x = display_log_axis(xlim)
        display_log_y = display_log_axis(ylim)
        bins = [
            adaptive_corner_bins(x, xlim, use_log=display_log_x),
            adaptive_corner_bins(y, ylim, use_log=display_log_y),
        ]
        if corner_lib is not None:
            corner_lib.hist2d(
                x,
                y,
                bins=bins,
                range=[xlim, ylim],
                axes_scale=["log" if display_log_x else "linear", "log" if display_log_y else "linear"],
                color=COLOR_POSTERIOR,
                ax=ax,
                quiet=True,
                smooth=CORNER_SMOOTH,
                plot_datapoints=False,
                plot_density=True,
                plot_contours=True,
                fill_contours=True,
                levels=(0.393, 0.675, 0.864, 0.955),
                contour_kwargs={"linewidths": 1.2},
            )
        elif gaussian_kde is not None and len(x) > 5 and np.std(x) > 0 and np.std(y) > 0:
            try:
                xk = np.log10(x) if display_log_x else x
                yk = np.log10(y) if display_log_y else y
                xlim_k = np.log10(xlim) if display_log_x else xlim
                ylim_k = np.log10(ylim) if display_log_y else ylim
                xgk = np.linspace(xlim_k[0], xlim_k[1], 80); ygk = np.linspace(ylim_k[0], ylim_k[1], 80)
                XXk, YYk = np.meshgrid(xgk, ygk)
                ZZ = gaussian_kde(np.vstack([xk, yk]))(np.vstack([XXk.ravel(), YYk.ravel()])).reshape(XXk.shape)
                XX = 10.0 ** XXk if display_log_x else XXk
                YY = 10.0 ** YYk if display_log_y else YYk
                ax.contourf(XX, YY, ZZ, levels=8, cmap="Blues", alpha=0.85)
            except Exception:
                pass
            ax.scatter(x, y, s=6, alpha=0.35, color="0.25")
        else:
            ax.scatter(x, y, s=6, alpha=0.35, color="0.25")
        if truth.size >= 2 and np.isfinite(truth[:2]).all():
            ax.axvline(truth[0], color=COLOR_TRUTH, ls="--", lw=1); ax.axhline(truth[1], color=COLOR_TRUTH, ls="--", lw=1)
            ax.plot(truth[0], truth[1], "*", ms=16, color=COLOR_TRUTH, label="truth")
        if display_log_x:
            ax.set_xscale("log")
        if display_log_y:
            ax.set_yscale("log")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.set_title(f"(b) joint posterior  (corr = {r:+.2f})")
        ax.legend(fontsize=8)
    else:
        ax.hist(S[:, 0], bins=30); ax.set_title("(b) posterior")

    # (c) SMC convergence
    ax = fig.add_subplot(gs[0, 2])
    if extra is not None and "smc_betas" in extra.files:
        betas = np.asarray(extra["smc_betas"]).reshape(-1)
        steps = np.arange(len(betas))
        ax.plot(steps, betas, "o-", color=COLOR_POSTERIOR, label="beta (temperature)")
        ax.set_xlabel("SMC step"); ax.set_ylabel("beta", color=COLOR_POSTERIOR); ax.set_ylim(-0.02, 1.05)
        ax.set_title("(c) SMC convergence")
        if "smc_ess" in extra.files:
            ess = np.asarray(extra["smc_ess"]).reshape(-1)
            N = int(extra["smc_num_particles"]) if "smc_num_particles" in extra.files else ess.max()
            ax2 = ax.twinx()
            ax2.plot(steps[1:], ess / N, "s-", color=COLOR_ACCENT, label="ESS / N")
            ax2.set_ylabel("ESS fraction", color=COLOR_ACCENT); ax2.set_ylim(0, 1.05)

    # (d,e) 1-D marginals: KDE curve (smooth even for a small swarm) + light hist.
    # Log axis (and log-space KDE via the log-Jacobian) if that parameter's prior is log-uniform.
    for i in range(min(2, len(names))):
        ax = fig.add_subplot(gs[1, i])
        v = S[:, i]
        natural_log = is_log(i)
        xlim = posterior_visible_range(v, use_log=natural_log, hard_bounds=(float(plo[i]), float(phi_[i])))
        log_v = display_log_axis(xlim)
        bins = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), POSTERIOR_HIST_BINS) if (log_v and xlim[0] > 0) else POSTERIOR_HIST_BINS
        ax.hist(v, bins=bins, density=True, alpha=0.30, color=COLOR_POSTERIOR)
        if gaussian_kde is not None and len(v) > 5 and np.std(v) > 0:
            try:
                if log_v:
                    v_kde = v[v > 0.0]
                    lv = np.log10(v_kde)
                    xs = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 300)
                    ys = gaussian_kde(lv)(np.log10(xs)) / (xs * np.log(10.0))
                else:
                    xs = np.linspace(xlim[0], xlim[1], 300)
                    ys = gaussian_kde(v)(xs)
                ax.plot(xs, ys, color=COLOR_POSTERIOR, lw=2)
            except Exception:
                pass
        if truth.size > i and np.isfinite(truth[i]):
            ax.axvline(truth[i], color=COLOR_TRUTH, lw=2, label=f"truth = {truth[i]:.1f}")
        q = np.percentile(v, [5, 50, 95])
        ax.axvspan(q[0], q[2], alpha=0.12, color=COLOR_POSTERIOR)
        ax.axvline(q[1], color=COLOR_POSTERIOR, ls="--", lw=1.5, label=f"median = {q[1]:.2f}")
        prior_label = False
        if xlim[0] <= plo[i] <= xlim[1]:
            ax.axvline(plo[i], color="0.6", ls=":", lw=1, label="prior range")
            prior_label = True
        if xlim[0] <= phi_[i] <= xlim[1]:
            ax.axvline(phi_[i], color="0.6", ls=":", lw=1, label=None if prior_label else "prior range")
        if log_v:
            ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_xlabel(labels[i]); ax.set_title(f"({'d' if i==0 else 'e'}) {labels[i]} posterior")
        ax.legend(fontsize=8); ax.set_yticks([])

    # (f) terminal brightness-temperature map (truth if injected, else posterior median)
    ax = fig.add_subplot(gs[1, 2])
    map_key, map_label = None, None
    if maps is not None:
        if "T_truth" in maps.files and np.isfinite(np.asarray(maps["T_truth"])).any():
            map_key, map_label = "T_truth", "truth"
        elif "T_post" in maps.files and np.isfinite(np.asarray(maps["T_post"])).any():
            map_key, map_label = "T_post", "posterior median"
    if map_key is not None:
        lon = np.degrees(np.asarray(maps["lon"])); lat = np.degrees(np.asarray(maps["lat"]))
        T = np.asarray(maps[map_key])
        im = ax.pcolormesh(lon, lat, T, shading="auto", cmap="inferno")
        fig.colorbar(im, ax=ax, label="T [K]")
        ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
        ax.set_title(f"(f) terminal brightness-T map ({map_label})")
    else:
        ax.set_title("(f) maps unavailable")

    path = PLOTS_DIR / "results_dashboard.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"[wrote {path}]")


if __name__ == "__main__":
    main()
