#!/usr/bin/env python3
"""make_dashboard.py — one consolidated results figure for a finished retrieval.

Reads the .npz outputs (no JAX, no re-run) and assembles a single
``results_dashboard.png`` with the panels that tell the whole story:
  (a) phase-curve fit (data + truth + posterior median + PPC band)
  (b) joint posterior with truth crosshair + correlation (the degeneracy)
  (c) SMC convergence (tempering schedule + ESS)
  (d,e) 1-D marginals with truth + prior range
  (f) terminal brightness-temperature map (truth)

    python make_dashboard.py [OUT_DIR]   # default ./swampe_jaxoplanet_retrieval_outputs
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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("MY_SWAMPE_PLOT_OUT_DIR", str(_RETRIEVAL_ROOT / "data")))
PLOTS_DIR = Path(os.environ.get("MY_SWAMPE_PLOTS_DIR", str(_RETRIEVAL_ROOT / "plots")))

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

# Publication display transform (same convention as plot_smc.py): math-text
# labels + unit scaling applied once to samples/prior bounds/truths.
PARAM_DISPLAY = {
    "tau_rad_hours": (r"$\tau_{\mathrm{rad}}$ [h]", 1.0),
    "tau_drag_hours": (r"$\tau_{\mathrm{drag}}$ [h]", 1.0),
    "planet_fpfs": (r"$F_p/F_s$ [ppm]", 1.0e6),
    "planet_radius_rjup": (r"$R_p$ [$R_{\mathrm{Jup}}$]", 1.0),
    "Phibar": (r"$\bar{\Phi}$ [$10^6\,\mathrm{m^2\,s^{-2}}$]", 1.0e-6),
    "DPhieq": (r"$\Delta\Phi_{\mathrm{eq}}$ [$10^6\,\mathrm{m^2\,s^{-2}}$]", 1.0e-6),
    "noise_inflation": (r"noise inflation $k$", 1.0),
}


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


# Astronomical constants for the eclipse-anchored baseline (same convention as plot_smc.py).
G_SI = 6.6743e-11
MSUN_KG = 1.98892e30
RSUN_M = 6.957e8
RJUP_M = 7.1492e7
DAY_S = 86400.0


def eclipse_anchored_stellar_baseline(t, f, cfg, period):
    """Stellar-flux baseline F_s(t) anchored on full-occultation bottoms (F = F_s
    during eclipse). Pins the F_p/F_s zero point and divides out the linear ramp,
    matching the JWST phase-curve papers (e.g. Kempton et al. 2023). Returns None
    when geometry is missing or no eclipse is covered."""
    try:
        t0 = float(cfg.get("time_transit_days", 0.0))
        m_star = float(cfg["star_mass_msun"]) * MSUN_KG
        r_star = float(cfg["star_radius_rsun"]) * RSUN_M
        r_planet = float(cfg["planet_radius_rjup"]) * RJUP_M
        b = float(cfg["impact_param"])
    except (KeyError, TypeError, ValueError):
        return None
    period_s = period * DAY_S
    a_orb = (G_SI * m_star * period_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
    a_rs = a_orb / r_star
    k = r_planet / r_star
    cos_i = b / a_rs
    sin_i = math.sqrt(max(0.0, 1.0 - cos_i**2))
    arg = (1.0 - k) ** 2 - b**2
    if arg <= 0.0 or sin_i <= 0.0 or a_rs <= 1.0:
        return None
    x = math.sqrt(arg) / (a_rs * sin_i)
    if x >= 1.0:
        return None
    t23_half = 0.5 * (period / math.pi) * math.asin(x)
    n_lo = int(math.floor((float(np.min(t)) - t0) / period - 0.5))
    n_hi = int(math.ceil((float(np.max(t)) - t0) / period - 0.5))
    groups = []
    for n in range(n_lo, n_hi + 1):
        g = np.abs(t - (t0 + (n + 0.5) * period)) < 0.85 * t23_half
        if int(g.sum()) >= 3:
            groups.append(g)
    if not groups:
        return None
    anchor = np.logical_or.reduce(groups)
    if len(groups) >= 2 and int(anchor.sum()) >= 6:
        return np.polyval(np.polyfit(t[anchor], f[anchor], 1), t)
    return np.full_like(np.asarray(t, dtype=float), float(np.median(f[anchor])))


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
    S = np.array(samps["samples"], dtype=float).reshape(-1, len(names))
    truth = np.asarray(cfg.get("inferred_param_truth") or [np.nan] * len(names), float)
    plo = np.asarray(cfg.get("inferred_param_prior_lo"), float)
    phi_ = np.asarray(cfg.get("inferred_param_prior_hi"), float)
    ptypes = [str(x) for x in cfg.get("inferred_param_prior_types", [])]

    for j, name in enumerate(names):
        disp = PARAM_DISPLAY.get(name)
        if disp is None:
            continue
        labels[j] = disp[0]
        scale = disp[1]
        if scale != 1.0:
            S[:, j] *= scale
            if j < plo.size:
                plo[j] *= scale
            if j < phi_.size:
                phi_[j] *= scale
            if j < truth.size:
                truth[j] *= scale

    def is_log(i):
        # Log axis whenever that parameter's own prior is log-uniform (its native
        # sampling space) -- e.g. tau_rad/tau_drag here.
        return i < len(ptypes) and ptypes[i].strip().lower() == "log10_uniform"

    t = np.asarray(obs["times_days"])
    fobs = np.asarray(obs["flux_obs"]); sigma = float(obs["obs_sigma"])
    ftrue = np.asarray(obs["flux_true"]) if "flux_true" in obs.files else np.full_like(fobs, np.nan)
    has_flux_true = bool(np.isfinite(ftrue).any())
    period = float(obs["orbital_period_days"]) if "orbital_period_days" in obs.files else float(
        cfg.get("orbital_period_override_days") or 0.0)
    f_star = eclipse_anchored_stellar_baseline(t, fobs, cfg, period) if period > 0 else None
    display_offset = 0.0 if has_flux_true else float(np.nanmedian(fobs))

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    run_kind = "injection-recovery" if has_flux_true else "real-data pilot"
    fig.suptitle(f"Differentiable MY_SWAMPE $\\rightarrow$ phase-curve retrieval: {run_kind} "
                 f"(N={int(extra['smc_num_particles']) if extra is not None and 'smc_num_particles' in extra.files else '?'} "
                 f"particles, {cfg.get('model_days')}-day spin-up, "
                 f"{sigma*1e6:.0f} ppm noise, float{'64' if cfg.get('use_x64') else '32'})",
                 fontsize=14, fontweight="bold")

    # (a) phase-curve fit — F_p/F_s [ppm] anchored on the eclipse bottoms when
    # the geometry allows (JWST phase-curve convention); otherwise the old
    # median-offset display.
    ax = fig.add_subplot(gs[0, 0])
    if f_star is not None:
        def to_display(f):
            return (np.asarray(f) / f_star - 1.0) * 1e6
        ax.axhline(0.0, lw=0.8, color="0.75", zorder=0)
        ax.set_ylabel(r"planet-to-star flux, $F_p/F_s$ [ppm]")
    else:
        def to_display(f):
            return (np.asarray(f) - display_offset) * 1e6
        ax.set_ylabel("planet flux [ppm]" if has_flux_true else "relative flux $-$ median [ppm]")
    ax.plot(t, to_display(fobs), ".", ms=3, color=COLOR_DATA, alpha=0.5, label="observed")
    if has_flux_true:
        ax.plot(t, to_display(ftrue), "-", lw=2, color=COLOR_TRUTH, label="truth")
    if ppc is not None:
        ax.plot(t, to_display(ppc["p50"]), "-", lw=1.5, color=COLOR_POSTERIOR, label="posterior median")
        ax.fill_between(t, to_display(ppc["p05"]), to_display(ppc["p95"]), alpha=0.3, color=COLOR_BAND, label="90% PPC")
    ax.set_xlabel("time [days]")
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
        ax.set_xlabel("SMC step"); ax.set_ylabel(r"$\beta$", color=COLOR_POSTERIOR); ax.set_ylim(-0.02, 1.05)
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
        # physically non-negative (pipeline floors T at Tmin_K > 0); clip defensively
        T = np.clip(np.asarray(maps[map_key]), 0.0, None)
        im = ax.pcolormesh(lon, lat, T, shading="auto", cmap="inferno", rasterized=True)
        fig.colorbar(im, ax=ax, label="T [K]")
        ax.plot(0.0, 0.0, marker="+", ms=10, mew=1.5, color="w")
        ax.set_xticks(np.arange(-180.0, 181.0, 60.0))
        ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
        ax.set_title(f"(f) terminal brightness-T map ({map_label})")
    else:
        ax.text(0.5, 0.5, "maps unavailable\n(re-run the maps stage of run_smc.py)",
                ha="center", va="center", fontsize=11, color="0.4", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title("(f) terminal brightness-T map")

    path = PLOTS_DIR / "results_dashboard.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"[wrote {path}]")


if __name__ == "__main__":
    main()
