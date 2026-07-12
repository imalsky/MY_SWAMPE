#!/usr/bin/env python3
"""
plot_smc.py

Plot all results from a completed `run_smc.py` run.

This script NEVER runs MY_SWAMPE and NEVER runs inference. It only reads saved outputs from
OUT_DIR and creates plots under OUT_DIR/plots.

This version adds much more defensive validation + verbose, terminal-friendly logging to help
diagnose common failure modes:
- missing / mismatched files
- wrong array shapes or missing NPZ keys
- non-finite data (NaN/Inf) poisoning plots
- posterior samples with unexpected ranges / degeneracies
- SMC diagnostics indicating weight collapse or stalled tempering
- optional file load failures (PPC, maps, diagnostics)

No CLI args by design: edit OUT_DIR below if needed (or override via MY_SWAMPE_PLOT_OUT_DIR).
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Force a non-interactive backend for headless / HPC environments.
import matplotlib

matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt

try:
    import corner as corner_lib
except Exception:
    corner_lib = None

try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None

# Okabe-Ito colorblind-safe qualitative palette (Okabe & Ito 2002; Wong, Nature
# Methods 2011) -- consistent role -> color mapping used across every retrieval plot.
COLOR_TRUTH = "#D55E00"       # vermillion
COLOR_POSTERIOR = "#0072B2"  # blue
COLOR_BAND = "#56B4E9"       # sky blue (shaded bands / PPC)
COLOR_DATA = "#000000"       # observed data points
COLOR_ACCENT = "#009E73"     # bluish green (secondary series, e.g. ESS)

# Publication display transform: math-text labels + unit scaling per parameter.
# The scale is applied ONCE to samples/prior bounds/truths right after loading,
# so every downstream panel agrees and no figure sprouts a colliding x10^6
# offset label. Parameters not listed keep their stored label, unscaled.
PARAM_DISPLAY: Dict[str, Tuple[str, float]] = {
    "tau_rad_hours": (r"$\tau_{\mathrm{rad}}$ [h]", 1.0),
    "tau_drag_hours": (r"$\tau_{\mathrm{drag}}$ [h]", 1.0),
    "planet_fpfs": (r"$F_p/F_s$ [ppm]", 1.0e6),
    "planet_radius_rjup": (r"$R_p$ [$R_{\mathrm{Jup}}$]", 1.0),
    "Phibar": (r"$\bar{\Phi}$ [$10^6\,\mathrm{m^2\,s^{-2}}$]", 1.0e-6),
    "DPhieq": (r"$\Delta\Phi_{\mathrm{eq}}$ [$10^6\,\mathrm{m^2\,s^{-2}}$]", 1.0e-6),
    "noise_inflation": (r"noise inflation $k$", 1.0),
}

# Publication style guide (the project's science.mplstyle, shipped alongside this
# script). Applied to every figure so retrieval plots match the paper figures.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_STYLE_FILE = _SCRIPTS_DIR / "science.mplstyle"
if _STYLE_FILE.exists():
    plt.style.use(str(_STYLE_FILE))


# =============================================================================
# CONFIG
# =============================================================================

# Layout: this script lives in retrieval/scripts/; data is read from retrieval/data/
# and figures are written to retrieval/plots/. Override with env vars if needed:
#   MY_SWAMPE_PLOT_OUT_DIR=/path/to/data MY_SWAMPE_PLOTS_DIR=/path/to/plots ./plot_smc.py
_RETRIEVAL_ROOT = _SCRIPTS_DIR.parent
OUT_DIR = Path(os.environ.get("MY_SWAMPE_PLOT_OUT_DIR", str(_RETRIEVAL_ROOT / "data")))
PLOTS_DIR = Path(os.environ.get("MY_SWAMPE_PLOTS_DIR", str(_RETRIEVAL_ROOT / "plots")))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging verbosity can be overridden without editing the file:
#   MY_SWAMPE_PLOT_LOG_LEVEL=DEBUG ./plot_smc.py
_LOG_LEVEL_NAME = os.environ.get("MY_SWAMPE_PLOT_LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)

POSTERIOR_VISIBLE_MASS = 0.99
POSTERIOR_RANGE_PAD_FRACTION = 0.08
POSTERIOR_HIST_BINS = 64
LOG_AXIS_MIN_VISIBLE_ORDERS = 1.0
CORNER_MIN_BINS = 16
CORNER_MAX_BINS = 32
CORNER_HIST_BIN_FACTOR = 2
CORNER_SMOOTH = 1.6


# =============================================================================
# LOGGING
# =============================================================================

log_path = OUT_DIR / "plot.log"
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    force=True,
)
logger = logging.getLogger("swampe_plot")


# =============================================================================
# Diagnostics helpers
# =============================================================================


def _utc_ts() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _describe_path(p: Path) -> str:
    """Summarize a filesystem path for logging."""
    try:
        st = p.stat()
    except FileNotFoundError:
        return "(missing)"
    size_kb = st.st_size / 1024.0
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"size={size_kb:.1f} KiB, mtime={mtime}"


def _tail_text_lines(path: Path, *, n_lines: int = 30, max_bytes: int = 64_000) -> List[str]:
    """Read the trailing lines from a text file."""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end <= 0:
                return []
            read_size = min(int(max_bytes), int(end))
            f.seek(end - read_size, os.SEEK_SET)
            chunk = f.read(read_size)
    except Exception as e:
        logger.debug(f"Could not read tail of {path}: {e}", exc_info=True)
        return []

    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception:
        text = chunk.decode(errors="replace")

    lines = text.splitlines()
    return lines[-int(n_lines) :]


def log_environment() -> None:
    """Log environment."""
    logger.info(f"=== plot_smc diagnostics start ({_utc_ts()}) ===")
    logger.info(f"OUT_DIR={OUT_DIR.resolve()}")
    logger.info(f"PLOTS_DIR={PLOTS_DIR.resolve()}")
    logger.info(f"CWD={Path.cwd().resolve()}")
    logger.info(f"Python={sys.version.splitlines()[0]}")
    logger.info(f"Platform={platform.platform()}")
    logger.info(f"NumPy={np.__version__}")
    logger.info(f"Matplotlib={matplotlib.__version__}, backend={matplotlib.get_backend()}")
    logger.info(f"Log level={_LOG_LEVEL_NAME}")
    if OUT_DIR.exists():
        try:
            files = sorted([p.name for p in OUT_DIR.iterdir()])
            logger.info(f"OUT_DIR contains {len(files)} entries: {files}")
        except Exception as e:
            logger.warning(f"Could not list OUT_DIR entries: {e}")

        run_log = OUT_DIR / "run.log"
        if run_log.exists():
            logger.info(f"Found run.log: {_describe_path(run_log)}")
            tail = _tail_text_lines(run_log, n_lines=25)
            if tail:
                logger.info("run.log tail (last 25 lines):")
                for line in tail:
                    logger.info(f"run.log| {line}")
            else:
                logger.info("run.log tail: (empty or unreadable)")
        else:
            logger.info("run.log not found in OUT_DIR.")
    else:
        logger.error(
            "OUT_DIR does not exist. This plot script only reads outputs; run the inference script first, "
            "or set MY_SWAMPE_PLOT_OUT_DIR to the directory that contains config.json/observations.npz."
        )


def _finite_mask(x: np.ndarray) -> np.ndarray:
    """Compute a finite-value mask for an array."""
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.number):
        return np.ones(x.shape, dtype=bool)
    return np.isfinite(x)


def log_array_stats(name: str, x: Any, *, max_quantile_elems: int = 2_000_000) -> None:
    """Log shape/dtype and basic finite/min/max stats.
    
    For very large arrays, we may sub-sample for quantiles to avoid excessive cost.
    """
    try:
        arr = np.asarray(x)
    except Exception as e:
        logger.warning(f"{name}: could not convert to ndarray for stats ({e})")
        return

    logger.info(f"{name}: dtype={arr.dtype}, shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        return

    flat = arr.reshape(-1)
    finite = np.isfinite(flat)
    n = flat.size
    n_fin = int(finite.sum())
    n_bad = n - n_fin
    if n == 0:
        logger.warning(f"{name}: empty array")
        return

    logger.info(f"{name}: finite={n_fin}/{n} ({100.0 * n_fin / max(n,1):.2f}%), nonfinite={n_bad}")
    if n_fin == 0:
        return

    v = flat[finite]
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    mean = float(np.mean(v))
    std = float(np.std(v))
    logger.info(f"{name}: min={vmin:.6g}, max={vmax:.6g}, mean={mean:.6g}, std={std:.6g}")

    # Quantiles: potentially sub-sample if huge.
    if v.size > max_quantile_elems:
        rng = np.random.default_rng(0)
        idx = rng.choice(v.size, size=max_quantile_elems, replace=False)
        vq = v[idx]
        logger.debug(f"{name}: quantiles computed on a random subsample of {max_quantile_elems} values")
    else:
        vq = v

    try:
        q01, q05, q50, q95, q99 = np.quantile(vq, [0.01, 0.05, 0.5, 0.95, 0.99])
        logger.info(f"{name}: q01={q01:.6g}, q05={q05:.6g}, q50={q50:.6g}, q95={q95:.6g}, q99={q99:.6g}")
    except Exception as e:
        logger.debug(f"{name}: could not compute quantiles ({e})")


def _require_file(path: Path, *, hint: str) -> None:
    """Raise an error if a required file is missing."""
    if not path.exists():
        msg = f"Missing required file: {path} ({hint})"
        logger.error(msg)
        raise FileNotFoundError(msg)
    logger.info(f"Found {path.name}: {_describe_path(path)}")


def load_json_required(path: Path) -> Dict[str, Any]:
    """Load JSON required."""
    _require_file(path, hint="run_smc.py should write this")
    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON at {path}: {e}") from e
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object (dict) in {path}, got {type(obj)}")
    return obj


def load_npz_required(path: Path, *, required_keys: Sequence[str], allow_pickle: bool = False) -> np.lib.npyio.NpzFile:
    """Load `.npz` required."""
    _require_file(path, hint="run_smc.py should write this")
    try:
        npz = np.load(path, allow_pickle=allow_pickle)
    except Exception as e:
        raise RuntimeError(f"Failed to load NPZ at {path}: {e}") from e

    keys = list(npz.files)
    logger.info(f"Loaded {path.name}: keys={keys}")
    missing = [k for k in required_keys if k not in keys]
    if missing:
        npz.close()
        raise KeyError(f"{path.name} missing keys {missing}. Available keys={keys}")
    return npz


def load_npz_optional(path: Path, *, allow_pickle: bool = False) -> Optional[np.lib.npyio.NpzFile]:
    """Load an `.npz` archive if it exists."""
    if not path.exists():
        logger.info(f"Optional file not present: {path.name}")
        return None
    try:
        npz = np.load(path, allow_pickle=allow_pickle)
    except Exception:
        logger.exception(f"Optional file exists but could not be loaded: {path}")
        return None
    logger.info(f"Loaded optional {path.name}: keys={list(npz.files)}")
    return npz


def validate_1d_same_length(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    """Validate 1d same length."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Length mismatch: {name_a} has {a.shape[0]} elems, {name_b} has {b.shape[0]} elems")


def check_monotonic_increasing(name: str, x: np.ndarray) -> None:
    """Check monotonic increasing."""
    x = np.asarray(x).reshape(-1)
    if x.size < 2:
        return
    if not np.all(np.isfinite(x)):
        logger.warning(f"{name}: contains non-finite values; cannot check monotonicity reliably")
        return
    if np.any(np.diff(x) < 0):
        logger.warning(f"{name}: NOT monotonic increasing (this can indicate corrupted time arrays)")
    else:
        logger.info(f"{name}: monotonic increasing")


# =============================================================================
# LOAD FILES (with aggressive diagnostics)
# =============================================================================

log_environment()

cfg_path = OUT_DIR / "config.json"
cfg: Dict[str, Any] = load_json_required(cfg_path)
logger.info(f"Loaded config.json with {len(cfg)} keys")

cfg_out_dir = cfg.get("out_dir", None)
if cfg_out_dir is not None:
    try:
        cfg_out = Path(str(cfg_out_dir)).expanduser().resolve()
        out_res = OUT_DIR.resolve()
        if cfg_out != out_res:
            logger.warning(
                "config.json out_dir does not match OUT_DIR used by plot_smc.py. "
                f"config out_dir={cfg_out}, plot OUT_DIR={out_res}. "
                "If you changed cfg.out_dir in run_smc.py, set MY_SWAMPE_PLOT_OUT_DIR accordingly."
            )
        else:
            logger.info("config.json out_dir matches plot OUT_DIR.")
    except Exception as e:
        logger.warning(f"Could not interpret config.json out_dir={cfg_out_dir!r} as a path: {e}")

# Apply DPI if present
if "fig_dpi" in cfg:
    try:
        plt.rcParams["figure.dpi"] = int(cfg["fig_dpi"])
        logger.info(f"Matplotlib figure.dpi set to {plt.rcParams['figure.dpi']}")
    except Exception:
        logger.exception("Failed to apply fig_dpi from config.json; continuing with matplotlib default.")

obs_path = OUT_DIR / "observations.npz"
obs = load_npz_required(
    obs_path,
    required_keys=("times_days", "flux_obs", "obs_sigma", "orbital_period_days"),
)
times_days = np.asarray(obs["times_days"])
flux_obs = np.asarray(obs["flux_obs"])
flux_true = (
    np.asarray(obs["flux_true"])
    if "flux_true" in obs.files
    else np.full_like(flux_obs, np.nan, dtype=float)
)
has_flux_true = bool(np.isfinite(flux_true).any())
obs_sigma = float(obs["obs_sigma"])
orbital_period_days = float(obs["orbital_period_days"])
obs_sigma_vec = (
    np.asarray(obs["obs_sigma_vec"], dtype=float) if "obs_sigma_vec" in obs.files
    else np.full_like(flux_obs, obs_sigma)
)
obs.close()

log_array_stats("times_days", times_days)
log_array_stats("flux_true", flux_true)
log_array_stats("flux_obs", flux_obs)
logger.info(f"obs_sigma={obs_sigma:.6g}")
logger.info(f"orbital_period_days={orbital_period_days:.6g}")

if not (math.isfinite(obs_sigma) and obs_sigma > 0.0):
    raise ValueError(f"obs_sigma must be finite and > 0. Got {obs_sigma!r}")
if not (math.isfinite(orbital_period_days) and orbital_period_days > 0.0):
    raise ValueError(f"orbital_period_days must be finite and > 0. Got {orbital_period_days!r}")

if has_flux_true:
    validate_1d_same_length("times_days", times_days, "flux_true", flux_true)
validate_1d_same_length("times_days", times_days, "flux_obs", flux_obs)
check_monotonic_increasing("times_days", times_days)

samples_path = OUT_DIR / "posterior_samples.npz"
samps = load_npz_required(samples_path, required_keys=("param_names", "samples"), allow_pickle=True)
param_names = [str(x) for x in samps["param_names"].tolist()]
param_labels = [str(x) for x in samps["param_labels"].tolist()] if "param_labels" in samps.files else param_names
samples = np.asarray(samps["samples"])  # (chains, draws, dim)
samps.close()

logger.info(f"Inferred parameters from posterior_samples.npz: {param_names}")
if len(param_labels) != len(param_names):
    logger.warning(
        f"param_labels length ({len(param_labels)}) != param_names length ({len(param_names)}); using param_names."
    )
    param_labels = param_names

if samples.ndim != 3:
    raise ValueError(f"posterior_samples['samples'] must have shape (chains, draws, dim); got {samples.shape}")
if samples.shape[-1] != len(param_names):
    logger.warning(
        f"samples dim={samples.shape[-1]} but len(param_names)={len(param_names)}. "
        "This usually indicates a corrupted posterior_samples.npz."
    )
logger.info(f"Loaded posterior samples cube: shape={samples.shape} (chains, draws, dim)")

log_array_stats("samples", samples)

# Optional: SMC diagnostics
extra_path = OUT_DIR / "mcmc_extra_fields.npz"
extra = load_npz_optional(extra_path)

# Optional: posterior predictive quantiles
ppc_quant_path = OUT_DIR / "posterior_predictive_quantiles.npz"
ppc_q: Optional[Dict[str, np.ndarray]] = None
q = load_npz_optional(ppc_quant_path)
if q is not None:
    required = ("p05", "p50", "p95")
    missing = [k for k in required if k not in q.files]
    if missing:
        logger.warning(
            f"posterior_predictive_quantiles.npz missing keys {missing}; expected {required}. Ignoring PPC file."
        )
    else:
        p05 = np.asarray(q["p05"])
        p50 = np.asarray(q["p50"])
        p95 = np.asarray(q["p95"])
        log_array_stats("ppc_p05", p05)
        log_array_stats("ppc_p50", p50)
        log_array_stats("ppc_p95", p95)

        # Shape checks: must match times_days length.
        try:
            validate_1d_same_length("times_days", times_days, "ppc_p50", p50)
            validate_1d_same_length("times_days", times_days, "ppc_p05", p05)
            validate_1d_same_length("times_days", times_days, "ppc_p95", p95)
        except Exception:
            logger.exception("PPC arrays do not match observation times; ignoring PPC file.")
        else:
            # Quantile ordering check
            if np.any(p05 > p50) or np.any(p50 > p95):
                logger.warning("PPC quantiles violate ordering (p05<=p50<=p95) at some times.")
            ppc_q = {"p05": p05, "p50": p50, "p95": p95}
            logger.info("PPC quantiles will be overlaid on phase curve plot.")

    q.close()

# Optional: maps
maps_path = OUT_DIR / "maps_truth_and_posterior_summary.npz"
maps = load_npz_optional(maps_path)
if maps is not None:
    for k in ("lon", "lat", "phi_truth", "T_truth", "I_truth", "phi_post", "T_post", "I_post"):
        if k not in maps.files:
            logger.warning(f"maps file missing key {k!r}; some plots may be skipped.")
    # Log a subset of arrays (avoid spamming huge logs)
    if "phi_truth" in maps.files:
        log_array_stats("maps.phi_truth", maps["phi_truth"])
    if "phi_post" in maps.files:
        log_array_stats("maps.phi_post", maps["phi_post"])
    if "T_truth" in maps.files:
        log_array_stats("maps.T_truth", maps["T_truth"])
    if "T_post" in maps.files:
        log_array_stats("maps.T_post", maps["T_post"])
    if "I_truth" in maps.files:
        log_array_stats("maps.I_truth", maps["I_truth"])
    if "I_post" in maps.files:
        log_array_stats("maps.I_post", maps["I_post"])


# =============================================================================
# Helpers used by plotting
# =============================================================================


def flatten_chain_draw(x: np.ndarray) -> np.ndarray:
    """(chains, draws, ...) -> (chains*draws, ...)"""
    x = np.asarray(x)
    return x.reshape((-1,) + x.shape[2:])


# Astronomical constants for the eclipse-geometry baseline below.
G_SI = 6.6743e-11
MSUN_KG = 1.98892e30
RSUN_M = 6.957e8
RJUP_M = 7.1492e7
DAY_S = 86400.0


def eclipse_anchored_stellar_baseline(t: np.ndarray, f: np.ndarray) -> Optional[np.ndarray]:
    """Stellar-flux baseline F_s(t) anchored on full-occultation (t2-t3) bottoms.

    During secondary eclipse the planet is hidden, so the measured flux IS the
    stellar flux. Fitting a linear trend through the in-eclipse points (both
    eclipses when the data cover two) pins the planet-to-star flux zero point
    and removes the linear ramp — the display convention of the JWST
    phase-curve papers (e.g. Kempton et al. 2023 for GJ 1214b), where the
    y-axis is F_p/F_s in ppm and eclipse bottoms sit at zero.

    Returns F_s evaluated at every t, or None when the system geometry is not
    in the config / the data contain no in-eclipse points (synthetic smoke
    runs) — callers then fall back to the median-offset display.
    """
    try:
        period = float(orbital_period_days)
        t0 = float(cfg.get("time_transit_days", 0.0))
        m_star = float(cfg["star_mass_msun"]) * MSUN_KG
        r_star = float(cfg["star_radius_rsun"]) * RSUN_M
        r_planet = float(cfg["planet_radius_rjup"]) * RJUP_M
        b = float(cfg["impact_param"])
    except (KeyError, TypeError, ValueError):
        return None

    # Full-occultation half-duration from the standard transit-geometry formula
    # (circular orbit; same geometry the data preparation used for edge masking).
    period_s = period * DAY_S
    a_orb = (G_SI * m_star * period_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
    a_rs = a_orb / r_star
    k = r_planet / r_star
    cos_i = b / a_rs
    sin_i = math.sqrt(max(0.0, 1.0 - cos_i**2))
    arg = (1.0 - k) ** 2 - b**2
    if arg <= 0.0 or sin_i <= 0.0 or a_rs <= 1.0:
        return None  # grazing geometry: no flat eclipse bottom
    x = math.sqrt(arg) / (a_rs * sin_i)
    if x >= 1.0:
        return None
    t23_half = 0.5 * (period / math.pi) * math.asin(x)

    # Eclipse centers t0 + (n + 1/2) P covering the data span.
    n_lo = int(math.floor((float(np.min(t)) - t0) / period - 0.5))
    n_hi = int(math.ceil((float(np.max(t)) - t0) / period - 0.5))
    groups: List[np.ndarray] = []
    for n in range(n_lo, n_hi + 1):
        tc = t0 + (n + 0.5) * period
        g = np.abs(t - tc) < 0.85 * t23_half  # stay clear of ingress/egress
        if int(g.sum()) >= 3:
            groups.append(g)
    if not groups:
        return None

    anchor = np.logical_or.reduce(groups)
    n_anchor = int(anchor.sum())
    if len(groups) >= 2 and n_anchor >= 6:
        coeffs = np.polyfit(t[anchor], f[anchor], 1)  # linear ramp through both bottoms
        f_star = np.polyval(coeffs, t)
    else:
        f_star = np.full_like(np.asarray(t, dtype=float), float(np.median(f[anchor])))
    logger.info(
        f"Eclipse-anchored stellar baseline: {len(groups)} eclipse(s), {n_anchor} in-eclipse points, "
        f"t23/2={t23_half:.5f} d, {'linear' if len(groups) >= 2 and n_anchor >= 6 else 'constant'} F_s(t)."
    )
    return f_star


def save_fig(fig: plt.Figure, filename: str) -> None:
    """Save a figure and close it."""
    path = PLOTS_DIR / filename
    # tight_layout can fail for some figures; don't let it kill the whole script.
    try:
        fig.tight_layout()
    except Exception:
        logger.debug(f"tight_layout failed for {filename}; saving without tight_layout.", exc_info=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def finite_1d(x: np.ndarray) -> np.ndarray:
    """Finite 1d."""
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x)]


def orders_of_magnitude_span(lo: float, hi: float) -> float:
    """Orders of magnitude span."""
    if lo <= 0.0 or hi <= 0.0:
        return 0.0
    return float(np.log10(hi) - np.log10(lo))


def should_use_log_axis(
    values: np.ndarray,
    *,
    orders_threshold: float,
    explicit_bounds: Optional[Tuple[float, float]] = None,
) -> bool:
    """Heuristic: use log axis only if range spans many orders and values are positive."""
    if explicit_bounds is not None:
        lo, hi = explicit_bounds
        if lo <= 0.0 or hi <= 0.0:
            return False
        return orders_of_magnitude_span(float(lo), float(hi)) >= float(orders_threshold)

    v = finite_1d(values)
    if v.size == 0:
        return False
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmin <= 0.0:
        return False
    return orders_of_magnitude_span(vmin, vmax) >= float(orders_threshold)


def log_axis_for_param(j: int, v: np.ndarray, bounds: Tuple[float, float]) -> bool:
    """Log axis whenever the parameter's own prior is log-uniform (its native sampling
    space); otherwise fall back to the orders-of-magnitude heuristic."""
    if j < len(prior_types) and str(prior_types[j]).strip().lower() == "log10_uniform":
        return True
    return should_use_log_axis(v, orders_threshold=orders_threshold, explicit_bounds=bounds)


def display_log_axis(bounds: Tuple[float, float]) -> bool:
    """Use log tick labels only when the visible range spans at least a decade."""
    lo, hi = bounds
    if lo <= 0.0 or hi <= 0.0:
        return False
    return orders_of_magnitude_span(float(lo), float(hi)) >= LOG_AXIS_MIN_VISIBLE_ORDERS


def quantile_summary(v: np.ndarray) -> Tuple[float, float, float]:
    """Compute quantile summary."""
    v = finite_1d(v)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    q16, q50, q84 = np.quantile(v, [0.16, 0.50, 0.84])
    return float(q16), float(q50), float(q84)


def posterior_visible_range(
    values: np.ndarray,
    *,
    use_log: bool,
    hard_bounds: Optional[Tuple[float, float]] = None,
    visible_mass: float = POSTERIOR_VISIBLE_MASS,
    pad_fraction: float = POSTERIOR_RANGE_PAD_FRACTION,
) -> Tuple[float, float]:
    """Return a padded plotting range around the central posterior mass."""
    v = finite_1d(values)
    if use_log:
        v = v[v > 0.0]

    if v.size == 0:
        if hard_bounds is not None:
            return hard_bounds
        return (-1.0, 1.0)

    tail = 0.5 * (1.0 - float(visible_mass))
    qlo = max(0.0, tail)
    qhi = min(1.0, 1.0 - tail)
    work = np.log10(v) if use_log else v

    lo_w, hi_w = np.quantile(work, [qlo, qhi])
    if not (math.isfinite(float(lo_w)) and math.isfinite(float(hi_w))) or float(lo_w) == float(hi_w):
        lo_w = float(np.min(work))
        hi_w = float(np.max(work))

    span = float(hi_w - lo_w)
    if span <= 0.0:
        center = float(lo_w)
        span = max(abs(center), 1.0) * 0.1
        lo_w = center - span
        hi_w = center + span
    else:
        lo_w = float(lo_w) - float(pad_fraction) * span
        hi_w = float(hi_w) + float(pad_fraction) * span

    lo = 10.0 ** lo_w if use_log else float(lo_w)
    hi = 10.0 ** hi_w if use_log else float(hi_w)

    if hard_bounds is not None:
        bound_lo, bound_hi = hard_bounds
        if math.isfinite(bound_lo):
            lo = max(lo, float(bound_lo))
        if math.isfinite(bound_hi):
            hi = min(hi, float(bound_hi))

    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hard_bounds is not None:
            bound_lo, bound_hi = hard_bounds
            lo = max(lo, float(bound_lo)) if math.isfinite(bound_lo) else lo
            hi = min(hi, float(bound_hi)) if math.isfinite(bound_hi) else hi
        if lo >= hi:
            delta = max(abs(lo), 1.0) * 0.1
            lo -= delta
            hi += delta

    return float(lo), float(hi)


def adaptive_corner_bins(values: np.ndarray, bounds: Tuple[float, float], *, use_log: bool) -> int:
    """Choose a stable corner-plot bin count from the visible samples."""
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


def get_param_meta_from_cfg(cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inferred-parameter metadata from config.json if present."""
    meta: Dict[str, Any] = {}
    for k in (
        "inferred_param_names",
        "inferred_param_labels",
        "inferred_param_prior_types",
        "inferred_param_prior_lo",
        "inferred_param_prior_hi",
        "inferred_param_truth",
        "log_axis_orders_threshold",
    ):
        if k in cfg_obj:
            meta[k] = cfg_obj[k]
    return meta


def _safe_float_array(x: Any, *, name: str) -> Optional[np.ndarray]:
    """Safely compute safe float array."""
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as e:
        logger.warning(f"Could not coerce {name} to float array: {e}")
        return None
    return arr


# =============================================================================
# Parameter metadata / consistency checks
# =============================================================================

param_meta = get_param_meta_from_cfg(cfg)

cfg_param_names = [str(x) for x in param_meta.get("inferred_param_names", [])]
cfg_param_labels = [str(x) for x in param_meta.get("inferred_param_labels", [])]
cfg_prior_types = [str(x) for x in param_meta.get("inferred_param_prior_types", [])]
cfg_prior_lo = _safe_float_array(param_meta.get("inferred_param_prior_lo", []), name="inferred_param_prior_lo")
cfg_prior_hi = _safe_float_array(param_meta.get("inferred_param_prior_hi", []), name="inferred_param_prior_hi")
cfg_truth = _safe_float_array(param_meta.get("inferred_param_truth", []), name="inferred_param_truth")

if cfg_param_names:
    if cfg_param_names != param_names:
        logger.warning(
            "Parameter name mismatch between config.json and posterior_samples.npz. "
            f"config.json names={cfg_param_names}, posterior names={param_names}. "
            "Proceeding using posterior_samples.npz names."
        )
    else:
        logger.info("Parameter names match between config.json and posterior_samples.npz.")
        if cfg_param_labels and len(cfg_param_labels) == len(param_labels):
            param_labels = cfg_param_labels
        prior_types = cfg_prior_types if len(cfg_prior_types) == len(param_names) else ["uniform"] * len(param_names)
        prior_lo = cfg_prior_lo if (cfg_prior_lo is not None and cfg_prior_lo.size == len(param_names)) else None
        prior_hi = cfg_prior_hi if (cfg_prior_hi is not None and cfg_prior_hi.size == len(param_names)) else None
        truth_vals = cfg_truth if (cfg_truth is not None and cfg_truth.size == len(param_names)) else None
else:
    prior_types = ["uniform"] * len(param_names)
    prior_lo = None
    prior_hi = None
    truth_vals = None

# Apply the display transform (labels + unit scaling) once, up front.
samples = np.array(samples, dtype=float, copy=True)
for _j, _name in enumerate(param_names):
    _disp = PARAM_DISPLAY.get(_name)
    if _disp is None:
        continue
    _label, _scale = _disp
    param_labels[_j] = _label
    if _scale != 1.0:
        samples[..., _j] *= _scale
        if prior_lo is not None and _j < prior_lo.size:
            prior_lo[_j] *= _scale
        if prior_hi is not None and _j < prior_hi.size:
            prior_hi[_j] *= _scale
        if truth_vals is not None and _j < truth_vals.size:
            truth_vals[_j] *= _scale
        logger.info(f"display transform: {_name} scaled by {_scale:g} -> {_label}")

orders_threshold = float(cfg.get("log_axis_orders_threshold", 3.0))
logger.info(f"log_axis_orders_threshold={orders_threshold:.3g}")

# Log per-parameter posterior summary (very useful when plots fail)
flat_all = flatten_chain_draw(samples)  # (N, D)
if flat_all.ndim == 2 and flat_all.shape[1] >= 1:
    n_all, d_all = flat_all.shape
    logger.info(f"Posterior flat view: N={n_all}, D={d_all}")
    nonfinite_rows = int(np.sum(~np.isfinite(flat_all).all(axis=1)))
    if nonfinite_rows:
        logger.warning(f"Posterior contains {nonfinite_rows}/{n_all} rows with non-finite values.")
    for j, name in enumerate(param_names[:d_all]):
        v = flat_all[:, j]
        v_fin = v[np.isfinite(v)]
        if v_fin.size == 0:
            logger.warning(f"Posterior[{name}]: no finite samples.")
            continue
        q16, q50, q84 = np.quantile(v_fin, [0.16, 0.5, 0.84])
        msg = f"Posterior[{name}]: median={q50:.6g}, q16={q16:.6g}, q84={q84:.6g}, std={np.std(v_fin):.3g}"
        if truth_vals is not None and j < truth_vals.size and math.isfinite(float(truth_vals[j])):
            msg += f", truth={float(truth_vals[j]):.6g}, median-truth={float(q50 - truth_vals[j]):.3g}"
        logger.info(msg)

        # Prior bound sanity (if known)
        if prior_lo is not None and prior_hi is not None and j < prior_lo.size:
            lo = float(prior_lo[j])
            hi = float(prior_hi[j])
            out_lo = int(np.sum(v_fin < lo))
            out_hi = int(np.sum(v_fin > hi))
            if out_lo or out_hi:
                logger.warning(
                    f"Posterior[{name}] has samples outside prior bounds: lo<{lo:.6g} count={out_lo}, "
                    f"hi>{hi:.6g} count={out_hi}. This should not happen if the transform is correct."
                )
else:
    logger.warning("Posterior samples array is not 2D after flattening; skipping per-parameter summaries.")


# =============================================================================
# Plotting routines
# =============================================================================


def plot_phase_curve() -> None:
    """Plot phase curve."""
    logger.info("Plotting phase_curve.png")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # Preferred display: planet-to-star flux ratio in ppm with the stellar
    # baseline anchored on the eclipse bottoms (F = F_s during occultation), the
    # convention of the JWST phase-curve papers. Zero = star alone; the eclipse
    # bottoms sit at zero and the linear ramp is divided out. Falls back to the
    # raw relative-flux display when no eclipse is covered (synthetic smoke runs).
    f_star = eclipse_anchored_stellar_baseline(times_days, flux_obs)
    if f_star is not None:
        def to_display(f: np.ndarray) -> np.ndarray:
            return (f / f_star - 1.0) * 1.0e6

        ax.set_ylabel(r"Planet-to-star flux, $F_p/F_s$ [ppm]")
        ax.axhline(0.0, lw=0.8, color="0.75", zorder=0)
    elif has_flux_true:
        def to_display(f: np.ndarray) -> np.ndarray:
            return f

        ax.set_ylabel("Relative system flux")
    else:
        _offset = float(np.nanmedian(flux_obs))

        def to_display(f: np.ndarray) -> np.ndarray:
            return (f - _offset) * 1.0e6

        ax.set_ylabel("Relative system flux $-$ median [ppm]")

    ax.plot(times_days, to_display(flux_obs), ".", ms=3, color=COLOR_DATA, label="observed", alpha=0.5)
    if has_flux_true:
        ax.plot(times_days, to_display(flux_true), "-", lw=2, color=COLOR_TRUTH, label="truth (noise-free)")

    if ppc_q is not None:
        ax.plot(times_days, to_display(ppc_q["p50"]), "-", lw=2, color=COLOR_POSTERIOR,
                label="posterior median")
        ax.fill_between(times_days, to_display(ppc_q["p05"]), to_display(ppc_q["p95"]),
                        alpha=0.35, color=COLOR_BAND, label="90% PPC band")

    # Mark transit and secondary eclipse (approx)
    t0 = float(cfg.get("time_transit_days", 0.0))
    for tv, tag in ((t0, "transit"), (t0 + 0.5 * orbital_period_days, "eclipse")):
        if times_days.min() <= tv <= times_days.max():
            ax.axvline(tv, ls="--", lw=1, alpha=0.6, color="0.4")
            ax.annotate(tag, xy=(tv, 0.02), xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=9, color="0.35", ha="left", va="bottom")

    ax.set_xlabel("Time [days]")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "phase_curve.png")


def plot_phase_curve_residuals() -> None:
    """Plot phase curve residuals."""
    logger.info("Plotting phase_curve_residuals.png")
    if ppc_q is not None:
        model = ppc_q["p50"]
        model_label = "model"
    elif has_flux_true:
        model = flux_true
        model_label = "truth"
    else:
        model = np.full_like(flux_obs, np.nanmedian(flux_obs), dtype=float)
        model_label = "median observed flux"
    resid = flux_obs - model
    log_array_stats("phase_curve_residuals", resid)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.errorbar(times_days, resid * 1e6, yerr=obs_sigma_vec * 1e6, fmt=".", ms=3, alpha=0.6, color=COLOR_DATA,
                ecolor="0.7", elinewidth=0.8, capsize=0, label=f"obs $-$ {model_label} ($\\pm 1\\sigma$ noise)")
    ax.axhline(0.0, lw=1, color=COLOR_TRUTH)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Residual [ppm]")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "phase_curve_residuals.png")


def plot_posterior_1d_and_overlay_priors() -> None:
    """For each inferred parameter: histogram posterior; overlay analytic prior density."""
    logger.info("Plotting per-parameter 1D posteriors")
    flat = flatten_chain_draw(samples)  # (N, D)
    if flat.ndim != 2:
        raise ValueError(f"Flattened samples must be 2D (N,D). Got shape={flat.shape}")

    n, d = flat.shape
    logger.info(f"Flattened samples: N={n}, D={d}")

    for j in range(d):
        name = param_names[j] if j < len(param_names) else f"param_{j}"
        label = param_labels[j] if j < len(param_labels) else name

        v = flat[:, j]
        v = v[np.isfinite(v)]
        if v.size == 0:
            logger.warning(f"No finite posterior samples for {name}; skipping 1D posterior plot.")
            continue

        # Plot range: use posterior mass, not the full prior, so concentrated
        # retrievals remain readable.
        prior_bounds: Optional[Tuple[float, float]] = None
        if prior_lo is not None and prior_hi is not None and j < prior_lo.size:
            prior_bounds = (float(prior_lo[j]), float(prior_hi[j]))

        if prior_bounds is not None:
            axis_probe_bounds = prior_bounds
        else:
            qlo, qhi = np.quantile(v, [0.001, 0.999])
            axis_probe_bounds = (float(qlo), float(qhi))

        # Axis scaling: log if the parameter's own prior is log-uniform (its native
        # sampling space), else fall back to the orders-of-magnitude heuristic.
        natural_log = log_axis_for_param(j, v, axis_probe_bounds)
        lo, hi = posterior_visible_range(v, use_log=natural_log, hard_bounds=prior_bounds)
        use_log = display_log_axis((lo, hi))
        logger.info(
            f"{name}: x-range from central {100.0 * POSTERIOR_VISIBLE_MASS:.1f}% posterior mass: "
            f"lo={lo:.6g}, hi={hi:.6g}, use_log={use_log}"
        )

        fig, ax = plt.subplots(figsize=(7.0, 4.0))

        # Histogram bins: use log-spaced bins if log axis, else linear.
        if use_log:
            if lo <= 0.0:
                logger.warning(f"{name}: requested log bins but lo<=0 (lo={lo}); falling back to linear bins.")
                use_log = False
            else:
                bins: Any = np.logspace(np.log10(lo), np.log10(hi), POSTERIOR_HIST_BINS)
        if not use_log:
            bins = POSTERIOR_HIST_BINS

        ax.hist(v, bins=bins, density=True, alpha=0.35, color=COLOR_POSTERIOR, label="posterior (hist)")

        # Grid shared by the prior curve and the posterior KDE overlay below.
        xx = np.logspace(np.log10(lo), np.log10(hi), 400) if (use_log and lo > 0.0) else np.linspace(lo, hi, 400)

        # Smooth posterior density (KDE) on top of the histogram -- overlaid KDEs read
        # more clearly than histograms alone (see chat sources on MCMC visualization
        # best practice). For a log-axis parameter, fit the KDE in log10-space (its
        # natural, well-behaved scale) and map back via the log Jacobian 1/(x*ln10).
        if gaussian_kde is not None and v.size > 5:
            try:
                if use_log:
                    v_kde = v[v > 0.0]
                    log_v = np.log10(v_kde)
                    kde_vals = gaussian_kde(log_v)(np.log10(xx)) / (xx * np.log(10.0)) if np.std(log_v) > 0 else None
                else:
                    kde_vals = gaussian_kde(v)(xx) if np.std(v) > 0 else None
                if kde_vals is not None:
                    ax.plot(xx, kde_vals, lw=2.5, color=COLOR_POSTERIOR, label="posterior (KDE)")
            except Exception:
                logger.debug(f"{name}: KDE overlay failed; showing histogram only.", exc_info=True)

        # Overlay prior density if we know it
        if prior_lo is not None and prior_hi is not None and j < prior_lo.size:
            ptype = str(prior_types[j]).strip().lower() if j < len(prior_types) else "uniform"
            prior_bound_lo = float(prior_lo[j])
            prior_bound_hi = float(prior_hi[j])

            if ptype == "uniform" and prior_bound_hi > prior_bound_lo:
                pdf = np.ones_like(xx) / (prior_bound_hi - prior_bound_lo)
            elif ptype == "log10_uniform":
                # Uniform in log10(x) => p(x) ∝ 1 / x
                if prior_bound_lo <= 0.0 or prior_bound_hi <= prior_bound_lo:
                    pdf = np.full_like(xx, np.nan)
                else:
                    pdf = 1.0 / (xx * np.log(prior_bound_hi / prior_bound_lo))
            else:
                logger.warning(f"{name}: unknown prior type {ptype!r}; not overlaying prior.")
                pdf = np.full_like(xx, np.nan)

            ax.plot(xx, pdf, lw=2, ls=":", color="0.4", label=f"prior ({ptype})")

        # Posterior median + 68% credible interval (always available from samples)
        q16, q50, q84 = quantile_summary(v)
        if math.isfinite(q16) and math.isfinite(q84):
            ax.axvspan(q16, q84, alpha=0.15, color=COLOR_POSTERIOR, label="68% CI")
        if math.isfinite(q50):
            ax.axvline(q50, color=COLOR_POSTERIOR, ls="--", lw=1.5, label=f"median = {q50:.3g}")

        # Truth line if present
        if truth_vals is not None and j < truth_vals.size:
            truth = float(truth_vals[j])
            if math.isfinite(truth):
                ax.axvline(truth, color=COLOR_TRUTH, lw=2, alpha=0.9, label="truth")

        ax.set_xlim(lo, hi)
        ax.set_xlabel(label)
        ax.set_ylabel("PDF")
        if use_log:
            ax.set_xscale("log")

        ax.legend(loc="best", fontsize=9)
        safe_name = name.replace("/", "_").replace(" ", "_")
        save_fig(fig, f"posterior_1d_{safe_name}.png")


def plot_corner_with_text() -> None:
    """Plot the posterior corner plot with the standard `corner` package."""
    logger.info("Plotting corner_posterior.png")
    if corner_lib is None:
        raise RuntimeError("The `corner` package is required for corner_posterior.png. Install with `python -m pip install corner`.")

    flat = flatten_chain_draw(samples)  # (N, D)
    if flat.ndim != 2:
        raise ValueError(f"Flattened samples must be 2D (N,D). Got shape={flat.shape}")
    n, d = flat.shape
    logger.info(f"Corner plot input: N={n}, D={d}")

    # Remove any rows with NaNs/Infs.
    mask = np.isfinite(flat).all(axis=1)
    dropped = int(np.sum(~mask))
    if dropped:
        logger.warning(f"Dropping {dropped}/{n} non-finite posterior draws before corner plot.")
        flat = flat[mask]
        n = flat.shape[0]
    if n == 0:
        logger.error("No finite posterior draws; skipping corner plot.")
        return

    # Ranges: use posterior mass, not full prior bounds, so concentrated
    # posteriors do not render as a tiny corner of the panel.
    ranges: List[Tuple[float, float]] = []
    corner_bins: List[int] = []
    use_log_axis: List[bool] = []
    for j in range(d):
        v = flat[:, j]
        prior_bounds: Optional[Tuple[float, float]] = None
        if prior_lo is not None and prior_hi is not None and j < prior_lo.size:
            prior_bounds = (float(prior_lo[j]), float(prior_hi[j]))
            axis_probe_bounds = prior_bounds
        else:
            qlo, qhi = np.quantile(v, [0.001, 0.999])
            axis_probe_bounds = (float(qlo), float(qhi))

        natural_log = log_axis_for_param(j, v, axis_probe_bounds)
        lo, hi = posterior_visible_range(v, use_log=natural_log, hard_bounds=prior_bounds)
        use_log = display_log_axis((lo, hi))
        ranges.append((lo, hi))
        use_log_axis.append(use_log)
        corner_bins.append(adaptive_corner_bins(v, (lo, hi), use_log=use_log))
        name = param_names[j] if j < len(param_names) else f"param_{j}"
        logger.info(f"corner axis {name}: range={(lo, hi)}, log={use_log}, bins={corner_bins[-1]}")

    truths: Optional[List[Optional[float]]] = None
    if truth_vals is not None and truth_vals.size >= d:
        truths = []
        for x in truth_vals[:d].tolist():
            truth = float(x)
            truths.append(truth if math.isfinite(truth) else None)

    fig = corner_lib.corner(
        flat,
        bins=corner_bins,
        range=ranges,
        axes_scale=["log" if use_log else "linear" for use_log in use_log_axis],
        color=COLOR_POSTERIOR,
        labels=[param_labels[j] if j < len(param_labels) else str(j) for j in range(d)],
        truths=truths,
        truth_color=COLOR_TRUTH,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_quantiles=[0.16, 0.50, 0.84],
        title_fmt=".4g",
        hist_bin_factor=CORNER_HIST_BIN_FACTOR,
        smooth=CORNER_SMOOTH,
        smooth1d=CORNER_SMOOTH,
        levels=(0.393, 0.675, 0.864, 0.955),
        plot_datapoints=False,
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        max_n_ticks=3,
        use_math_text=True,
        quiet=True,
        labelpad=0.10,
        label_kwargs={"fontsize": 15},
        title_kwargs={"fontsize": 11.5, "pad": 7},
        hist_kwargs={"color": COLOR_POSTERIOR, "lw": 1.6},
        contour_kwargs={"colors": "#08306B", "linewidths": 1.1},
        contourf_kwargs={"colors": ["#FFFFFF", "#D3E4F3", "#9EC9E2", "#5BA3CF", "#2171B5"]},
        pcolor_kwargs={"cmap": "Blues"},
    )
    for ax in fig.get_axes():
        ax.tick_params(labelsize=10)

    path = PLOTS_DIR / "corner_posterior.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_smc_diagnostics() -> None:
    """Plot smc diagnostics."""
    logger.info("Plotting SMC diagnostics (if present)")
    if extra is None:
        logger.info("No mcmc_extra_fields.npz; skipping SMC diagnostics plots.")
        return

    if "smc_betas" not in extra.files:
        logger.info("smc_betas not present; skipping SMC diagnostics.")
        return

    betas = np.asarray(extra["smc_betas"]).reshape(-1)
    ess = np.asarray(extra["smc_ess"]).reshape(-1) if "smc_ess" in extra.files else None
    acc = np.asarray(extra["smc_acceptance_rate"]).reshape(-1) if "smc_acceptance_rate" in extra.files else None

    # Different BlackJAX versions may use different key names; accept both.
    logz = None
    if "smc_logZ" in extra.files:
        logz = np.asarray(extra["smc_logZ"]).reshape(-1)
    elif "smc_logZ_increment" in extra.files:
        inc = np.asarray(extra["smc_logZ_increment"]).reshape(-1)
        logz = np.cumsum(np.nan_to_num(inc, nan=0.0))

    n_particles = int(extra["smc_num_particles"]) if "smc_num_particles" in extra.files else int(cfg.get("smc_num_particles", 0) or 0)
    kernel = str(cfg.get("smc_mcmc_kernel", extra["smc_kernel"][()] if "smc_kernel" in extra.files else "unknown"))

    log_array_stats("smc_betas", betas)
    if ess is not None:
        log_array_stats("smc_ess", ess)
    if acc is not None:
        log_array_stats("smc_acceptance_rate", acc)
    if logz is not None:
        log_array_stats("smc_logZ", logz)

    if betas.size >= 2 and np.any(np.diff(betas) < -1e-12):
        logger.warning("smc_betas is not monotonic increasing (unexpected for tempered SMC).")

    if betas.size >= 1:
        logger.info(f"SMC final beta={betas[-1]:.6g} (should be ~1.0 for completed inference)")
        if betas[-1] < 0.999:
            logger.warning(
                "Final beta is < 1.0. This indicates adaptive tempering did not reach the posterior. "
                "In that case, posterior_samples may not represent the true posterior."
            )

    if ess is not None and ess.size > 0 and np.isfinite(ess).any() and n_particles > 0:
        ess_frac_min = float(np.nanmin(ess) / float(n_particles))
        logger.info(f"SMC ESS min fraction={ess_frac_min:.3f} (lower means worse weight degeneracy)")
        if ess_frac_min < 0.1:
            logger.warning(
                "Severe weight degeneracy detected (ESS/N < 0.1). If posteriors look wrong, increase particles, "
                "increase mutation steps, or reduce target_ess_frac."
            )

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Beta schedule
    ax = axs[0, 0]
    ax.plot(np.arange(betas.size), betas, marker="o", ms=4, lw=1.5)
    ax.set_xlabel("tempering step")
    ax.set_ylabel("beta")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Adaptive tempering schedule")

    # ESS
    ax = axs[0, 1]
    if ess is not None and ess.size > 0 and np.isfinite(ess).any():
        ax.plot(np.arange(ess.size), ess, marker="o", ms=4, lw=1.5)
        ax.set_xlabel("tempering step")
        ax.set_ylabel("ESS")
        title = "ESS after reweighting"
        if n_particles > 0:
            title += f" (N={n_particles})"
        ax.set_title(title)
        if n_particles > 0:
            ax2 = ax.twinx()
            ax2.plot(np.arange(ess.size), ess / float(n_particles), marker=".", ms=6, lw=1.0)
            ax2.set_ylabel("ESS / N")
            ax2.set_ylim(0.0, 1.05)
    else:
        ax.text(0.5, 0.5, "ESS not saved", ha="center", va="center")
        ax.axis("off")

    # Acceptance
    ax = axs[1, 0]
    if acc is not None and np.isfinite(acc).any():
        ax.plot(np.arange(acc.size), acc, marker="o", ms=4, lw=1.5)
        ax.set_xlabel("tempering step")
        ax.set_ylabel("mean acceptance")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Mutation acceptance ({kernel})")
    else:
        ax.text(0.5, 0.5, "Acceptance not saved", ha="center", va="center")
        ax.axis("off")

    # log Z
    ax = axs[1, 1]
    if logz is not None and np.isfinite(logz).any():
        ax.plot(np.arange(logz.size), logz, marker="o", ms=4, lw=1.5)
        ax.set_xlabel("tempering step")
        ax.set_ylabel("log Z (cumulative)")
        ax.set_title("SMC log normalizer (diagnostic)")
    else:
        ax.text(0.5, 0.5, "logZ not saved", ha="center", va="center")
        ax.axis("off")

    save_fig(fig, "smc_diagnostics.png")


def plot_maps() -> None:
    """Plot maps."""
    logger.info("Plotting maps.png (if present)")
    if maps is None:
        logger.info("No maps file; skipping maps.png")
        return

    required_keys = {"lon", "lat", "phi_truth", "T_truth", "I_truth", "phi_post", "T_post", "I_post"}
    missing = sorted(required_keys - set(maps.files))
    if missing:
        logger.warning(f"maps file missing required keys {missing}; skipping maps.png")
        return

    lon = np.asarray(maps["lon"])
    lat = np.asarray(maps["lat"])

    def _edges_1d(x: np.ndarray, *, is_lat: bool) -> np.ndarray:
        """Edges 1d."""
        x = np.asarray(x).reshape(-1)
        if x.size < 2:
            return np.array([x[0] - 0.5, x[0] + 0.5])
        edges = np.zeros(x.size + 1)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * (x[1] - x[0])
        edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
        if is_lat:
            edges[0] = -0.5 * np.pi
            edges[-1] = 0.5 * np.pi
        return edges

    def _pcolormesh(ax, lon_rad: np.ndarray, lat_rad: np.ndarray, z: np.ndarray, title: str,
                    *, cmap: str = "viridis", cbar_label: str = "") -> None:
        """Render a `pcolormesh` panel with consistent axes and color scaling."""
        lon_edges = _edges_1d(lon_rad, is_lat=False)
        lat_edges = _edges_1d(lat_rad, is_lat=True)
        lon_e, lat_e = np.meshgrid(lon_edges, lat_edges)
        pcm = ax.pcolormesh(np.degrees(lon_e), np.degrees(lat_e), z, shading="auto", cmap=cmap, rasterized=True)
        ax.set_xticks(np.arange(-180.0, 181.0, 60.0))
        ax.set_yticks(np.arange(-60.0, 61.0, 30.0))
        # substellar point (lon=0, lat=0) for orientation
        ax.plot(0.0, 0.0, marker="+", ms=10, mew=1.5, color="w")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)
        ax.get_figure().colorbar(pcm, ax=ax, shrink=0.85, label=cbar_label)

    def intensity_title(base: str) -> str:
        """Compute intensity title (mathtext; unicode ∝/λ glyphs are missing from
        some sans fonts and render as boxes)."""
        mode = str(cfg.get("emission_model", "bolometric")).strip().lower()
        if mode == "bolometric":
            return f"{base} ($I \\propto T^4$)"
        if mode == "planck":
            band = cfg.get("planck_band_wavelengths_m", None)
            if band:
                try:
                    lam_lo, lam_hi = 1e6 * float(min(band)), 1e6 * float(max(band))
                    return f"{base} ($I \\propto \\Sigma\\, w\\, B_\\lambda[T]$, {lam_lo:.3g}$-${lam_hi:.3g} $\\mu$m)"
                except Exception:
                    return f"{base} ($I \\propto \\Sigma\\, w\\, B_\\lambda[T]$)"
            lam_m = cfg.get("planck_wavelength_m", None)
            if lam_m is None:
                return f"{base} ($I \\propto B_\\lambda[T]$)"
            try:
                lam_um = 1e6 * float(lam_m)
                return f"{base} ($I \\propto B_\\lambda[T]$, $\\lambda$={lam_um:.3g} $\\mu$m)"
            except Exception:
                return f"{base} ($I \\propto B_\\lambda[T]$)"
        return f"{base} (I; emission_model={mode})"

    has_truth_maps = bool(np.isfinite(np.asarray(maps["phi_truth"])).any())
    has_post_maps = bool(np.isfinite(np.asarray(maps["phi_post"])).any())
    if not (has_truth_maps or has_post_maps):
        logger.warning("maps arrays are entirely non-finite; skipping maps.png (re-run the maps stage of run_smc.py).")
        return

    def _nonneg(z: np.ndarray, name: str) -> np.ndarray:
        """T and I are physically non-negative (pipeline floors T at Tmin_K > 0);
        clip for display and warn loudly if a stale/buggy file violates that."""
        z = np.asarray(z)
        n_neg = int((z < 0.0).sum())
        if n_neg:
            logger.warning(f"maps.{name}: {n_neg} negative pixels (min={np.nanmin(z):.4g}); "
                           "clipping at 0 for display — the pipeline should never produce these.")
            z = np.clip(z, 0.0, None)
        return z

    phi_label = r"$\Phi$ [m$^2$ s$^{-2}$]"
    if has_truth_maps and has_post_maps:
        fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        _pcolormesh(axs[0, 0], lon, lat, np.asarray(maps["phi_truth"]), r"$\Phi$ truth",
                    cmap="viridis", cbar_label=phi_label)
        _pcolormesh(axs[0, 1], lon, lat, _nonneg(maps["T_truth"], "T_truth"), "T truth",
                    cmap="inferno", cbar_label="T [K]")
        _pcolormesh(axs[0, 2], lon, lat, _nonneg(maps["I_truth"], "I_truth"), intensity_title("I truth"),
                    cmap="magma", cbar_label="I [arb.]")
        post_axs = axs[1]
    else:
        # Real-data run (or missing posterior): a single row of whichever maps exist.
        fig, post_axs = plt.subplots(1, 3, figsize=(14, 3.9), constrained_layout=True)
        if not has_post_maps:
            _pcolormesh(post_axs[0], lon, lat, np.asarray(maps["phi_truth"]), r"$\Phi$ truth",
                        cmap="viridis", cbar_label=phi_label)
            _pcolormesh(post_axs[1], lon, lat, _nonneg(maps["T_truth"], "T_truth"), "T truth",
                        cmap="inferno", cbar_label="T [K]")
            _pcolormesh(post_axs[2], lon, lat, _nonneg(maps["I_truth"], "I_truth"), intensity_title("I truth"),
                        cmap="magma", cbar_label="I [arb.]")
            post_axs = None
    if post_axs is not None:
        _pcolormesh(post_axs[0], lon, lat, np.asarray(maps["phi_post"]), r"$\Phi$ posterior median",
                    cmap="viridis", cbar_label=phi_label)
        _pcolormesh(post_axs[1], lon, lat, _nonneg(maps["T_post"], "T_post"), "T posterior median",
                    cmap="inferno", cbar_label="T [K]")
        _pcolormesh(post_axs[2], lon, lat, _nonneg(maps["I_post"], "I_post"), intensity_title("I posterior median"),
                    cmap="magma", cbar_label="I [arb.]")
    fig.suptitle("Terminal MY_SWAMPE maps and band intensity ($+$ marks the substellar point)")
    path = PLOTS_DIR / "maps.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_disk_renders() -> None:
    """Render visible disk images from saved Ylm coefficients (truth + posterior median).
    
    Requires jax + jaxoplanet/starry; otherwise skipped.
    """
    logger.info("Plotting disk renders (if possible)")
    if maps is None:
        logger.info("No maps file; skipping disk renders.")
        return

    if "y_truth" not in maps.files or "y_post" not in maps.files:
        logger.info("maps file missing y_truth/y_post; skipping disk renders.")
        return

    try:
        import jax.numpy as jnp
        from jaxoplanet.starry.surface import Surface
        from jaxoplanet.starry.ylm import Ylm
    except Exception as e:
        logger.info(f"jax/jaxoplanet not importable; skipping disk renders. Error: {e}")
        return

    ydeg = int(cfg.get("ydeg", 10))
    inc = float(cfg.get("map_inc_rad", math.pi / 2))
    obl = float(cfg.get("map_obl_rad", 0.0))
    phase0 = float(cfg.get("phase_at_transit_rad", math.pi))
    time_transit = float(cfg.get("time_transit_days", 0.0))
    render_res = int(cfg.get("render_res", 250))
    render_phases = cfg.get("render_phases", [0.0, 0.25, 0.49, 0.51, 0.75])
    render_phases = [float(x) for x in render_phases]

    lm_list: List[Tuple[int, int]] = [(ell, m) for ell in range(ydeg + 1) for m in range(-ell, ell + 1)]

    def ylm_from_dense(y_dense: np.ndarray) -> Ylm:
        """Convert a dense coefficient array into the flattened harmonic ordering used downstream."""
        y = jnp.asarray(y_dense)
        data = {lm: y[i] for i, lm in enumerate(lm_list)}
        return Ylm(data)

    def make_surface(y_dense: np.ndarray) -> Surface:
        """Create a `Surface` object from dense harmonic coefficients."""
        return Surface(
            y=ylm_from_dense(y_dense),
            u=(),
            inc=jnp.asarray(inc),
            obl=jnp.asarray(obl),
            period=jnp.asarray(orbital_period_days),
            phase=jnp.asarray(phase0),
            amplitude=jnp.asarray(1.0),
            normalize=False,
        )

    def safe_render(surface: Surface, phase: float, res: int) -> np.ndarray:
        """Render a surface map while tolerating signature differences and failures."""
        try:
            sig = inspect.signature(surface.render)
            if "theta" in sig.parameters:
                img = surface.render(theta=jnp.asarray(phase), res=res)
            elif "phase" in sig.parameters:
                img = surface.render(phase=jnp.asarray(phase), res=res)
            else:
                img = surface.render(res=res)
        except Exception:
            img = surface.render(res=res)
        return np.asarray(img)

    def render_grid(y_dense: np.ndarray, label: str, filename: str) -> None:
        """Render grid."""
        surface = make_surface(y_dense)
        fig, axs = plt.subplots(1, len(render_phases), figsize=(3.2 * len(render_phases), 3.0), constrained_layout=True)
        if len(render_phases) == 1:
            axs = [axs]
        for ax, ph in zip(axs, render_phases):
            t = time_transit + ph * orbital_period_days
            # Matches the pipeline's rotation convention (negative Surface period):
            # the sub-observer longitude DECREASES with time for a tidally locked
            # prograde planet, so an eastward hot spot faces us before eclipse.
            theta = phase0 - 2.0 * math.pi * (t - time_transit) / orbital_period_days
            img = safe_render(surface, theta, render_res)
            # The Ylm-projected intensity can undershoot (Gibbs ringing); brightness
            # is physically non-negative, so clip for display.
            img = np.clip(img, 0.0, None)
            ax.imshow(img, origin="lower", cmap="inferno")
            ax.set_title(f"{label}\nphase = {ph:.2f}")
            ax.axis("off")
        path = PLOTS_DIR / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved {path}")

    y_truth = np.asarray(maps["y_truth"])
    if np.isfinite(y_truth).all():
        render_grid(y_truth, "Truth", "disk_renders_truth.png")
    else:
        logger.info("y_truth is NaN (real-data run); skipping truth disk renders.")
    y_post = np.asarray(maps["y_post"])
    if np.isfinite(y_post).all():
        render_grid(y_post, "Posterior median", "disk_renders_posterior.png")
    else:
        logger.warning("y_post contains non-finite values; skipping posterior disk renders "
                       "(re-run the maps stage of run_smc.py).")


# =============================================================================
# RUN (with per-step exception logging)
# =============================================================================


def _run_step(name: str, fn) -> Optional[str]:
    """Advance one step of the surrounding iterative procedure."""
    logger.info(f"--- {name} ---")
    t0 = time.perf_counter()
    try:
        fn()
    except Exception:
        logger.exception(f"FAILED step: {name}")
        return name
    dt = time.perf_counter() - t0
    logger.info(f"Finished {name} in {dt:.2f} s")
    return None


logger.info("Generating plots...")

failures: List[str] = []
for step_name, step_fn in [
    ("phase_curve", plot_phase_curve),
    ("phase_curve_residuals", plot_phase_curve_residuals),
    ("posterior_1d", plot_posterior_1d_and_overlay_priors),
    ("corner", plot_corner_with_text),
    ("smc_diagnostics", plot_smc_diagnostics),
    ("maps", plot_maps),
    ("disk_renders", plot_disk_renders),
]:
    failed = _run_step(step_name, step_fn)
    if failed is not None:
        failures.append(failed)

if maps is not None:
    maps.close()
if extra is not None:
    extra.close()

if failures:
    logger.error(f"Plotting completed with failures: {failures}")
    logger.error(f"See {log_path} for the full traceback(s).")
    raise SystemExit(1)

logger.info(f"DONE. Plots saved to {PLOTS_DIR.resolve()}")
logger.info(f"Log written to {log_path.resolve()}")
logger.info(f"=== plot_smc diagnostics end ({_utc_ts()}) ===")
