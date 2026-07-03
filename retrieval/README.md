# Differentiable SWAMP → phase-curve retrieval

**This README is the single human-maintained document for the retrieval suite.**
Every production choice — numerics, sampler, data preparation, priors, budget —
is specified and justified here. The only other prose files under `retrieval/`
are machine-generated per-run artifacts (`RETRIEVAL_SUMMARY.md` in an output
directory) and provenance JSON; if you find another doc floating around, it is
stale — fold it in here and delete it.

A downstream **application** of `my_swamp` (not part of the core package). It runs
end-to-end Bayesian retrievals that differentiate through the full shallow-water
time integration to recover a tidally locked planet's **governing forcing
timescales** (radiative `tau_rad`, drag `tau_drag`, and optionally other GCM
scalars) from a noisy thermal phase curve:

```
parameters → SWAMP (my_swamp) terminal Φ → brightness-temperature map → intensity map
           → starry/jaxoplanet spherical-harmonic phase curve → Gaussian likelihood
```

Because the forward model is differentiable, inference uses a **gradient-informed
sampler**: BlackJAX **adaptive tempered SMC** — a swarm of particles annealed from
prior to posterior with a (preconditioned) MALA mutation kernel. SMC is the design
target because the particle swarm vmaps onto a GPU.

## Layout

```
retrieval/
├── README.md          # this file — THE doc; all choices live here
├── scripts/           # all shared code + launchers + style guide
│   ├── pipeline.py        # importable core (forward model, likelihood, SMC)
│   ├── run_smc.py         # driver: build → observe → SMC → write outputs
│   ├── plot_smc.py        # outputs → figures
│   ├── make_dashboard.py  # one consolidated results figure
│   ├── summarize_run.py   # outputs → RETRIEVAL_SUMMARY.md (generated artifact)
│   ├── coverage_study.py  # SBC / coverage (the "run N at once" workload)
│   ├── run.sh             # SLURM/local launcher (JPL edge GPU)
│   ├── run_nas.pbs        # PBS launcher (NASA NAS GH200)
│   ├── full_retrieval.ipynb  # Colab launcher (same pipeline, no SLURM/conda)
│   ├── science.mplstyle   # publication style (applied to all plots)
│   └── tests/             # pytest correctness suite
├── data/ , plots/     # synthetic-run outputs/figures (regenerable, gitignored)
└── wasp_43b_test/     # the real-data WASP-43b retrieval
    ├── config/wasp43b_production_gpu.json   # PRODUCTION config (30 h budget)
    ├── config/wasp43b_pilot_gpu.json        # 2026-07-02 pilot (kept for provenance)
    ├── run_nas_wasp43b.pbs / run_slurm_wasp43b.sh   # cluster launchers
    ├── scripts/           # fetch / prepare / GCM-comparison helpers
    ├── data/provenance/   # Zenodo + preparation provenance JSON
    └── outputs/ , plots/  # run products (regenerable)
```

## Environment

Use the project conda env (has `jaxoplanet` 0.1.0 + `blackjax` 1.3):

```bash
conda activate MY_SWAMP
```

`pipeline.py` prepends this working tree's `src/` to `sys.path`, so it always uses
the in-tree (current, differentiable, x64-aware) `my_swamp`, never a stale
pip-installed copy.

## Quickstart

```bash
cd retrieval/scripts

# fast local CPU smoke retrieval (~2-day spin-up, float32, ~30-55 min) -> ../data/
SWAMP_RETRIEVAL_PRESET=fast python run_smc.py

# figures -> ../plots/ ; recovery report -> ../data/RETRIEVAL_SUMMARY.md
python plot_smc.py
python make_dashboard.py
python summarize_run.py

# correctness tests (fast subset, ~4 min):
python -m pytest tests -q -m "not slow"
# include the end-to-end SMC tests:
python -m pytest tests -q
```

No local GPU/conda env? Open `scripts/full_retrieval.ipynb` in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imalsky/MY_SWAMP/blob/master/retrieval/scripts/full_retrieval.ipynb)

Presets and overrides (env vars, read before JAX import):
`SWAMP_RETRIEVAL_PRESET` = `fast` | `gpu` | `prod`; `SWAMP_RETRIEVAL_USE_X64` =
`0`/`1`; `SWAMP_RETRIEVAL_OVERRIDES` = inline JSON of Config fields;
`SWAMP_RETRIEVAL_OVERRIDES_FILE` = JSON file of Config fields (how the WASP-43b
configs are applied).

---

# The WASP-43b production run: every choice, and why

The production retrieval fits the public JWST/MIRI phase curve of WASP-43 b with
six free parameters. Submit with either launcher (both fetch + prepare the data,
then run SMC + plots; walltime 36 h covers the budgeted worst case):

```bash
qsub retrieval/wasp_43b_test/run_nas_wasp43b.pbs        # NASA NAS GH200
sbatch retrieval/wasp_43b_test/run_slurm_wasp43b.sh     # JPL edge GPU (SLURM)
```

Config: `wasp_43b_test/config/wasp43b_production_gpu.json`. To reproduce the
2026-07 pilot instead, set
`SWAMP_RETRIEVAL_OVERRIDES_FILE=.../config/wasp43b_pilot_gpu.json`.

### Pilot post-mortem (what this run fixes)

The 2026-07-02 pilot (15.1 h, 37 tempering stages, N=64, explicit `dt=120 s`)
completed and produced a good fit, but had three defects:

1. **Frozen mutation kernel.** The MALA step size was tuned once at prior scale
   (0.00857) and never re-adapted; as tempering concentrated the posterior,
   acceptance collapsed 1.0 → 0.001 over the last ~15 stages, so the final
   "512 samples" contained only **25 unique particles**. Medians were fine;
   interval shapes and the lumpy corner-plot marginals were sampling artifacts.
2. **Eclipse-contact residual spikes.** −1000 to −1500 ppm (~15–19 σ) residuals
   confined to the secondary-eclipse ingress/egress windows (fixed-geometry
   eclipse model vs. real contacts), inflating χ²/dof to 6.7 and driving the
   fitted noise inflation to 2.6.
3. **Expensive forward model.** Explicit `dt=120 s` + retuned `K6=5e33` cost
   ~24.5 min per tempering stage, capping the affordable particle count at 64.

### Data (unchanged science, one new mask)

- **Source**: Bell et al. 2024 reduced light curves (Zenodo
  `10.5281/zenodo.10525170`, Eureka! v1, JWST DD-ERS 1366 MIRI/LRS; times are
  BMJD_TDB). Both launchers fetch + prepare automatically; to do it locally:

  ```bash
  cd retrieval/wasp_43b_test
  python scripts/fetch_wasp43b_data.py            # -> data/raw/WASP43b_MIRI_Data.zip
  python scripts/prepare_wasp43b_observations.py  # -> outputs/observations.npz
  ```

  Preparation provenance (masks, geometry, bin segments, ephemeris) is written
  to `wasp_43b_test/data/provenance/wasp43b_preparation.json`.
- **Preparation**: combine the 5.0–10.5 µm channels (inverse-variance), mask bad
  samples + the first 779 integrations (MIRI ramp, as Bell et al.), mask primary
  transit (this model has no transit-depth physics), bin to 320 points, inflate
  per-bin errors ×1.25 (Bell et al.'s broadband scatter multiplier). The
  stellar-Planck-corrected channel weights (`T_star = 4400 K`) feed the model's
  band-integrated Planck emission.
- **NEW — eclipse-edge mask (production)**: the ingress/egress contact windows of
  both secondary eclipses (contact times from the Esposito et al. 2017 geometry,
  T14/2 ≈ 35.5 min, T23/2 ≈ 18.3 min, ± 0.002 d pad) are excluded — 533 of the
  ~7400 retained integrations. The retrieval's eclipse model has fixed geometry
  and no timing freedom, and the pilot showed the few contact-window points
  dominate χ². In-eclipse and out-of-eclipse points are all kept. Binning is
  **segment-aware**: the retained points are split into contiguous segments at
  every masked gap (ramp, transit, contact windows — 8 segments for this visit)
  and binned per segment, so no bin ever averages flux across a gap (an
  equal-count bin straddling a contact would mix out-of-eclipse and in-eclipse
  flux ~the eclipse depth apart, recreating the artifact). Output is still 320
  bins, none inside a contact window. Disable with `--no-eclipse-edge-mask`;
  the mask, geometry, and per-segment bin counts are recorded in the
  preparation provenance.
- **Ephemeris**: Ivshina & Winn 2022 (`P = 0.813474037 d`,
  `T0(BJD_TDB) = 2457423.449697`). Do **not** use the NASA Archive default
  (Hellier 2011) — it lands ~6 min late at the JWST epoch.
- **System parameters** (fixed): Esposito et al. 2017, the source Bell et al.
  2024 adopted — `M*=0.688 Msun`, `R*=0.6506 Rsun`, `Rp=1.006 Rjup`, `b=0.689`,
  `g=49.66 m/s²`. `a_planet_m` in the config is the **shallow-water sphere
  radius** (planet radius), not the orbital semi-major axis (jaxoplanet derives
  the orbit from `star_mass_msun` + period → a/R* = 4.98 vs Esposito's 4.97±0.14).

### Forward model numerics (new: semi-implicit + RAW + mixed precision)

- **Scheme: `semi_implicit=true`, `dt=600 s`, default `K6=1.24e33`,
  `raw_filter=true` (`williams_alpha=0.53`), filter strength `alpha=0.05`,
  `si_alpha=0.5`.** The semi-implicit gravity-wave leapfrog + exponential
  hyperdiffusion (MY_SWAMP readme §9, CLAUDE.md §13.3) removes the gravity-wave
  dt ceiling that forced the pilot's `dt=120`/`K6=5e33`; ~6× cheaper per
  likelihood evaluation (5× fewer steps, ~17% cheaper steps).
  - **Corner validation** (2026-07-02, `scripts/benchmark_new_numerics.py`
    stage `corners`): at these exact settings 13/16 prior-box corners are stable
    over 20 days — including both corners that kill the explicit pilot solver.
    The 3 failures are the `DPhieq/Phibar = 2.5` nightside-collapse corners,
    which are (a) soft-rejected by the NaN → −1e30 likelihood guard and (b)
    strongly data-excluded: the pilot posterior's contrast tops out at 1.96.
  - **Scheme conditioning (report this with the results)**: vs the explicit
    reference equilibrium the SI scheme has an identical hot-spot offset and a
    ~4.6% smaller day-night amplitude. This is a fixed scheme-level offset —
    posteriors are conditioned on the SI forward model and should not be
    compared numerically against explicit-scheme posteriors without noting it.
    It is well inside the pilot's posterior widths (tau_rad 68% CI spans ±25%).
  - Within the scheme, the equilibrated state is dt-converged to ~1e-5 from
    `dt=120` to `2400`, so `dt=600` is conservative.
- **`mixed_precision=true`** (f32 dynamics scan, f64 emission → starry →
  light-curve stage). Forward flux matches pure f64 to **0.006 ppm** on a
  ~4100 ppm curve (data noise: 78 ppm); the fwd-JVP likelihood gradient matches
  to ~1e-4 relative. The all-f32 failure was localized to the eclipse-contact
  derivatives in the light-curve stage, which stays f64. The claimed ~2× GPU
  speedup is unverified on GH200 (1.26× on CPU) and is treated as margin, not
  load-bearing budget.
- **`model_days=20`, `M=42`**: unchanged from the pilot (equilibrated; ≥10
  tau_drag across the prior box).

### Sampler (new: per-stage adaptive preconditioned MALA, N=256)

- **`mcmc_stage_adapt=true`** — the fix for pilot defect 1. After every
  tempering stage the loop (a) re-adapts the MALA step size toward
  `mcmc_target_accept_mala=0.574` (the MALA optimum; Robbins–Monro on log step,
  gain 1.0), and (b) recomputes a **diagonal proposal preconditioner** from the
  weighted particle std per u-dimension (unit geometric mean, clipped to
  [1/20, 20]) so the proposal tracks the posterior's shape and anisotropy as
  tempering narrows it. The kernel (`_build_preconditioned_mala_kernel`) reduces
  exactly to `blackjax.mala` for unit scale; its MH invariance is unit-tested on
  an anisotropic Gaussian. The one-shot pilot tuner is skipped in this mode.
- **`smc_num_particles=256`** (pilot: 64) and **`smc_num_mcmc_steps=30`**
  (pilot: 20), funded by the ~6× cheaper forward model. More particles → real
  posterior resolution; longer chains + live acceptance → real per-stage mixing.
- **Diagnostics to watch in the log**: each stage line now prints
  `unique=<n>/<N>` (unique particle count — the pilot's failure mode is directly
  visible if it ever drops toward ~25) and `step_size=<used> -> <next>`. The
  full histories land in `mcmc_extra_fields.npz`
  (`smc_step_size_history`, `smc_unique_particles`, `smc_scale_diag_final`).
- **Checkpointing**: `smc_checkpoint.npz` is atomically rewritten after every
  stage; a walltime kill loses at most one stage.

### Inference setup (unchanged from the pilot)

- **Six inferred parameters**: `tau_rad`, `tau_drag`, `F_p/F_s`, `Phibar`,
  `DPhieq`, and a multiplicative noise inflation (σ scale). Everything
  constrained by independent measurements (ephemeris, radii, masses, impact
  parameter) is fixed; everything constrained only by this dataset is inferred.
- **Priors** (log-uniform; unchanged so pilot/production posteriors are directly
  comparable): tau_rad, tau_drag ∈ [0.5, 48] h; F_p/F_s ∈ [1e-4, 8e-3];
  `Phibar` ∈ [2e6, 8e6] m²/s² (mean brightness temperature Phibar/R_d ~
  530–2100 K); `DPhieq` ∈ [5e5, 5e6]; noise inflation ∈ [0.5, 5].
- **Likelihood**: Gaussian with per-point σ, a profiled linear-in-time baseline
  (`likelihood_baseline_mode="linear_time"`), and the inferred σ-scale
  multiplying every error bar. Non-finite forward models → log-likelihood −1e30.
- **Emission**: band-integrated Planck over the 11 MIRI channels with
  stellar-Planck-corrected weights; `T = (Phibar + Φ)/R_d`, `R_d = 3.78e3`.
- **Orientation**: eastward hot spot peaks **before** secondary eclipse (negative
  rotation period in jaxoplanet; pinned by
  `tests/test_pipeline.py::test_eastward_hot_spot_peaks_before_eclipse`).

### Budget (the 30 h envelope)

Cost model from the measured pilot (24.5 min/stage at N=64, 20 MALA steps,
`dt=120`): ×4 particles, ×1.5 mutation steps, ÷6 semi-implicit forward →
~25 min/stage; 40–50 stages → **17–21 h** with zero credit for mixed precision,
**~9–12 h** if the ~2× GPU speedup materializes. Walltime is 36 h; per-stage
checkpointing bounds any overrun loss.

### Modeling caveats (carried honestly)

- The likelihood profiles out only a linear-in-time baseline; Bell et al. 2024
  additionally fit an exponential ramp + detector decorrelation. The inferred
  noise inflation absorbs (does not model) residual red noise.
- The stellar spectrum is approximated as a blackbody in the band weights.
- The planet map is the terminal SWAMP snapshot, assumed static in the
  corotating frame over the 26.5 h visit.
- Results are conditioned on the one-layer shallow-water forward model and the
  semi-implicit scheme (see above); expect σ-inflation > 1 even after the
  eclipse-edge mask.

---

# Reference

## The science (what to expect)

The forward model is a forced-dissipative shallow-water tidally locked planet
(Perez-Becker & Showman 2013-style forcing):

- **`tau_rad` sets the day-night amplitude** (strong signal); **`tau_drag` sets
  the eastward hot-spot offset** (weaker). A disk-integrated phase curve
  constrains ~2 longitudinal harmonics, so `tau_rad` is tight while `tau_drag`
  is broader and partially degenerate with `tau_rad` and `Phibar` (gravity-wave
  speed). The degeneracy is real and shown honestly in the corner plot.
- **Spin-up**: the thermal pattern equilibrates in a few `tau_rad`; the
  jet/offset needs ~10 `tau_drag`.
- **Emission layer** (`cfg.emission_temp_mode`): default `"geopotential"`,
  `T = (Φ̄+Φ)/R_d` (`R_d=3.78e3`); `"linear"` is a tunable toy. Intensity is
  `T⁴` (bolometric) or Planck (single wavelength or band-integrated). For real
  targets choose `Φ̄` so `Φ̄/R_d` lands at the planet's actual brightness
  temperature — the Planck curvature is wrong otherwise.
- **Noise models**: `"white"` (constant σ) or `"photon"` (heteroscedastic);
  `infer_noise_inflation=True` adds the σ-scale parameter (standard for real
  light curves).
- **Priors**: timescales use log-uniform (standard for scale parameters).

Inferring more parameters (`Phibar`, `DPhieq`, `omega`, …) is supported via the
`infer_*` flags; those trigger the general path that rebuilds `static` each
evaluation (still differentiable). Shape-changing parameters (`M`, `dt`) are not
inferrable.

## Precision

`float32` is the default for local iteration and validated posterior-unbiased on
the synthetic problem (forward ~1e-6 of f64, log-posterior shape <0.03 of ~470
log-units, identical MAP). Production real-data runs use `use_x64=True` +
`mixed_precision=True` (see the production section above for the measured
agreement).

## Synthetic GPU run (`sbatch scripts/run.sh`)

The `gpu` preset runs the paper-aligned synthetic retrieval (64-particle swarm,
20-day spin-up, photon noise, float64; truth `tau_rad=10 h`, `tau_drag=6 h`,
`Phibar=3e5`, `DPhieq=1e6`, `dt=240`). The likelihood uses a custom
forward-mode-JVP gradient, so there is no reverse-mode tape through the scan —
peak memory is `O(n_particles · J · I)`, not `O(n_steps · …)`. The launchers do
not `module load cuda` (that shadows the bundled-CUDA wheel → silent CPU
fallback) and abort if the backend isn't GPU. For the synthetic preset, A100
throughput saturates at a few dozen simultaneous trajectories, so N=64 is its
sweet spot; the WASP-43b production run instead spends its cheaper semi-implicit
forward on N=256. For many independent retrievals (calibration), use
`coverage_study.py` as a SLURM array (`--n_sim 1 --seed $SLURM_ARRAY_TASK_ID`,
then `--aggregate`).

## Outputs

`run_smc.py` writes to the output dir (`retrieval/data/` for synthetic,
`wasp_43b_test/outputs/` for WASP-43b): `config.json`, `observations.npz`,
`posterior_samples.npz`, `mcmc_extra_fields.npz` (SMC diagnostics incl. step
size / unique-particle histories), `posterior_predictive*.npz`,
`maps_truth_and_posterior_summary.npz`, `smc_checkpoint.npz`, `run.log`, and the
generated `RETRIEVAL_SUMMARY.md`. `plot_smc.py` + `make_dashboard.py` write the
figures (dashboard, phase-curve fit + residuals, 1-D posteriors, corner, SMC
diagnostics, maps, disk renders). All of it is regenerable and gitignored.

## Tests

`scripts/tests/` (run with `conda run -n MY_SWAMP python -m pytest tests -q`
from `retrieval/scripts/`):

- `test_pipeline.py` — config validation, registry, forward parity vs a direct
  my_swamp call, projector, u-space/prior, likelihood, custom-VJP gradient vs
  finite differences, orientation regression, end-to-end SMC recovery (slow).
- `test_production_upgrades.py` — preconditioned-MALA invariance on an
  anisotropic Gaussian, non-finite-proposal rejection, `_weighted_scale_diag`,
  stage-adapt config validation, end-to-end adaptive SMC on a toy target
  (diversity + adaptation assertions), semi-implicit/RAW flag pass-through.
- `test_mixed_precision.py` — mixed-precision forward/gradient agreement.
- `test_wasp43b_prepare.py` — channel combination, binning, ephemeris, and the
  eclipse-edge mask (contact durations, window logic, provenance, and the
  segment-aware-binning guarantee that no bin straddles a masked gap).
- `test_wasp43b_realdata_path.py` — the real-data (non-synthetic) config path.

## Provenance

Restored from git history (`run_mala.py`/`plot_mala.py`, removed in commit
`9d0d400`), refactored into the `pipeline.py` + `run_smc.py` split, then
reorganized into `scripts/` + `data/` + `plots/`. The WASP-43b suite's former
separate README was folded into this file 2026-07-03 (single-doc policy).
