# WASP-43 b Real-Data Pilot Retrieval

This folder runs a pilot science retrieval on public JWST/MIRI phase-curve
products for WASP-43 b. It uses the reduced light curves archived with Bell et al.
2024 (Nature Astronomy; JWST DD-ERS 1366), not a raw JWST reduction.

## Data Provenance

- Target: WASP-43 b.
- Primary product: `WASP43b_MIRI_Data.zip` from Zenodo DOI
  `10.5281/zenodo.10525170` (Eureka! v1 light curves; times are BMJD_TDB per the
  file attributes).
- Observation program provenance: JWST DD-ERS 1366 / MIRI LRS phase curve.
- Ephemeris: Ivshina & Winn 2022 (ApJS 259, 62), `P = 0.813474037 d`,
  `T0(BJD_TDB) = 2457423.449697`. (The NASA Exoplanet Archive *default* row is
  still the Hellier et al. 2011 discovery solution; propagated to the JWST epoch
  it predicts the transit ~6 min late, ~1.9 deg of orbital phase — do not use it.)
- System parameters (fixed in `config/wasp43b_pilot_gpu.json`): Esposito et al.
  2017 (A&A 601, A53), the same source Bell et al. 2024 adopted for their
  fiducial fit — `M* = 0.688 Msun`, `R* = 0.6506 Rsun`, `Rp = 1.006 Rjup`,
  `b = 0.689` (`i = 82.109 deg`), `log g_p = 3.696`.

The default preparation reads:

```text
WASP43b_MIRI_Data/1_Light_Curves/eureka_v1.h5
```

It combines the 5.0-10.5 micron spectroscopic channels (11 channels of 0.5 um)
into a broadband relative system-flux light curve by inverse-variance weighting,
masks bad channel samples, masks the first 779 integrations (the MIRI ramp, as
in Bell et al. 2024), masks primary transit, keeps both secondary eclipses, bins
to about 320 points, and inflates per-bin errors by 1.25 (Bell et al.'s
broadband scatter-to-photon-noise ratio). It also writes the per-channel
wavelengths and stellar-Planck-corrected combination weights
(`band_wavelengths_um`, `band_weights`, using `T_star = 4400 K`) that the
retrieval uses for its band-integrated Planck emission model.

## Local Preparation

From this folder:

```bash
python scripts/fetch_wasp43b_data.py --metadata-only
python scripts/fetch_wasp43b_data.py
python scripts/prepare_wasp43b_observations.py
```

This writes:

```text
outputs/observations.npz
data/provenance/zenodo_10525170.json
data/provenance/wasp43b_preparation.json
```

## NAS GH200 Run

Submit from the repo root or this folder:

```bash
qsub retrieval/wasp_43b_test/run_nas_wasp43b.pbs
```

The launcher fetches/prepares data, then delegates to the shared
`retrieval/scripts/run_nas.pbs` JAX/BlackJAX launcher with:

```text
SWAMP_RETRIEVAL_OVERRIDES_FILE=retrieval/wasp_43b_test/config/wasp43b_pilot_gpu.json
SWAMP_PLOT_OUT_DIR=retrieval/wasp_43b_test/outputs
SWAMP_PLOTS_DIR=retrieval/wasp_43b_test/plots
```

Expected outputs:

```text
outputs/config.json
outputs/observations.npz
outputs/posterior_samples.npz
outputs/mcmc_extra_fields.npz
outputs/posterior_predictive_quantiles.npz
outputs/RETRIEVAL_SUMMARY.md
plots/results_dashboard.png
plots/corner_posterior.png
```

## Model Configuration Notes

- `a_planet_m` in the config is the **shallow-water sphere radius** (the planet
  radius, 1.006 Rjup = 7.192e7 m), *not* the orbital semi-major axis — jaxoplanet
  derives the orbit from `star_mass_msun` + the orbital period (giving
  a/R* = 4.98, vs Esposito et al.'s 4.97 +/- 0.14).
- The retrieval infers six parameters: `tau_rad`, `tau_drag`, `F_p/F_s`,
  `Phibar`, `DPhieq`, and a multiplicative noise-inflation factor (sigma
  scale). Everything constrained by *independent* measurements (ephemeris,
  stellar/planet radii and masses, impact parameter) is fixed; everything
  constrained only by this dataset is inferred. Timescale priors are
  log-uniform over 0.5-48 h (the 20-day spin-up is >= 10 tau_drag across the
  prior); `Phibar` is log-uniform over 2e6-8e6 m^2/s^2 (mean brightness
  temperature Phibar/R_d ~ 530-2100 K, bracketing the PBS13 standard gH=4e6)
  and `DPhieq` log-uniform over 5e5-5e6. The config's point values for
  Phibar/DPhieq are initialization only.
- Because Phibar sets the day-night flux-ratio curvature (via the Planck
  function at T = (Phibar+Phi)/R_d) *and* the gravity-wave speed, and DPhieq
  sets the forcing amplitude, inferring them removes the two strongest
  model-conditioning assumptions from the tau posteriors (at the cost of the
  expected amplitude degeneracies, which the corner plot shows honestly).
- Solver: `dt = 120 s`, `K6 = 5e33`. The Phibar~4e6 regime has ~3.7x faster
  gravity waves than the synthetic default (Phibar=3e5, dt=240, K6=1.24e33), so
  the hyperdiffusion is rescaled with the wave speed and dt halved. 20-day
  stability was validated over all 16 corners of the (tau_rad, tau_drag,
  Phibar, DPhieq) prior box: 14/16 stable. The only unstable region requires
  *simultaneously* tau_rad = 48 h, tau_drag = 48 h, and Phibar = 8e6 (both
  damping channels nearly off plus the fastest gravity waves). Proposals there
  are soft-rejected by the NaN likelihood guard, and the region is
  independently excluded by the data (tau_rad = 48 h predicts a near-flat
  phase curve against the observed ~4200 ppm amplitude). Diagnostic: if a
  posterior presses against those three upper bounds at once, rerun with K6
  scaled as sqrt(Phibar/4e6) and revalidate the box.

## Modeling Caveats

- This is a pilot real-data retrieval, not a paper-grade reduction.
- The likelihood profiles out a linear-in-time baseline for each proposed planet
  model; Bell et al. 2024 additionally used an exponential ramp and detector
  decorrelation terms. The inferred noise-inflation parameter absorbs (but does
  not model) residual red noise.
- The broadband MIRI passband is modeled as a weighted sum of Planck functions
  over the 11 combined channels (weights from the data combination, corrected by
  the stellar Planck function at `T_star = 4400 K`); the stellar spectrum is
  approximated as a blackbody.
- Primary transit is masked because this retrieval models the planet phase curve,
  not stellar transit depth.
- Secondary eclipse is retained because the `jaxoplanet` phase-curve model
  handles occultation (with the Esposito et al. 2017 impact parameter, the model
  eclipse duration matches the observed ~1.2 h).
- The planet map is the terminal SWAMP snapshot (a single time slice), assumed
  static in the corotating frame over the 26.5 h visit.
