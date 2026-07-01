# WASP-43 b Real-Data Pilot Retrieval

This folder runs a pilot science retrieval on public JWST/MIRI phase-curve
products for WASP-43 b. It uses the reduced light curves archived with Bell et al.
2024, not a raw JWST reduction.

## Data Provenance

- Target: WASP-43 b.
- Primary product: `WASP43b_MIRI_Data.zip` from Zenodo DOI
  `10.5281/zenodo.10525170`.
- Observation program provenance: JWST DD-ERS 1366 / MIRI LRS phase curve.
- System parameters come from the NASA Exoplanet Archive and are fixed in
  `config/wasp43b_pilot_gpu.json`.

The default preparation reads:

```text
WASP43b_MIRI_Data/1_Light_Curves/eureka_v1.h5
```

It combines the 5.0-10.5 micron spectroscopic channels into a broadband relative
system-flux light curve, masks bad channel samples, masks the first 779
integrations, masks primary transit, keeps secondary eclipse, bins to about 320
points, and inflates per-bin errors by 1.25.

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

## Modeling Caveats

- This is a pilot real-data retrieval, not a paper-grade reduction.
- The likelihood profiles out a linear-in-time baseline for each proposed planet
  model.
- The broadband MIRI passband is approximated with one Planck wavelength:
  `7.75e-6 m`.
- Primary transit is masked because this retrieval models the planet phase curve,
  not stellar transit depth.
- Secondary eclipse is retained because the `jaxoplanet` phase-curve model
  handles occultation.
