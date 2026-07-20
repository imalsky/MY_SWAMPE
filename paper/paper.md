---
title: 'SWAMPE-JAX: A differentiable JAX implementation of the SWAMPE shallow-water model for exoplanet atmospheres'
tags:
  - Python
  - JAX
  - astronomy
  - exoplanets
  - atmospheric dynamics
  - shallow-water model
  - automatic differentiation
  - general circulation model
authors:
  - name: Isaac Malsky
    orcid: 0000-0003-0217-3880
    corresponding: true
    affiliation: 1
  - name: Tiffany Kataria
    orcid: 0000-0003-3759-9080
    affiliation: 1
  - name: Ekaterina Landgren
    orcid: 0000-0001-6029-5216
    affiliation: 2
affiliations:
  - name: Jet Propulsion Laboratory, California Institute of Technology, Pasadena, CA 91109, USA
    index: 1
  - name: Department of Environmental Social Sciences, Stanford Doerr School of Sustainability, Stanford University, Stanford, CA 94305, USA
    index: 2
date: 20 July 2026
bibliography: paper.bib
---

# Summary

The majority of giant exoplanets with measurable atmospheres are spin-synchronized: one hemisphere is constantly irradiated, while the other never receives direct starlight. This permanent imbalance in thermal forcing, coupled with rotation, results in planets with large day-night temperature differences [@Showman:2009]. Because exoplanets are observed almost entirely through indirect, disk-integrated measurements, characterizing the multi-dimensional nature of these atmospheres relies on comparisons to forward models.

`SWAMPE-JAX` is a Python package for modeling the two-dimensional dynamics and forcing timescales of exoplanet atmospheres. It is a `JAX` [@jax:2018] reimplementation of `SWAMPE` [@Landgren:2022], and the two codes agree to high precision. `SWAMPE-JAX` solves the shallow-water equations for a rotating sphere using the spectral-transform method. `SWAMPE-JAX` advances absolute vorticity, divergence, and geopotential forward in time. The underlying shallow-water model has been validated for synchronously rotating hot Jupiters and applied to sub-Neptunes [@Landgren:2022; @Landgren:2023].

`SWAMPE-JAX` offers two improvements over the original code. First, `SWAMPE-JAX` is differentiable: the simulated geopotential and wind fields, and any quantity derived from them, can be differentiated with respect to the planet's input physical parameters through automatic differentiation. This novel capability returns the sensitivity of the simulated atmosphere to the mechanisms that shape it, enabling sensitivity analysis and gradient-based parameter studies. Second, `SWAMPE-JAX` is roughly $32\times$ faster than the original `SWAMPE` on a single CPU core, and between ${\sim}710\times$ and ${\sim}6{,}330\times$ faster on a GPU (the latter amortized over a batch).

# Statement of need

Observations showing the intrinsic coupling of radiative, chemical, and dynamical processes in exoplanet atmospheres are now ubiquitous. However, our ability to supplement and interpret observations with numerical simulations is limited by computational constraints. There is therefore a pressing need for a multidimensional forward model fast enough to sweep the broad range of conditions *a priori* possible for exoplanet atmospheres and determine the physical drivers that shape them.

On one extreme (in terms of complexity), one-dimensional models are a critical tool for atmospheric characterization. Fast, parametric retrieval frameworks built on these models have revealed the composition, temperature structure, and cloud properties of exoplanets from observations [e.g., @Madhusudhan:2009; @Rustamkulov:2023]. However, they cannot capture the transport and feedbacks between radiation and dynamics that shape atmospheric structure [@Madhusudhan:2019]. On the other extreme, full three-dimensional general circulation models [e.g., @Carone:2020; @Kataria:2016; @Showman:2009] capture the full spatial structure of exoplanet atmospheres, but are extremely computationally demanding. One GCM simulation can cost tens of thousands of CPU-hours and months of real-world wall-clock time [@Wang:2020]. This dramatically restricts the type of investigations that can be performed with this class of models.

Two-dimensional models can serve as a reduced-complexity, computationally tractable intermediate step between 1D retrieval frameworks and full 3D GCMs. Two-dimensional models can capture day--night circulation patterns [e.g., @Perez-Becker:2013; @Showman:2011], equatorial jet formation [@Showman:2011], and global temperature structures. Simulated phase curves and eclipse maps from these models can then be used to infer the underlying physical structure of these planets. For example, the hotspot offset [@Penn:2017], or day--night temperature differences [@Perez-Becker:2013].

`SWAMPE-JAX` was designed to fill this gap. It occupies the intermediate regime between fast one-dimensional retrieval models and computationally intensive three-dimensional GCMs, while retaining the complexity needed to interpret multidimensional atmospheres. `SWAMPE-JAX` is extremely fast (in order to enable broad grid searches). Furthermore, the differentiable `JAX` implementation further enables sensitivity analysis and is designed to support future gradient-informed Bayesian retrieval frameworks [e.g., @Kawahara:2025].

# State of the field

`SWAMPE-JAX` is a full `JAX` rewrite of `SWAMPE` [@Landgren:2022]. It shares the interface and architecture but no source code. Related modeling tools include:

- `SWAMPE`: the original NumPy implementation of this shallow-water model [@Landgren:2022].
- Fully three-dimensional general circulation models that solve a set of fluid dynamics equations coupled to radiative transfer. These models determine horizontal as well as vertical structure, but are far more computationally demanding [e.g., @Carone:2020; @Kataria:2016; @Roth:2024; @Showman:2009].
- Other two-dimensional, single-layer numerical models of close-in giant planets [e.g., @Cho:2008; @LangtonLaughlin:2008; @Menou:2003; @Penn:2017].
- Similar two-dimensional shallow-water or primitive-equation models exist for Earth's atmosphere and oceans [e.g., @Klower:2024]. Some of these are also fully differentiable [e.g., @Davenport:2026; @Kochkov:2024; @Moses:2026].

# Software design

`SWAMPE-JAX` preserves the public interface and the numerical behavior of `SWAMPE`, so that existing workflows port with minimal changes, while adding the capabilities that `JAX` makes possible. Specifically, it reproduces `SWAMPE`'s spectral-transform solver [@Hack:1992], modified-Euler time differencing [@Langton:2008], Robert--Asselin filtering [@Robert:1966; @Asselin:1972], sixth-order hyperdiffusion [@Gelb:2001], and Newtonian-relaxation-plus-drag forcing [@ShowmanGuillot:2002].

Internally, `SWAMPE-JAX` expresses the full time integration as a single `jax.lax.scan` over a JIT-compiled step function: the core design that makes the model both fast and end-to-end differentiable.

Agreement is not exact between `SWAMPE-JAX` and `SWAMPE`, as `XLA` and NumPy do not evaluate floating-point expressions in the same order. However, \autoref{fig:parity} shows that the two codes agree to at least within $2\times10^{-5}\,\%$ over a $100$-day forced, synchronously rotating benchmark integration. Other tests (not shown here) show similar agreement.

![Numerical parity between reference NumPy `SWAMPE` (left column) and `SWAMPE-JAX` (middle column) over a $100$-day forced, synchronously rotating benchmark ($a=8.2\times10^{7}$ m, $\Omega=3.2\times10^{-5}$ s$^{-1}$, $\overline{\Phi}=3\times10^{5}$ m$^{2}$ s$^{-2}$, $\Delta\Phi_{\mathrm{eq}}=10^{6}$ m$^{2}$ s$^{-2}$, $\tau_{\mathrm{rad}}=10$ h, $\tau_{\mathrm{drag}}=6$ h). Rows show the absolute vorticity $\eta$, divergence $\delta$, geopotential $\Phi$, and the zonal and meridional winds $U$ and $V$. Overall, the two codes show very close agreement.\label{fig:parity}](parity_comparison.png)

## Speed (CPU and GPU)

`SWAMPE-JAX` can run on either a CPU or GPU. A ten-day, $M=42$ forced integration ($120$ s timestep, ${\sim}7{,}200$ steps) takes about $817$ s (${\sim}14$ min) for `SWAMPE`. Comparatively, the same run for `SWAMPE-JAX` takes $25$ s on a CPU and about $1.15$ s on a GPU[^1]. This is roughly a $32\times$ (CPU) and $710\times$ (GPU) speedup respectively. This speedup comes mainly from expressing the spectral transforms as vectorized array contractions in place of the reference code's per-element Python loops and compiling the entire time-integration loop into a single fused kernel. Using `jax.vmap`, we can also simulate a batch of models in a single compiled call. On the same A100, throughput saturates at a few dozen simultaneous trajectories, resulting in a speedup per simulation of approximately $6{,}330\times$ compared to the original `SWAMPE`. Combined with gradient-informed Bayesian retrieval methods, this speedup could make `SWAMPE-JAX` viable for use within inverse models.

[^1]: CPU timings were measured on an Apple M3 Pro. GPU timings on a single `NVIDIA A100`.

## Differentiability

Although the code supports both forward and backward mode automatic differentiation (AD), forward mode AD is the better choice given the long model integration times. The computational cost of forward mode differentiation grows with the number of inputs, but the memory cost is independent of the length of the integration. In our case, there are relatively few model inputs (e.g. timescales), but integration length can be hundreds of thousands of steps. While backward mode automatic differentiation is possible in theory with our model, in practice the entire forward trajectory must be held in memory. This can grow to be tens of gigabytes, and can easily exceed the memory of a single GPU.

\autoref{fig:sensitivity} shows how the simulated temperature field responds to changes in the radiative and drag forcing timescales over a $100$-day forced, synchronously rotating integration (the benchmark configuration of \autoref{fig:parity}). The final model state is differentiable with respect to the input parameters (day--night forcing contrast, radiative and drag timescales, mean geopotential, rotation rate, planetary radius, timestep, and hyperdiffusion coefficients). We determine these sensitivities using forward mode automatic differentiation.

![Automatic differentiation sensitivities (from the benchmark configuration of \autoref{fig:parity}) obtained by forward-mode automatic differentiation through a $100$-day integration. From left to right, the panels show the simulated temperatures (where $T=(\overline{\Phi}+\Phi)/R_d$, following @Landgren:2023), and the exact derivative of that field with respect to the radiative timescale, $\partial T/\partial\tau_{\mathrm{rad}}$, and the drag timescale, $\partial T/\partial\tau_{\mathrm{drag}}$. Lengthening either timescale decreases the day--night contrast: the radiative timescale by weakening the thermal forcing, the drag timescale by letting the stronger circulation redistribute more heat.\label{fig:sensitivity}](temperature_sensitivity_perhour_100d.png)

# Research impact statement

`SWAMPE` has already been used to study atmospheric circulation on sub-Neptunes [@Landgren:2023], alongside other 2D shallow-water model investigations of close-in giant planets [e.g., @Cho:2008; @Penn:2017]. `SWAMPE-JAX` is designed to reduce simulation times and allow larger grid explorations, with the goal of more accurately interpreting observations and characterizing exoplanet atmospheres. Additionally, a gradient-informed Bayesian retrieval pipeline built on `SWAMPE-JAX` is in active development.

# AI usage disclosure

The authors used Claude (Anthropic, Opus 4.8 and Fable 5), via the Claude Code CLI, to assist with code development. All AI-assisted code was reviewed and validated by the authors against the package's test suite and the benchmark data in `paper/benchmark_data/`. The authors made all core design decisions and take full responsibility for the accuracy of the software and this manuscript.

# Acknowledgements

This research was carried out in part at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration (80NM0018D0004). We acknowledge the developers of the open-source packages on which this work depends, in particular `JAX` [@jax:2018], `NumPy` [@Harris:2020], `SciPy` [@Virtanen:2020], and `Matplotlib` [@Hunter:2007].

# References
