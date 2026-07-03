# -*- coding: utf-8 -*-
# ruff: noqa: E741
"""my_swamp.semi_implicit_tdiff

Semi-implicit leapfrog time differencing (Hoskins & Simmons 1975) for the
spectral shallow-water equations, paired with exponential (integrating-factor)
hyperdiffusion. This is the opt-in scheme described in CLAUDE.md section 13.3;
it is NOT part of the reference-SWAMPE parity contract and shares no historical
quirks with `modEuler_tdiff` / `explicit_tdiff`.

Scheme
------
Split each tendency into nonlinear terms N(x) (evaluated explicitly at the
current level n) and the *linear* terms, which are treated implicitly with
off-centering ``si_alpha`` (0.5 = centered trapezoid). Two families of linear
terms are handled:

1. **Gravity-wave coupling** (the dt bottleneck in the hot-Jupiter regime):

       d(delta)/dt += lam*Phi          lam = n(n+1)/a^2  (i.e. -del^2)
       d(Phi)/dt   -= Phibar*delta

2. **Linear relaxation/drag** (forced mode only). Leapfrog is unstable for
   damping terms evaluated at the centered level — the computational mode
   grows at ~dt/tau per step — and the retrieval prior box reaches
   tau = 0.5 h, where no explicit treatment survives a large dt. The linear
   parts of the Perez-Becker & Showman forcing are diagonal in spectral
   space and are folded into the same implicit solve:

       Q = (Phieq - Phi - Phibar)/tau_rad          -> -Phi/tau_rad implicit
       (F, G) = (Ru, Rv) - (U, V)/tau_drag         -> -(eta - f)/tau_drag and
                                                      -delta/tau_drag implicit

   The remainders stay explicit and are recovered *exactly* in truncated
   Fourier space by FFT linearity (no extra transforms):

   - Thermal: ``PhiFm + Phim/tau_rad`` = FFT((Phieq - Phibar)/tau_rad), a
     *constant* — Q is exactly linear in Phi, so nothing nonlinear remains.
   - Momentum: the ``Ru, Rv`` mass-source terms (``-U*Q+/(Phi+Phibar)``) are
     genuinely nonlinear and can be *stiff* (their effective damping rate
     approaches contrast/tau_rad, unbounded in the super-contrast regime), so
     they are evaluated at the **lagged** leapfrog level n-1
     (Williamson-style physics lagging, carried in ``State.Rum_lag``/
     ``Rvm_lag``): for a damping term, lagged evaluation is stable for
     2*dt/tau_eff < 2 whereas centered evaluation is unconditionally
     unstable.

Because the Laplacian and the relaxation terms are diagonal per spherical-
harmonic mode, the implicit "solve" is closed-form scalar arithmetic — no
matrix, no iteration, AD-clean:

    xi   = 2*si_alpha*dt
    b    = xi*lam ;  c = xi*Phibar ;  kd = xi/tau_drag ;  kr = xi/tau_rad
    delta_new = [delta* + b*Phi*/(1+kr)] / [(1+kd) + b*c/(1+kr)]
    Phi_new   = [Phi* - c*delta_new] / (1+kr)
    eta_new   = [eta* + kd*fmn] / (1+kd)

where the starred quantities carry the n-1 base, the explicit nonlinear
terms, and the (1-si_alpha) explicit share of every linear term. With
``si_alpha=0.5`` each linear term is a trapezoid (A-stable); gravity waves no
longer limit dt, so the ceiling becomes the advective CFL — typically 10x+
larger where sqrt(Phibar) far exceeds the wind speed.

The exponential hyperdiffusion factors (``filters.sigma6_exponential`` /
``sigma6Phi_exponential``) are the exact solution of the linear del^6 operator
over the step and are unconditionally stable, so K6 does not need to be
retuned as dt grows.
"""

from __future__ import annotations

import jax.numpy as jnp

from .branching import maybe_apply, select
from .dtypes import Scalar
from . import spectral_transform as st


def si_timestep(
    Rum_lag: jnp.ndarray,
    Rvm_lag: jnp.ndarray,
    etam0: jnp.ndarray,
    etam1: jnp.ndarray,
    deltam0: jnp.ndarray,
    deltam1: jnp.ndarray,
    Phim0: jnp.ndarray,
    Phim1: jnp.ndarray,
    I: int,
    J: int,
    M: int,
    N: int,
    Am: jnp.ndarray,
    Bm: jnp.ndarray,
    Cm: jnp.ndarray,
    Dm: jnp.ndarray,
    Em: jnp.ndarray,
    Fm: jnp.ndarray,
    Gm: jnp.ndarray,
    Um: jnp.ndarray,
    Vm: jnp.ndarray,
    fmn: jnp.ndarray,
    Pmn: jnp.ndarray,
    Hmn: jnp.ndarray,
    Pmnw: jnp.ndarray,
    Hmnw: jnp.ndarray,
    tstepcoeff: jnp.ndarray,
    tstepcoeff2: jnp.ndarray,
    tstepcoeffmn: jnp.ndarray,
    marray: jnp.ndarray,
    mJarray: jnp.ndarray,
    narray: jnp.ndarray,
    PhiFm: jnp.ndarray,
    dt: Scalar,
    a: Scalar,
    Phibar: Scalar,
    taurad: Scalar,
    taudrag: Scalar,
    forcflag: bool,
    diffflag: bool,
    relax_implicit: bool,
    sigma_exp: jnp.ndarray,
    sigmaPhi_exp: jnp.ndarray,
    si_alpha: Scalar,
    test,
    t: jnp.ndarray,
) -> tuple:
    """One semi-implicit leapfrog update of (eta, delta, Phi) plus winds.

    Uses the n-1 Fourier levels (``etam0``, ``deltam0``, ``Phim0``) as the
    leapfrog base, the current-level nonlinear spectra (``Am..Em``), and the
    *lagged* (n-1) nonlinear momentum-forcing remainders ``Rum_lag``/
    ``Rvm_lag`` maintained by the scan driver. ``relax_implicit`` must be True
    only when the physical Perez-Becker & Showman forcing is active
    (``forcflag`` and ``test is None``); it moves the linear relaxation/drag
    terms into the implicit solve. Returns the same 13-tuple as
    :func:`my_swamp.time_stepping.tstepping`.
    """

    dtype_r = jnp.zeros(0, dtype=Em.dtype).real.dtype
    two_dt = 2.0 * jnp.asarray(dt, dtype=dtype_r)
    alpha_i = jnp.asarray(si_alpha, dtype=dtype_r)
    lam = narray / (a**2)  # l(l+1)/a^2 per (m,n); -del^2 eigenvalue
    zero = jnp.asarray(0.0, dtype=dtype_r)

    if relax_implicit:
        # Inverse timescales for the implicit relaxation terms. taudrag == -1
        # is the SWAMPE "drag disabled" sentinel (matching forcing.Rfun).
        taudrag_arr = jnp.asarray(taudrag, dtype=dtype_r)
        inv_taudrag = jnp.where(taudrag_arr == -1.0, zero, 1.0 / taudrag_arr)
        inv_taurad = 1.0 / jnp.asarray(taurad, dtype=dtype_r)
        # Explicit shares of the forcing: the stiff nonlinear momentum
        # remainders at the LAGGED level (stable damping treatment), and the
        # constant thermal remainder recovered exactly by peeling the linear
        # -Phi/tau_rad off the current-level spectrum (FFT linearity).
        Fm_expl = Rum_lag
        Gm_expl = Rvm_lag
        PhiFm_expl = PhiFm + Phim1 * inv_taurad
    else:
        inv_taudrag = zero
        inv_taurad = zero
        Fm_expl, Gm_expl, PhiFm_expl = Fm, Gm, PhiFm

    # Effective advective fluxes with the (nonlinear part of the) momentum
    # forcing folded in — forced mode only, no unforced-branch leakage quirks.
    A_eff = select(forcflag, Am - Gm_expl, Am)
    B_eff = select(forcflag, Bm + Fm_expl, Bm)

    # Forward Legendre transforms. tstepcoeff = 2*dt/(a*(1-mu^2)) carries the
    # leapfrog 2*dt; the remaining terms are scaled post-transform.
    p_terms = jnp.stack(
        (
            etam0,
            deltam0,
            Phim0,
            tstepcoeff * (1j) * mJarray * A_eff,
            tstepcoeff * (1j) * mJarray * B_eff,
            tstepcoeff * (1j) * mJarray * Cm,
            Em,
        ),
        axis=0,
    )
    etamn0, deltamn0, Phimn0, imA, imB, imC, Emn = st.fwd_leg_w_batch(p_terms, Pmnw)
    HA, HB, HD = st.fwd_leg_w_batch(
        jnp.stack((tstepcoeff * A_eff, tstepcoeff * B_eff, tstepcoeff * Dm), axis=0), Hmnw
    )

    # Implicit weights (xi = 2*si_alpha*dt) and their explicit (1-si_alpha)
    # shares (gamma = 2*dt*(1-si_alpha)/tau).
    xi = two_dt * alpha_i
    expl = two_dt * (1.0 - alpha_i)
    kd = xi * inv_taudrag
    kr = xi * inv_taurad
    gd = expl * inv_taudrag
    gr = expl * inv_taurad

    # Vorticity: explicit leapfrog + implicit linear drag on (eta - f).
    eta_star = etamn0 - imA + HB - gd * (etamn0 - fmn)
    etamn_new = (eta_star + kd * fmn) / (1.0 + kd)

    # Explicit provisional values for the coupled delta/Phi solve.
    delta_star = (
        deltamn0
        + imB
        + HA
        + two_dt * lam * Emn
        + expl * lam * Phimn0
        - gd * deltamn0
    )
    Phi_star = Phimn0 - imC + HD - expl * Phibar * deltamn0 - gr * Phimn0

    def _add_phi_forcing(x):
        """Add the (nonlinear/constant part of the) geopotential forcing."""
        return x + st.fwd_leg_w(two_dt * PhiFm_expl, Pmnw)

    Phi_star = maybe_apply(forcflag, _add_phi_forcing, Phi_star)

    # Closed-form per-degree implicit solve (gravity waves + relaxation).
    b = xi * lam
    c = xi * Phibar
    deltamn_new = (delta_star + b * Phi_star / (1.0 + kr)) / ((1.0 + kd) + b * c / (1.0 + kr))
    Phimn_new = (Phi_star - c * deltamn_new) / (1.0 + kr)

    # Exponential (integrating-factor) hyperdiffusion — exact and
    # unconditionally stable.
    def _diffuse(coeffs):
        etamn_d, deltamn_d, Phimn_d = coeffs
        return etamn_d * sigma_exp, deltamn_d * sigma_exp, Phimn_d * sigmaPhi_exp

    etamn_new, deltamn_new, Phimn_new = maybe_apply(
        diffflag, _diffuse, (etamn_new, deltamn_new, Phimn_new)
    )

    # Inverse transforms back to Fourier / physical space.
    newetamtstep = st.invrs_leg(etamn_new, I, J, M, N, Pmn)
    newetam_trunc = newetamtstep[:, : (M + 1)]
    newetatstep = st.invrs_fft(newetamtstep, I)

    newdeltamtstep = st.invrs_leg(deltamn_new, I, J, M, N, Pmn)
    newdeltam_trunc = newdeltamtstep[:, : (M + 1)]
    newdeltatstep = st.invrs_fft(newdeltamtstep, I)

    newPhimtstep = st.invrs_leg(Phimn_new, I, J, M, N, Pmn)
    newPhim_trunc = newPhimtstep[:, : (M + 1)]
    newPhitstep = st.invrs_fft(newPhimtstep, I)

    Unew, Vnew, newUm, newVm = st.invrsUV_with_coeffs(
        deltamn_new, etamn_new, fmn, I, J, M, N, Pmn, Hmn, tstepcoeffmn, marray
    )

    return (
        etamn_new,
        newetatstep,
        newetam_trunc,
        deltamn_new,
        newdeltatstep,
        newdeltam_trunc,
        Phimn_new,
        newPhitstep,
        newPhim_trunc,
        Unew,
        Vnew,
        newUm,
        newVm,
    )
