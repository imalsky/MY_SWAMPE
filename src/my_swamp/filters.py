"""
This module contains the functions associated with filters needed for numerical stability.
Matches the original SWAMPE numpy implementation.
"""
from __future__ import annotations

import jax.numpy as jnp
from .dtypes import float_dtype


def diffusion(Ximn: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Applies the diffusion filter described in Gelb and Gleeson (eq. 12).

    Parameters
    ----------
    Ximn : jnp.ndarray
        Spectral coefficients with shape ``(M+1, N+1)``.
    sigma : jnp.ndarray
        Diffusion coefficients with shape ``(M+1, N+1)``.

    Returns
    -------
    jnp.ndarray
        Filtered spectral coefficients with the same shape and dtype as
        ``Ximn``.
    """
    return Ximn * sigma


def sigma(M: int, N: int, K4: float, a: float, dt: float) -> jnp.ndarray:
    """Computes the coefficient for the fourth degree diffusion filter
    described in Gelb and Gleeson (eq. 12) for vorticity and divergence.

    Uses the original implicit filter formulation.

    Note: the maintained driver only wires up the sixth-order filters
    (:func:`sigma6`/:func:`sigma6Phi`); this fourth-order variant is retained
    to mirror the reference SWAMPE ``filters`` API and is otherwise unused.

    Parameters
    ----------
    M : int
    N : int
    K4 : float
    a : float
    dt : float
    
    Returns
    -------
    jnp.ndarray
    """
    nvec = jnp.arange(N + 1, dtype=float_dtype())
    ncoeff = (nvec * nvec / a**2) * ((nvec + 1) * (nvec + 1) / a**2)
    factor1 = 4 / a**4
    factor2 = 2 * dt * K4
    
    sigmacoeff = 1 + factor2 * (ncoeff - factor1)
    sigmas = 1 / sigmacoeff
    
    return jnp.broadcast_to(sigmas[None, :], (M + 1, N + 1))


def sigmaPhi(M: int, N: int, K4: float, a: float, dt: float) -> jnp.ndarray:
    """Computes the coefficient for the fourth degree diffusion filter
    described in Gelb and Gleeson (eq. 12) for geopotential.

    Uses original implicit filter formulation (no factor1 subtraction).

    Note: the maintained driver only wires up the sixth-order filters
    (:func:`sigma6`/:func:`sigma6Phi`); this fourth-order variant is retained
    to mirror the reference SWAMPE ``filters`` API and is otherwise unused.
    
    Parameters
    ----------
    M : int
    N : int
    K4 : float
    a : float
    dt : float
    
    Returns
    -------
    jnp.ndarray
    """
    nvec = jnp.arange(N + 1, dtype=float_dtype())
    ncoeff = (nvec * nvec / a**2) * ((nvec + 1) * (nvec + 1) / a**2)
    factor2 = 2 * dt * K4
    
    sigmacoeff = 1 + factor2 * ncoeff
    sigmas = 1 / sigmacoeff
    
    return jnp.broadcast_to(sigmas[None, :], (M + 1, N + 1))


def sigma6(M: int, N: int, K6: float, a: float, dt: float) -> jnp.ndarray:
    """Computes the coefficient for the sixth degree diffusion filter 
    for vorticity and divergence.
    
    Uses original implicit filter formulation.
    
    Parameters
    ----------
    M : int
    N : int
    K6 : float
    a : float
    dt : float
    
    Returns
    -------
    jnp.ndarray
    """
    nvec = jnp.arange(N + 1, dtype=float_dtype())
    
    # n^3 * (n+1)^3 / a^6
    ncoeff = ((nvec * nvec * nvec) / a**3) * (((nvec + 1) * (nvec + 1) * (nvec + 1)) / a**3)
    factor1 = 8 / a**6
    factor2 = 2 * dt * K6
    
    sigmacoeff = 1 + factor2 * (ncoeff - factor1)
    sigmas = 1 / sigmacoeff
    
    return jnp.broadcast_to(sigmas[None, :], (M + 1, N + 1))


def sigma6Phi(M: int, N: int, K6: float, a: float, dt: float) -> jnp.ndarray:
    """Computes the coefficient for the sixth degree diffusion filter for geopotential.

    Uses original implicit filter formulation (no factor1 subtraction).

    Parameters
    ----------
    M : int
    N : int
    K6 : float
    a : float
    dt : float

    Returns
    -------
    jnp.ndarray
    """
    nvec = jnp.arange(N + 1, dtype=float_dtype())

    ncoeff = ((nvec * nvec * nvec) / a**3) * (((nvec + 1) * (nvec + 1) * (nvec + 1)) / a**3)
    factor2 = 2 * dt * K6

    sigmacoeff = 1 + factor2 * ncoeff
    sigmas = 1 / sigmacoeff

    return jnp.broadcast_to(sigmas[None, :], (M + 1, N + 1))


def sigma6_exponential(M: int, N: int, K6: float, a: float, dt: float) -> jnp.ndarray:
    """Exponential (integrating-factor) sixth-order hyperdiffusion for eta/delta.

    Exact solution of ``d(x_n)/dt = -K6 * [ (n(n+1))^3/a^6 - 8/a^6 ] * x_n`` over a
    leapfrog step (2*dt): ``x_n -> x_n * exp(-2*dt*K6*(ncoeff - 8/a^6))``.

    The ``8/a^6`` offset mirrors :func:`sigma6` (neutral at n=1); unlike the
    implicit form it is clamped at zero so n=0 is left untouched rather than
    amplified. Unconditionally stable for any ``dt``; used by the opt-in
    semi-implicit scheme.

    Parameters
    ----------
    M : int
    N : int
    K6 : float
    a : float
    dt : float

    Returns
    -------
    jnp.ndarray
    """
    nvec = jnp.arange(N + 1, dtype=float_dtype())

    ncoeff = ((nvec * nvec * nvec) / a**3) * (((nvec + 1) * (nvec + 1) * (nvec + 1)) / a**3)
    factor1 = 8 / a**6
    factor2 = 2 * dt * K6

    sigmas = jnp.exp(-factor2 * jnp.maximum(ncoeff - factor1, 0.0))

    return jnp.broadcast_to(sigmas[None, :], (M + 1, N + 1))


def sigma6Phi_exponential(M: int, N: int, K6: float, a: float, dt: float) -> jnp.ndarray:
    """Exponential (integrating-factor) sixth-order hyperdiffusion for geopotential.

    Same as :func:`sigma6_exponential` but with no ``8/a^6`` offset, mirroring
    the :func:`sigma6Phi` convention. Unconditionally stable for any ``dt``.

    Parameters
    ----------
    M : int
    N : int
    K6 : float
    a : float
    dt : float

    Returns
    -------
    jnp.ndarray
    """
    nvec = jnp.arange(N + 1, dtype=float_dtype())

    ncoeff = ((nvec * nvec * nvec) / a**3) * (((nvec + 1) * (nvec + 1) * (nvec + 1)) / a**3)
    factor2 = 2 * dt * K6

    sigmas = jnp.exp(-factor2 * ncoeff)

    return jnp.broadcast_to(sigmas[None, :], (M + 1, N + 1))
