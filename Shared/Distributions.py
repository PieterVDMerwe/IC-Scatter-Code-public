"""Sampling distributions for photons and electrons in the simulation.

This module provides functions to sample photon and electron energies,
temperatures, and direction vectors. It implements pair annihilation
spectrum generation, Planck blackbody distributions (using CUDA), Maxwell-Jüttner
thermal electron distributions, isotropic direction sampling, and monoenergetic
sources.
"""

from Shared.Common import sample_subset, Sum

from numba import cuda, float64
import numba
import numpy as np
import cupy as cp
import math as m
from numba.cuda import libdevice as cudalib
import scipy.constants as scpc
from numba.cuda.random import create_xoroshiro128p_states as cStates
from numba.cuda.random import xoroshiro128p_uniform_float64 as uniform
import random

from pathlib import Path
import pickle

############## Constants ##################
DTYPE = cp.float64
MODULE_DIR = Path(__file__).parent
RAND_SEED_TOPEND = 4294967295
BLOCK_SIZE = 256


######################################################## Pair annihilation ########################################################

def GetPairAnnihilation(N_photons, Theta_r):
    """Samples photon energies from the pair annihilation distribution.

    Loads a pre-calculated energy grid and cumulative distribution function (CDF)
    from disk if available. Otherwise, calculates, caches, and uses the CDF.
    Photon energies are returned in units of m_e * c^2.

    Args:
        N_photons: Number of photon energies to sample.
        Theta_r: Dimensionless temperature parameter.

    Returns:
        cupy.ndarray: Array of sampled photon energies, shape (N_photons,).
    """
    try:
        print("Trying to load Pair annihilation grid")
        with open(str(MODULE_DIR) + "epsi_p_CPU_" + str(Theta_r) + ".dat", 'rb') as f:
            epsi_p_CPU = pickle.load(f)
        with open(str(MODULE_DIR) + "CFD_Pair_CPU_" + str(Theta_r) + ".dat", 'rb') as f:
            CFD_Pair_CPU = pickle.load(f)

        epsi_p = cp.array(epsi_p_CPU)
        CFD_Pair = cp.array(CFD_Pair_CPU)

    except (FileNotFoundError, IOError, pickle.PickleError):
        print("Failed to load Pair annihilation grid")
        print("... re-calculating grid")

        # Energy grid
        epsi_p = cp.arange(1.0e-5, 30.0, 1.0e-6)
        N_interval = len(epsi_p)

        # Initialize arrays
        spectrum = cp.zeros(N_interval)
        Trapezoids = cp.zeros(N_interval - 1)
        Integrals = cp.zeros(N_interval - 1)
        CFD_Pair = cp.zeros(N_interval - 1)

        # Calculate spectrum: exp(-(C1 * epsi_p^2) / Theta_r^2) / epsi_p
        C1 = 0.045
        spectrum = cp.exp(-(C1 * epsi_p**2) / (Theta_r**2)) / epsi_p

        # Calculate trapezoids
        Trapezoids = 0.5 * 1.0e-6 * (spectrum[:-1] + spectrum[1:])

        # Calculate cumulative integrals
        Integrals = cp.cumsum(Trapezoids)

        # Finalize integrals: normalize by the last integral value
        CFD_Pair = Integrals / Integrals[-1]

        with open(str(MODULE_DIR) + "epsi_p_CPU_" + str(Theta_r) + ".dat", 'wb') as f:
            pickle.dump(epsi_p, f)

        with open(str(MODULE_DIR) + "CFD_Pair_CPU_" + str(Theta_r) + ".dat", 'wb') as f:
            pickle.dump(CFD_Pair, f)

    finally:
        random_number = cp.random.random(size=N_photons, dtype=DTYPE)

        k = cp.searchsorted(CFD_Pair, random_number, side='right') - 1
        k = cp.clip(k, 0, len(CFD_Pair) - 2)
        u = (random_number - CFD_Pair[k]) / (CFD_Pair[k + 1] - CFD_Pair[k])
        photon_Energy = epsi_p[k] + (epsi_p[k + 1] - epsi_p[k]) * u

        return photon_Energy.astype(DTYPE)


def GetPairAnnihilationRejectionMethod(N_photons, Theta_r, a=1.0e-7, b=30.0):
    """Samples photon energies from the pair annihilation distribution using rejection sampling.

    Args:
        N_photons: Number of photon energies to draw.
        Theta_r: Dimensionless temperature parameter.
        a: Minimum energy (epsilon) range boundary. Defaults to 1e-7.
        b: Maximum energy (epsilon) range boundary. Defaults to 30.0.

    Returns:
        cupy.ndarray: Sampled photon energies in units of m_e * c^2.
    """
    C1 = 0.045
    C = C1 / (Theta_r**2)

    # Allocate output
    photon_Energy = cp.empty(N_photons, dtype=DTYPE)

    i = 0
    batch_factor = 4
    while i < N_photons:
        # propose more to reduce rejections
        m_size = (N_photons - i) * batch_factor

        # proposals ~ 1/eps distribution
        U = cp.random.rand(m_size, dtype=DTYPE)
        eps_prop = a * (b / a) ** U

        # accept with prob exp(-C * eps^2)
        V = cp.random.rand(m_size, dtype=DTYPE)
        accept = V <= cp.exp(-C * eps_prop**2)

        accepted = eps_prop[accept]
        take = min(len(accepted), N_photons - i)
        if take > 0:
            photon_Energy[i:i + take] = accepted[:take]
            i += take

    return photon_Energy.astype(DTYPE)


################################################################### Planck #######################################################################

@cuda.jit(fastmath=False)
def GeneratePlanckEnergy(photon_Energy, states, Theta_r, SIZE):
    """CUDA kernel to generate photon energies according to a Planck blackbody distribution.

    Args:
        photon_Energy: Output device array for generated photon energies.
        states: XOROSHIRO128+ random states.
        Theta_r: Input device array of dimensionless temperature values.
        SIZE: Total number of photons (length of the arrays).
    """
    i = cuda.grid(1)
    if i < SIZE:
        sigma1 = uniform(states, i)
        sigma2 = uniform(states, i)
        sigma3 = uniform(states, i)
        photon_Energy[i] = - ((Theta_r[i]) / (GenerateAlpha(states, SIZE))) * m.log(sigma1 * sigma2 * sigma3)


@cuda.jit(device=True, fastmath=False)
def GenerateAlpha(states, SIZE):
    """Helper device function to generate the scaling factor alpha for the Planck sampler.

    Args:
        states: XOROSHIRO128+ random states.
        SIZE: Size limit check parameter.

    Returns:
        float: Computed alpha scaling factor.
    """
    i = cuda.grid(1)
    if i < SIZE:
        alpha = 0.0
        while alpha == 0.0:
            sigma1 = 1.202 * uniform(states, i)
            if sigma1 < 1.0:
                alpha = 1.0
            else:
                m_idx = 1
                checkHigher = True
                while checkHigher:
                    m_idx = 1 + m_idx
                    if (Sum(m_idx - 1) <= sigma1) and (sigma1 < Sum(m_idx)):
                        alpha = float(m_idx)
                        checkHigher = False
                    if m_idx > 100000:
                        checkHigher = False
        return alpha


def GetPlanck(N_photons, Theta_r):
    """Generates blackbody photon energies from a Planck distribution using CUDA.

    Args:
        N_photons: Number of photons to generate.
        Theta_r: A CuPy array of dimensionless temperatures in units of m_e * c^2.

    Returns:
        cupy.ndarray: Generated photon energies.
    """
    rng_states = cStates(N_photons, seed=random.randint(0, RAND_SEED_TOPEND))
    photon_Energy = cp.zeros(N_photons, dtype=DTYPE)
    grid_size = (N_photons + BLOCK_SIZE - 1) // BLOCK_SIZE
    GeneratePlanckEnergy[grid_size, BLOCK_SIZE](photon_Energy, rng_states, Theta_r, N_photons)

    return photon_Energy.astype(DTYPE)


################################################## Maxwell_Juttner #########################################################################

def CalcGammaLow(Theta_e):
    """Helper method to sample Maxwell-Jüttner momentum for low electron temperatures (Theta_e <= 0.29).

    Uses the Sobol method for lower temperatures.

    Args:
        Theta_e: CuPy array of electron dimensionless temperatures.

    Returns:
        cupy.ndarray: Sampled electron momenta normalized by m_e * c.
    """
    N = Theta_e.shape[0]
    accepted = cp.zeros(N, dtype=cp.bool_)
    eta = cp.zeros(N, dtype=DTYPE)

    while not cp.all(accepted):
        idxs = cp.where(~accepted)[0]
        n = Theta_e[idxs]

        sigma1 = cp.random.random(size=idxs.shape[0])
        sigma2 = cp.random.random(size=idxs.shape[0])
        xsiPrime = -(3.0 / 2.0) * cp.log(sigma1)

        accept = sigma2**2 < 0.151 * ((1.0 + n * xsiPrime)**2) * xsiPrime * (2.0 + n * xsiPrime) * sigma1

        # Compute eta where accepted
        new_eta = cp.sqrt(n * xsiPrime * (2.0 + n * xsiPrime))
        eta[idxs[accept]] = new_eta[accept]
        accepted[idxs[accept]] = True

    return eta.astype(DTYPE)


def CalcGammaHigh(Theta_e):
    """Helper method to sample Maxwell-Jüttner momentum for high temperatures (Theta_e > 0.29).

    Uses the Pozdnyakov method for higher temperatures.

    Args:
        Theta_e: CuPy array of electron dimensionless temperatures.

    Returns:
        cupy.ndarray: Sampled electron momenta normalized by m_e * c.
    """
    N = Theta_e.shape[0]
    accepted = cp.zeros(N, dtype=cp.bool_)
    eta = cp.zeros(N, dtype=DTYPE)

    while not cp.all(accepted):
        idxs = cp.where(~accepted)[0]
        n = Theta_e[idxs]

        sigma1 = cp.random.random(size=idxs.shape[0])
        sigma2 = cp.random.random(size=idxs.shape[0])
        sigma3 = cp.random.random(size=idxs.shape[0])
        sigma4 = cp.random.random(size=idxs.shape[0])

        eta1 = -n * cp.log(sigma1 * sigma2 * sigma3)
        eta2 = -n * cp.log(sigma1 * sigma2 * sigma3 * sigma4)

        accept = (eta2**2 - eta1**2) > 1.0
        eta[idxs[accept]] = eta1[accept]
        accepted[idxs[accept]] = True

    return eta.astype(DTYPE)


def GetElectronGamma(Theta_e):
    """Generates electron Lorentz factors (gamma) from a Maxwell-Jüttner distribution.

    Args:
        Theta_e: CuPy array of dimensionless electron temperatures (k_B * T / m_e * c^2).

    Returns:
        cupy.ndarray: Sampled electron Lorentz factors (gamma).
    """
    eta = cp.zeros_like(Theta_e, dtype=DTYPE)
    low_mask = Theta_e <= 0.29
    high_mask = ~low_mask

    if cp.any(low_mask):
        eta[low_mask] = CalcGammaLow(Theta_e[low_mask])
    if cp.any(high_mask):
        eta[high_mask] = CalcGammaHigh(Theta_e[high_mask])

    return cp.sqrt(1.0 + eta**2, dtype=DTYPE)


###################################################### Isotropic Direction #######################################################

def GetIsotropicDirection(N, opening_angle=cp.pi, uniform_theta=False):
    """Generates isotropic 3D unit direction vectors.

    Args:
        N: Number of direction vectors to generate.
        opening_angle: Limit the generated direction to a cone of this opening angle.
            Defaults to cp.pi (full sphere).
        uniform_theta: If True, samples theta uniformly instead of cosmically (sin(theta)).
            Defaults to False.

    Returns:
        cupy.ndarray: Array of shape (N, 3) containing 3D unit vectors.
    """
    sigma_theta = cp.random.random(size=N, dtype=DTYPE)
    if uniform_theta:
        theta = sigma_theta * opening_angle
    else:
        theta = cp.arccos(1.0 - sigma_theta * (1.0 - cp.cos(opening_angle)))
    phi = 2.0 * cp.pi * cp.random.random(size=N)
    return cp.concatenate([(cp.sin(theta) * cp.cos(phi))[:, None],
                           (cp.sin(theta) * cp.sin(phi))[:, None],
                           cp.cos(theta)[:, None]], axis=1, dtype=DTYPE)


################################################### Monoenergetic ##################################################################

def GetMonoenergeticPhotons(N_photons, photon_energy):
    """Creates a CuPy array filled with a single photon energy value.

    Args:
        N_photons: Number of photons to generate.
        photon_energy: The monoenergetic energy value.

    Returns:
        cupy.ndarray: Array filled with photon_energy of shape (N_photons,).
    """
    return cp.full(N_photons, photon_energy, dtype=DTYPE)


def GetMonoenergeticElectrons(N_photons, lorentz_factor):
    """Creates a CuPy array filled with a single electron Lorentz factor (gamma) value.

    Args:
        N_photons: Number of electrons to generate.
        lorentz_factor: The monoenergetic electron Lorentz factor.

    Returns:
        cupy.ndarray: Array filled with lorentz_factor of shape (N_photons,).
    """
    return cp.full(N_photons, lorentz_factor, dtype=DTYPE)
