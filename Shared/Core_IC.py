"""Core Inverse Compton (IC) and Compton scattering physics calculations.

This module implements the primary physical interactions of Compton and
Inverse Compton scattering using GPU-accelerated computing via CuPy and Numba.
It includes cross-section calculations, scattering angle and polarization updates,
Lorentz boosts for polarization vectors, and calculation of Stokes parameters.
"""

import Shared.Common as SC

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
FORWARD = False
BACKWARD = True


################################## IC Scattering Functions ########################################################

def CalcPhotonKNCrossection(photon_energies):
    """Calculates the Klein-Nishina cross-section for given photon energies.

    Energies must be in naturalized units (m_e * c^2).

    Args:
        photon_energies: CuPy array of photon energies.

    Returns:
        cupy.ndarray: Klein-Nishina cross-sections normalized by the Thomson cross-section.
    """
    return (3.0 / 4.0) * (
        ((1.0 + photon_energies) / (photon_energies**3)) * (
            (2.0 * photon_energies * (1.0 + photon_energies)) / (1.0 + 2.0 * photon_energies)
            - cp.log(1.0 + 2.0 * photon_energies)
        )
        + ((cp.log(1.0 + 2.0 * photon_energies)) / (2.0 * photon_energies))
        - (1.0 + 3.0 * photon_energies) / (1.0 + 2.0 * photon_energies)**2
    )


@cuda.jit(device=True, fastmath=False)
def ScatteringAngle(photon_energy, mu):
    """Device function evaluating the integral of the differential KN cross-section.

    Evaluated up to cosine of the polar scattering angle (mu). Used for CDF matching.

    Args:
        photon_energy: Photon energy in electron rest mass units.
        mu: Cosine of the polar scattering angle.

    Returns:
        float: Evaluated CDF component at mu.
    """
    lower = 2.0 / 3.0 + 2.0 * photon_energy + m.log(1.0 + 2.0 * photon_energy) / photon_energy
    upper = (photon_energy * (3.0 / 2.0 + mu * (1.0 - mu / 2.0))
             + (1.0 + mu**3) / 3.0
             - (m.log(1.0 + photon_energy * (1.0 - mu)) - m.log(1.0 + 2.0 * photon_energy)) / photon_energy)
    return (upper / lower)


@cuda.jit(fastmath=False)
def CalcScatMu(photon_energies, xi_theta_sc, mu):
    """CUDA kernel to calculate the cosine of the scattering angle (mu) via binary search.

    Args:
        photon_energies: Device array of incoming photon energies.
        xi_theta_sc: Device array of random numbers [0, 1) for CDF inversion.
        mu: Output device array to store the computed scattering cosines.
    """
    i = cuda.grid(1)
    if i >= len(photon_energies):
        return
    else:
        mu_low = -1.0
        mu_high = 1.0
        mu_mid = 0.0

        # Binary search
        zlz = 0
        Not_Found = True
        while (zlz < 500) and (Not_Found):  # 50 iterations is plenty
            mu_mid = 0.5 * (mu_low + mu_high)
            cdf_mid = ScatteringAngle(photon_energies[i], mu_mid)

            if (mu_high - mu_low) < 1e-9:
                Not_Found = False
            elif cdf_mid > xi_theta_sc[i]:
                mu_high = mu_mid
            else:
                mu_low = mu_mid
            zlz = zlz + 1

        mu[i] = mu_mid


def CalcScEnergies(photon_energies, scattered_mu):
    """Calculates scattered photon energies using Compton formula.

    Args:
        photon_energies: CuPy array of incoming photon energies.
        scattered_mu: CuPy array of cosines of polar scattering angles.

    Returns:
        cupy.ndarray: Scattered photon energies.
    """
    return photon_energies / (1.0 + photon_energies * (1.0 - scattered_mu))


@cuda.jit(device=True, fastmath=False)
def PhiSc(pNu, scattered_mu, pNuSc, phiSc):
    """Device function evaluating the CDF of the azimuthal scattering angle (phi).

    Args:
        pNu: Incoming photon energy.
        scattered_mu: Cosine of the polar scattering angle.
        pNuSc: Scattered photon energy.
        phiSc: Trial azimuthal scattering angle.

    Returns:
        float: Evaluated CDF component at phiSc.
    """
    y = 1.0 - scattered_mu**2
    epe = pNuSc / pNu
    eep = pNu / pNuSc
    return phiSc - (y * cudalib.sin(phiSc) * cudalib.cos(phiSc)) / (epe + eep - y)


@cuda.jit(fastmath=False)
def CalcScPhi(xi_phi, photon_energies, scattered_mu, scattered_photon_energies, phiSc):
    """CUDA kernel to calculate azimuthal scattering angle (phi) via binary search.

    Args:
        xi_phi: Device array of random numbers [0, 1) for CDF inversion.
        photon_energies: Device array of incoming photon energies.
        scattered_mu: Device array of cosines of polar scattering angles.
        scattered_photon_energies: Device array of scattered photon energies.
        phiSc: Output device array to store computed azimuthal scattering angles.
    """
    i = cuda.grid(1)
    if i >= len(photon_energies):
        return
    else:
        target = 2.0 * m.pi * xi_phi[i]  # The random target value

        # Binary search bounds
        phi_low = 0.0
        phi_high = 2.0 * m.pi
        phi_mid = 0.0

        # Binary search loop
        klk = 0
        Not_Found = True
        while (klk < 500) and (Not_Found):  # up to 50 iterations
            phi_mid = 0.5 * (phi_low + phi_high)
            phi_val = PhiSc(photon_energies[i], scattered_mu[i], scattered_photon_energies[i], phi_mid)

            if (phi_high - phi_low) < 1e-9:
                Not_Found = False
            elif phi_val > target:
                phi_high = phi_mid
            else:
                phi_low = phi_mid

        phiSc[i] = phi_mid


def CalcScDir(thetaSc, phiSc, pDirec, pPolarization):
    """Calculates the scattered photon direction 4-vector in the scattering frame.

    Args:
        thetaSc: CuPy array of polar scattering angles.
        phiSc: CuPy array of azimuthal scattering angles.
        pDirec: CuPy array of incoming photon direction 4-vectors.
        pPolarization: CuPy array of incoming polarization 4-vectors.

    Returns:
        cupy.ndarray: Scattered photon direction 4-vectors.
    """
    pDirecSc = cp.empty(pDirec.shape, dtype=DTYPE)
    ctSc = cp.cos(thetaSc)
    stcpSc = cp.sin(thetaSc) * cp.cos(phiSc)
    stspSc = cp.sin(thetaSc) * cp.sin(phiSc)

    pDirecSc[:, 1] = pDirec[:, 1] * ctSc + pPolarization[:, 1] * stcpSc + stspSc * (pDirec[:, 2] * pPolarization[:, 3] - pDirec[:, 3] * pPolarization[:, 2])
    pDirecSc[:, 2] = pDirec[:, 2] * ctSc + pPolarization[:, 2] * stcpSc + stspSc * (pDirec[:, 3] * pPolarization[:, 1] - pDirec[:, 1] * pPolarization[:, 3])
    pDirecSc[:, 3] = pDirec[:, 3] * ctSc + pPolarization[:, 3] * stcpSc + stspSc * (pDirec[:, 1] * pPolarization[:, 2] - pDirec[:, 2] * pPolarization[:, 1])
    pDirecSc[:, 0] = 1.0

    pDirecSc[:, 1:] = pDirecSc[:, 1:] / cp.sqrt(pDirecSc[:, 1]**2 + pDirecSc[:, 2]**2 + pDirecSc[:, 3]**2)[:, None]

    return pDirecSc


def CalcPD(pNu, thetaSc, pNuSc, phiSc):
    """Computes the polarization probability parameter (PD) for acceptance-rejection.

    Args:
        pNu: CuPy array of incoming photon energies.
        thetaSc: CuPy array of polar scattering angles.
        pNuSc: CuPy array of scattered photon energies.
        phiSc: CuPy array of azimuthal scattering angles.

    Returns:
        cupy.ndarray: Acceptance probability array.
    """
    temp = (cp.sin(thetaSc) ** 2) * (cp.cos(phiSc) ** 2)
    epe = pNuSc / pNu
    eep = pNu / pNuSc
    denominator = epe + eep - 2.0 * temp
    return 2.0 * ((1.0 - temp) / denominator)


def CalcPolarization(pDirecSc, thetaSc, phiSc, pPolarization):
    """Computes the updated polarization vector for accepted scattered photons.

    Args:
        pDirecSc: CuPy array of scattered photon directions.
        thetaSc: CuPy array of polar scattering angles.
        phiSc: CuPy array of azimuthal scattering angles.
        pPolarization: CuPy array of incoming polarizations.

    Returns:
        cupy.ndarray: Updated polarization 4-vectors.
    """
    sin_theta_cos_phi = cp.sin(thetaSc) * cp.cos(phiSc)

    pPolarizationSc = cp.zeros(pPolarization.shape, dtype=DTYPE)

    pPolarizationSc[:, 1] = pDirecSc[:, 1] * sin_theta_cos_phi - pPolarization[:, 1]
    pPolarizationSc[:, 2] = pDirecSc[:, 2] * sin_theta_cos_phi - pPolarization[:, 2]
    pPolarizationSc[:, 3] = pDirecSc[:, 3] * sin_theta_cos_phi - pPolarization[:, 3]

    pPolarizationSc[:, 1:] = pPolarizationSc[:, 1:] / cp.sqrt(pPolarizationSc[:, 1]**2 + pPolarizationSc[:, 2]**2 + pPolarizationSc[:, 3]**2)[:, None]

    return pPolarizationSc


def CalcRandomPolarization(direc):
    """Generates a random polarization vector perpendicular to the photon direction.

    Used when a photon's polarization state is re-randomized on scattering.

    Args:
        direc: CuPy array of shape (N, 4) containing photon direction vectors.

    Returns:
        cupy.ndarray: Random polarization 4-vectors of shape (N, 4).
    """
    N = direc.shape[0]
    pol = cp.zeros((N, 4), dtype=cp.float64)

    # Sample random alpha_r values
    alpha_r = 2.0 * cp.pi * cp.random.random(size=N)

    # Extract directional components
    dx = direc[:, 1]
    dy = direc[:, 2]
    dz = direc[:, 3]

    # Avoid divide-by-zero
    denom = cp.sqrt(1.0 - dz**2 + 1e-20)
    QCoeff = 1.0 / denom

    # Compute Qp basis vector
    Qp_x = -QCoeff * dy
    Qp_y = QCoeff * dx
    Qp_z = cp.zeros_like(dz)

    # Compute Qm basis vector
    Qm_x = -QCoeff * dx * dz
    Qm_y = -QCoeff * dy * dz
    Qm_z = QCoeff * (1.0 - dz**2)

    # Combine with rotation
    sin_alpha = cp.sin(alpha_r)
    cos_alpha = cp.cos(alpha_r)

    pol[:, 1] = sin_alpha * Qm_x + cos_alpha * Qp_x
    pol[:, 2] = sin_alpha * Qm_y + cos_alpha * Qp_y
    pol[:, 3] = sin_alpha * Qm_z + cos_alpha * Qp_z

    # Special-case: if direction is ~parallel to z-axis, pick xy-plane polarization
    mask = cp.abs(dz) > 1.0 - 1e-12
    if cp.any(mask):
        pol[mask, 1] = cp.cos(alpha_r[mask])
        pol[mask, 2] = cp.sin(alpha_r[mask])
        pol[mask, 3] = 0.0

    return pol


def CalculateComptonScattering(photon_energies, direction, polarization, send_theta_sc=False):
    """Simulates Compton scattering of polarized photons.

    Updates photon energy, direction, and polarization vectors on scattering.

    Args:
        photon_energies: CuPy array of incoming photon energies.
        direction: CuPy array of incoming direction 4-vectors.
        polarization: CuPy array of incoming polarization 4-vectors.
        send_theta_sc: If True, returns polar scattering angle as well.
            Defaults to False.

    Returns:
        tuple: (scattered_energies, scattered_direction, scattered_polarization)
            or (scattered_energies, scattered_direction, scattered_polarization, theta_scatter) if send_theta_sc.
    """
    xi_theta = cp.random.random(photon_energies.size, dtype=DTYPE)
    mu_scatter = cp.empty(photon_energies.size, dtype=DTYPE)

    N = len(photon_energies)
    GRID_SIZE = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Compute scattering cosine
    CalcScatMu[GRID_SIZE, BLOCK_SIZE](photon_energies, xi_theta, mu_scatter)

    # Compute scattered energy
    scattered_photon_energies = CalcScEnergies(photon_energies, mu_scatter)

    # Compute scattered phi
    phi_scatter = cp.empty(photon_energies.size, dtype=DTYPE)
    xi_phi = cp.random.random(photon_energies.size, dtype=DTYPE)
    CalcScPhi[GRID_SIZE, BLOCK_SIZE](xi_phi, photon_energies, mu_scatter, scattered_photon_energies, phi_scatter)

    theta_scatter = cp.arccos(mu_scatter)

    # Compute new direction
    scattered_direction = CalcScDir(theta_scatter, phi_scatter, direction, polarization)

    # Determine polarization changes via acceptance-rejection
    PD = CalcPD(photon_energies, theta_scatter, scattered_photon_energies, phi_scatter)
    xi_PD = cp.random.random(photon_energies.size, dtype=DTYPE)

    accepted = xi_PD < PD

    scattered_polarization = cp.empty(polarization.shape, dtype=DTYPE)

    # Calculate polarization for accepted, randomize for rejected
    scattered_polarization[accepted] = CalcPolarization(scattered_direction[accepted], theta_scatter[accepted], phi_scatter[accepted], polarization[accepted])
    scattered_polarization[~accepted] = CalcRandomPolarization(scattered_direction[~accepted])
    scattered_polarization[:, 0] = 0.0

    if send_theta_sc:
        return scattered_photon_energies, scattered_direction, scattered_polarization, theta_scatter
    else:
        return scattered_photon_energies, scattered_direction, scattered_polarization


def LorentzTransformPolarizationVector(photon_direction, polarization_vector, new_frame, Inverse=False):
    """Applies a Lorentz transform to a polarization 4-vector.

    Maintains the gauge condition (perpendicularity to the photon direction).

    Args:
        photon_direction: CuPy array of photon direction 4-vectors.
        polarization_vector: CuPy array of polarization 4-vectors.
        new_frame: CuPy array specifying the boost frame parameters.
        Inverse: If True, performs the inverse boost. Defaults to False.

    Returns:
        cupy.ndarray: Lorentz-boosted polarization vectors.
    """
    temp_polarization = SC.GetLorentzTransform(polarization_vector, new_frame, Inverse)
    new_polarization = cp.empty(temp_polarization.shape, dtype=DTYPE)
    for k in [1, 2, 3]:
        new_polarization[:, k] = temp_polarization[:, k] - temp_polarization[:, 0] * photon_direction[:, k] / photon_direction[:, 0]
    new_polarization = SC.NormalizeFourVector(new_polarization, 0.0)
    return new_polarization


def CalcICScattering(photon_energies, photon_wave_vectors, polarization_vector, electron_frame, send_theta_sc=False):
    """Calculates Inverse Compton (IC) scattering in the electron rest frame.

    Boosts incoming photons to the electron rest frame, computes the Compton
    scattering, and boosts the resulting scattered photons back to the laboratory frame.

    Args:
        photon_energies: CuPy array of lab frame photon energies.
        photon_wave_vectors: CuPy array of lab frame photon direction 4-vectors.
        polarization_vector: CuPy array of lab frame polarization 4-vectors.
        electron_frame: CuPy array of electron boost frames (gammas, directions).
        send_theta_sc: If True, returns polar scattering angle. Defaults to False.

    Returns:
        tuple: (lab_scattered_energies, lab_scattered_directions, lab_scattered_polarizations)
            or (..., theta_scatter) if send_theta_sc.
    """
    electron_frame_wave_vectors = SC.GetLorentzTransform(photon_wave_vectors, electron_frame)
    electron_frame_photon_energies = photon_energies * electron_frame_wave_vectors[:, 0]
    electron_frame_polarization_vector = LorentzTransformPolarizationVector(electron_frame_wave_vectors, polarization_vector, electron_frame)
    electron_frame_wave_vectors = SC.NormalizeFourVector(electron_frame_wave_vectors)

    if send_theta_sc:
        scattered_photon_energies, scattered_direction, scattered_polarization, theta_scatter = CalculateComptonScattering(electron_frame_photon_energies, electron_frame_wave_vectors, electron_frame_polarization_vector, send_theta_sc=send_theta_sc)
    else:
        scattered_photon_energies, scattered_direction, scattered_polarization = CalculateComptonScattering(electron_frame_photon_energies, electron_frame_wave_vectors, electron_frame_polarization_vector)

    emission_frame_wave_vectors = SC.GetLorentzTransform(scattered_direction, electron_frame, BACKWARD)
    emission_frame_photon_energies = scattered_photon_energies * emission_frame_wave_vectors[:, 0]
    emission_frame_polarization_vector = LorentzTransformPolarizationVector(emission_frame_wave_vectors, scattered_polarization, electron_frame, BACKWARD)
    emission_frame_wave_vectors = SC.NormalizeFourVector(emission_frame_wave_vectors)

    if send_theta_sc:
        return emission_frame_photon_energies, emission_frame_wave_vectors, emission_frame_polarization_vector, theta_scatter
    else:
        return emission_frame_photon_energies, emission_frame_wave_vectors, emission_frame_polarization_vector


def CalcStokes(pDirec, pPolarization):
    """Calculates the linear Stokes parameters (Q, U) for given photon directions and polarizations.

    Args:
        pDirec: CuPy array of photon directions, shape (N, 4).
        pPolarization: CuPy array of photon polarizations, shape (N, 4).

    Returns:
        tuple[cupy.ndarray, cupy.ndarray]: A tuple containing:
            - Q: The Q Stokes parameter array, shape (N,).
            - U: The U Stokes parameter array, shape (N,).
    """
    d = pDirec[:, 1:4]
    p = pPolarization[:, 1:4]

    d1, d2, d3 = d[:, 0], d[:, 1], d[:, 2]
    one_minus_d3sq = 1.0 - d3**2

    # mask photons close to z-axis
    eps = 1e-12
    mask = cp.abs(one_minus_d3sq) < eps

    # allocate outputs
    Q = cp.zeros(d1.shape, dtype=cp.float64)
    U = cp.zeros(d1.shape, dtype=cp.float64)

    # --- general case ---
    if cp.any(~mask):
        QCoeff = 1.0 / cp.sqrt(one_minus_d3sq[~mask])
        UCoeff = 1.0 / cp.sqrt(2.0 * one_minus_d3sq[~mask])

        Qp = cp.stack([-QCoeff * d2[~mask],
                       QCoeff * d1[~mask],
                       cp.zeros_like(d1[~mask])], axis=1)

        Qm = cp.stack([-QCoeff * d1[~mask] * d3[~mask],
                       -QCoeff * d2[~mask] * d3[~mask],
                        QCoeff * (1.0 - d3[~mask]**2)], axis=1)

        Up = cp.stack([-UCoeff * (d2[~mask] + d1[~mask] * d3[~mask]),
                        UCoeff * (d1[~mask] - d2[~mask] * d3[~mask]),
                        UCoeff * (1.0 - d3[~mask]**2)], axis=1)

        Um = cp.stack([UCoeff * (d2[~mask] - d1[~mask] * d3[~mask]),
                       -UCoeff * (d1[~mask] + d2[~mask] * d3[~mask]),
                        UCoeff * (1.0 - d3[~mask]**2)], axis=1)

        dot_Qp = cp.sum(Qp * p[~mask], axis=1)
        dot_Qm = cp.sum(Qm * p[~mask], axis=1)
        dot_Up = cp.sum(Up * p[~mask], axis=1)
        dot_Um = cp.sum(Um * p[~mask], axis=1)

        Q[~mask] = dot_Qp**2 - dot_Qm**2
        U[~mask] = dot_Up**2 - dot_Um**2

    # --- z-axis special case ---
    if cp.any(mask):
        # pick x and y as polarization basis
        px, py = p[mask, 0], p[mask, 1]
        Q[mask] = px**2 - py**2
        U[mask] = 2.0 * px * py

    return Q, U
