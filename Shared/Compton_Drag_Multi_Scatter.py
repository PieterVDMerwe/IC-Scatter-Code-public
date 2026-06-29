"""Simulation of multi-scattering Compton Drag effects on relativistic jet shells.

This module simulates multi-scattering events of radiation field photons interacting
with target electrons within a relativistic jet shell. It models the spatial evolution,
Lorentz boost dynamics, multiple scattering propagation limits, and polarization/Stokes parameters.
"""

from Shared import Distributions as SD
from Shared import Core_IC as IC
from Shared import Common as SC

import cupy as cp
import pickle
import numpy as np
import scipy.constants as scpc
import matplotlib.pyplot as plt

############################################################## CONST ##########################################################################

DTYPE = cp.float64
JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e * scpc.c**2
NATURAL_TO_KEV = M_EC2 * JOULE_TO_KEV

PARAMETER_B = 0.5
PARAMETER_G = 2.0
PARAMETER_A = 7.56e-16  # in SI

MAX_ITERATIONS = 100

FORWARD = False
BACKWARD = True


def GetMeanFreePath(COMOVING_DENSITY_COEFF, BL, current_radius, mu_lab, cross_section):
    """Calculates the mean free path of a photon under multi-scatter Compton drag.

    Args:
        COMOVING_DENSITY_COEFF: Target density coefficient of target electrons.
        BL: Relativistic bulk Lorentz factor.
        current_radius: Radial distance of the photon.
        mu_lab: Cosine of propagation angle in lab frame.
        cross_section: Klein-Nishina cross-section.

    Returns:
        cupy.ndarray: Sampled mean free paths.
    """
    return (-cp.log(1.0 - cp.random.random(size=current_radius.size, dtype=DTYPE))) / (
        (COMOVING_DENSITY_COEFF / (BL * current_radius**2)) * cross_section * (BL - cp.sqrt(BL**2 - 1.0) * mu_lab)
    )


def GetInitialRadius(Z_0, Z_MAX, N_PHOTONS):
    """Generates initial starting radii for the photons in the jet.

    Args:
        Z_0: Minimum initial radius boundary.
        Z_MAX: Maximum initial radius boundary.
        N_PHOTONS: Number of photons.

    Returns:
        cupy.ndarray: Sampled initial radii.
    """
    return Z_0 * (Z_MAX / Z_0)**cp.random.random(N_PHOTONS, dtype=DTYPE)


def GetT(initial_radius, Z_STAR, initial_T):
    """Returns local temperature (T) at a given radius in units of m_e * c^2.

    Args:
        initial_radius: Photon radii.
        Z_STAR: Reference radius parameter.
        initial_T: Base temperature at Z_STAR.

    Returns:
        cupy.ndarray: Computed local temperatures.
    """
    return initial_T * (initial_radius / Z_STAR)**(-PARAMETER_B)


def GetT0(Z_0, Z_STAR, T_STAR):
    """Calculates temperature at Z_0 based on reference Z_STAR temperature.

    Args:
        Z_0: Base radius.
        Z_STAR: Reference radius.
        T_STAR: Temperature at Z_STAR.

    Returns:
        float: Temperature at Z_0.
    """
    return T_STAR * (Z_0 / Z_STAR)**(-PARAMETER_B)


def GetGammaBelowZStar(initial_radius, BL_0, THETA_J, T_0, Z_0, E_F):
    """Calculates the Lorentz factor at a radius below Z_STAR.

    Args:
        initial_radius: Radial positions.
        BL_0: Initial bulk Lorentz factor.
        THETA_J: Opening semi-angle.
        T_0: Temperature at Z_0.
        Z_0: Base radius.
        E_F: Energy flux factor.

    Returns:
        cupy.ndarray: Computed Lorentz factors.
    """
    return BL_0 / (1.0 + 2.0 * cp.pi * (THETA_J**2) * PARAMETER_A * (T_0**4) * (BL_0**2) * (Z_0**2) * (initial_radius - Z_0) / E_F)


def GetGammaBelowZT(initial_radius, BL_0, BL_STAR, THETA_J, T_STAR, Z_STAR, E_F):
    """Calculates the Lorentz factor at a radius below Z_T but above Z_STAR.

    Args:
        initial_radius: Radial positions.
        BL_0: Initial bulk Lorentz factor.
        BL_STAR: Lorentz factor at Z_STAR.
        THETA_J: Opening semi-angle.
        T_STAR: Temperature at Z_STAR.
        Z_STAR: Reference radius.
        E_F: Energy flux factor.

    Returns:
        cupy.ndarray: Computed Lorentz factors.
    """
    return BL_STAR / (1.0 + 2.0 * cp.pi * (THETA_J**2) * PARAMETER_A * (T_STAR**4) * (BL_0 * BL_STAR) * (Z_STAR**2) * (initial_radius - Z_STAR) / E_F)


def GetGamma(initial_radius, BL_0, THETA_J, Theta_star, Z_0, Z_STAR, Z_T, E_F):
    """Calculates bulk Lorentz factors for given initial radii.

    Determines which radial regime the photon belongs to and applies the
    respective analytic formula.

    Args:
        initial_radius: Radial positions.
        BL_0: Initial bulk Lorentz factor.
        THETA_J: Opening semi-angle.
        Theta_star: Temperature parameter at Z_STAR.
        Z_0: Base radius.
        Z_STAR: Reference radius.
        Z_T: Termination radius.
        E_F: Energy flux parameter.

    Returns:
        cupy.ndarray: Local shell Lorentz factors.
    """
    gamma = cp.empty(initial_radius.size, dtype=DTYPE)
    T_STAR = Theta_star * M_EC2 / scpc.k
    T_0 = GetT0(Z_0, Z_STAR, T_STAR)
    BL_STAR = GetGammaBelowZStar(Z_STAR, BL_0, THETA_J, T_0, Z_0, E_F)
    below_Z_STAR = initial_radius <= Z_STAR
    gamma[below_Z_STAR] = GetGammaBelowZStar(initial_radius[below_Z_STAR], BL_0, THETA_J, T_0, Z_0, E_F)
    below_Z_T = cp.logical_and(~below_Z_STAR, initial_radius < Z_T)
    gamma[below_Z_T] = GetGammaBelowZT(initial_radius[below_Z_T], BL_0, BL_STAR, THETA_J, T_STAR, Z_STAR, E_F)
    above_Z_T = initial_radius >= Z_T
    gamma[above_Z_T] = GetGammaBelowZT(Z_T, BL_0, BL_STAR, THETA_J, T_STAR, Z_STAR, E_F)
    return gamma


def GetTimeIntegral(BL_A, B, u_A, u_i):
    """Analytical evaluation of the time integral for Lorentz factor evolution.

    Args:
        BL_A: Reference bulk Lorentz factor.
        B: Evolution coefficient parameter.
        u_A: Upper limit boundary value.
        u_i: Lower limit integration variable.

    Returns:
        cupy.ndarray: Integral values.
    """
    return (BL_A / B) * (cp.arcsin(u_i / BL_A) - cp.arcsin(u_A / BL_A))


def GetInitialTime(initial_radius, gamma, BL_0, THETA_J, Theta_star, Z_0, Z_STAR, Z_T, E_F):
    """Calculates emission/initial lab-frame times for photons generated along the jet.

    Integrates inverse velocities analytically to determine propagation times.

    Args:
        initial_radius: Radial positions of generated photons.
        gamma: Shell Lorentz factors at initial_radius.
        BL_0: Initial bulk Lorentz factor.
        THETA_J: Opening semi-angle.
        Theta_star: Temperature parameter at Z_STAR.
        Z_0: Base radius.
        Z_STAR: Reference radius.
        Z_T: Termination radius.
        E_F: Energy flux parameter.

    Returns:
        cupy.ndarray: Initial generation times in lab frame.
    """
    time = cp.empty(initial_radius.size, dtype=DTYPE)
    T_STAR = Theta_star * M_EC2 / scpc.k
    T_0 = GetT0(Z_0, Z_STAR, T_STAR)
    BL_STAR = GetGammaBelowZStar(Z_STAR, BL_0, THETA_J, T_0, Z_0, E_F)
    B_1 = (2.0 * cp.pi * PARAMETER_A * (THETA_J * BL_0 * Z_0 * T_0**2)**2) / E_F
    B_2 = (2.0 * cp.pi * PARAMETER_A * BL_0 * BL_STAR * (THETA_J * Z_STAR * T_STAR**2)**2) / E_F
    u_A = 1.0
    below_Z_STAR = initial_radius <= Z_STAR
    u_star_i = B_1 * (initial_radius[below_Z_STAR] - Z_0) + 1.0
    time[below_Z_STAR] = GetTimeIntegral(BL_0, B_1, u_A, u_star_i)
    u_star = B_1 * (Z_STAR - Z_0) + 1.0
    t_1 = GetTimeIntegral(BL_0, B_1, u_A, u_star)

    below_Z_T = cp.logical_and(~below_Z_STAR, initial_radius < Z_T)
    u_T_i = B_2 * (initial_radius[below_Z_T] - Z_STAR) + 1.0
    time[below_Z_T] = GetTimeIntegral(BL_STAR, B_2, u_A, u_T_i) + t_1
    u_T = B_2 * (Z_T - Z_STAR) + 1.0
    t_2 = GetTimeIntegral(BL_STAR, B_2, u_A, u_T)

    above_Z_T = initial_radius >= Z_T
    time[above_Z_T] = t_1 + t_2 + (initial_radius[above_Z_T] - Z_T) / cp.sqrt(1.0 - 1.0 / (gamma[above_Z_T]**2))

    return time


def RunComptonDrag(THETA_R, THETA_E, THETA_J, BULK_LORENTZ_0, Z_0, Z_STAR, Z_T, Z_MAX, E_F, N_PHOTONS, COMOVING_DENSITY_COEFF):
    """Runs a Monte Carlo simulation of Compton Drag with multiple scattering events.

    Simulates the propagation and interaction of photons up to MAX_ITERATIONS,
    handling escapes outside the jet boundaries or termination radius.

    Args:
        THETA_R: Base radiation temperature parameter.
        THETA_E: Target electron temperature.
        THETA_J: Jet opening angle limit.
        BULK_LORENTZ_0: Base bulk Lorentz factor.
        Z_0: Start radius.
        Z_STAR: Transition radius.
        Z_T: Termination radius.
        Z_MAX: Outer boundary of initial generation.
        E_F: Energy flux parameter.
        N_PHOTONS: Number of photons.
        COMOVING_DENSITY_COEFF: Comoving density coefficient.

    Returns:
        tuple: Simulation results for surviving scattered photons:
            - photon_energies: Array of scattered photon energies.
            - photon_wave_vector: Direction 4-vectors.
            - photon_position: Position 4-vectors.
            - Q: Q Stokes parameters.
            - U: U Stokes parameters.
            - final_iteration: Iteration index of escape.
            - photon_theta: Polar angles.
            - photon_phi: Azimuthal angles.
    """
    initial_radius = GetInitialRadius(Z_0, Z_MAX, N_PHOTONS)

    photon_energies = SD.GetPlanck(N_PHOTONS, GetT(initial_radius, Z_STAR, THETA_R))

    photon_position = (initial_radius[:, None]) * cp.concatenate([cp.zeros((N_PHOTONS, 1), dtype=DTYPE), SD.GetIsotropicDirection(N_PHOTONS, THETA_J)], axis=1)

    photon_wave_vector = cp.concatenate([cp.ones((N_PHOTONS, 1), dtype=DTYPE), SD.GetIsotropicDirection(N_PHOTONS)], axis=1)

    emession_frame = cp.zeros((N_PHOTONS, 4), dtype=DTYPE)
    emession_frame[:, 0] = GetGamma(initial_radius, BULK_LORENTZ_0, THETA_J, THETA_R, Z_0, Z_STAR, Z_T, E_F)

    photon_position[:, 0] = GetInitialTime(initial_radius, emession_frame[:, 0], BULK_LORENTZ_0, THETA_J, THETA_R, Z_0, Z_STAR, Z_T, E_F)

    alive = cp.ones(N_PHOTONS, dtype=cp.bool_)
    final_iteration = cp.full(N_PHOTONS, -1, dtype=cp.int32)

    Q = cp.empty(N_PHOTONS, dtype=DTYPE)
    U = cp.empty(N_PHOTONS, dtype=DTYPE)
    photon_polarization_vector = cp.empty(photon_wave_vector.shape, dtype=DTYPE)
    electrons = cp.empty(emession_frame.shape, dtype=DTYPE)
    temp_position = cp.empty(photon_position.shape, dtype=DTYPE)

    for iteration in range(MAX_ITERATIONS):
        if not alive.any():
            break

        idx = cp.where(alive)[0]
        emession_frame[idx, 1:4] = photon_position[idx, 1:4] / cp.linalg.norm(photon_position[idx, 1:4], axis=1)[:, None]
        emession_frame[idx, 0] = GetGamma(cp.linalg.norm(photon_position[idx, 1:4], axis=1), BULK_LORENTZ_0, THETA_J, THETA_R, Z_0, Z_STAR, Z_T, E_F)

        electrons[idx] = cp.concatenate([SD.GetElectronGamma(cp.full(idx.size, THETA_E, dtype=DTYPE))[:, None], SD.GetIsotropicDirection(idx.size)], axis=1)

        compton_cross_section = IC.CalcPhotonKNCrossection(photon_energies[idx] * SC.GetLorentzTransform(SC.GetLorentzTransform(photon_wave_vector[idx], emession_frame[idx]), electrons[idx])[:, 0])
        mean_free_path = GetMeanFreePath(COMOVING_DENSITY_COEFF, emession_frame[idx, 0], cp.linalg.norm(photon_position[idx, 1:4], axis=1), cp.sum(photon_wave_vector[idx, 1:4] * emession_frame[idx, 1:4], axis=1), compton_cross_section)
        if iteration > 0:
            temp_position[idx, :] = photon_position[idx, 0:4] + photon_wave_vector[idx, 0:4] * mean_free_path[:, None]
            out_of_cork = cp.logical_or(cp.linalg.norm(temp_position[idx, 1:4], axis=1) > Z_T, cp.arccos(temp_position[idx, 3]/cp.linalg.norm(temp_position[idx, 1:4], axis=1)) > THETA_J)

            if out_of_cork.any():
                dead_idx = idx[out_of_cork]
                alive[dead_idx] = False
                final_iteration[dead_idx] = iteration + 1
                idx = idx[~out_of_cork]
            if idx.size == 0:
                break
            photon_position[idx, 0:4] += photon_wave_vector[idx, 0:4] * mean_free_path[(~out_of_cork), None]

        photon_wave_vector[idx] = SC.GetLorentzTransform(photon_wave_vector[idx], emession_frame[idx])
        photon_energies[idx] *= photon_wave_vector[idx, 0]
        photon_wave_vector[idx] = SC.NormalizeFourVector(photon_wave_vector[idx])

        if iteration == 0:
            photon_polarization_vector = IC.CalcRandomPolarization(photon_wave_vector)

        photon_energies[idx], photon_wave_vector[idx], photon_polarization_vector[idx] = IC.CalcICScattering(photon_energies[idx], photon_wave_vector[idx], photon_polarization_vector[idx], electrons[idx])
        Q[idx], U[idx] = IC.CalcStokes(photon_wave_vector[idx], photon_polarization_vector[idx])

        photon_wave_vector[idx] = SC.GetLorentzTransform(photon_wave_vector[idx], emession_frame[idx], BACKWARD)
        photon_energies[idx] *= photon_wave_vector[idx, 0]
        photon_wave_vector[idx] = SC.NormalizeFourVector(photon_wave_vector[idx])

        out_of_cork = cp.logical_or(cp.linalg.norm(photon_position[idx, 1:4], axis=1) > Z_T, cp.arccos(photon_position[idx, 3]/cp.linalg.norm(photon_position[idx, 1:4], axis=1)) > THETA_J)

        if out_of_cork.any():
            dead_idx = idx[out_of_cork]
            alive[dead_idx] = False
            final_iteration[dead_idx] = iteration + 1
            idx = idx[~out_of_cork]
        if idx.size == 0:
            break
        if iteration % 10 == 0:
            print("================================== ", iteration, "/", MAX_ITERATIONS)

    del compton_cross_section
    cp.get_default_memory_pool().free_all_blocks()

    alive = ~alive
    photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[alive])
    return photon_energies[alive], photon_wave_vector[alive], photon_position[alive], Q[alive], U[alive], final_iteration[alive], photon_theta, photon_phi
