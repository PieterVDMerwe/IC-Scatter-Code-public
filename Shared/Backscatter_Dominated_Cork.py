"""Simulation of backscatter-dominated jet cork with pair annihilation spectrum.

This module simulates photon transport, propagation, and Inverse Compton
scattering within a relativistic jet shell (cork) where backscattering dominates.
It uses GPU-accelerated computing via CuPy and Numba.
"""

from Shared import Distributions as SD
from Shared import Core_IC as IC
from Shared import Common as SC

import cupy as cp
import pickle
import numpy as np
import scipy.constants as scpc

############################################################## CONST ##########################################################################

DTYPE = cp.float64
JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e * scpc.c**2
NATURAL_TO_KEV = M_EC2 * JOULE_TO_KEV
MAX_ITERATIONS = 26

FORWARD = False
BACKWARD = True


def GetMeanFreePath(COMOVING_DENSITY_COEFF, BL, current_radius, mu_lab, cross_section):
    """Calculates the mean free path for photons traveling in the jet.

    Args:
        COMOVING_DENSITY_COEFF: Scale factor for the comoving target electron density.
        BL: Bulk Lorentz factor of the jet shell.
        current_radius: CuPy array of current radial distances of the photons.
        mu_lab: Cosine of the angle of photon propagation in the lab frame.
        cross_section: Klein-Nishina cross-sections.

    Returns:
        cupy.ndarray: Sampled mean free paths for each photon.
    """
    return (-cp.log(1.0 - cp.random.random(size=current_radius.size, dtype=DTYPE))) / (
        (COMOVING_DENSITY_COEFF * cross_section / current_radius**2)
    )


fma_kernel = cp.ElementwiseKernel(
    "float64 a, float64 b, float64 c",
    "float64 out",
    "out = fma(a, b, c);",
    "fma_kernel"
)


def RunBackscatterDominatedCorkWPairAn(THETA_R, THETA_E, THETA_J, BULK_LORENTZ, INITIAL_RADIUS, COMOVING_DENSITY_COEFF, N_PHOTONS):
    """Runs a Monte Carlo simulation of photons propagating and scattering in a jet cork.

    Initially models photons from a pair annihilation spectrum, lets them propagate,
    calculates their mean free paths, performs Lorentz transforms, simulates
    Inverse Compton scattering events, and tracks polarizations/Stokes parameters.

    Args:
        THETA_R: Dimensionless temperature parameter of the radiation field.
        THETA_E: Dimensionless electron temperature.
        THETA_J: Jet opening/semi-angle limit.
        BULK_LORENTZ: Relativistic bulk Lorentz factor of the shell.
        INITIAL_RADIUS: Starting radial distance of the shell.
        COMOVING_DENSITY_COEFF: Comoving density coefficient of targets.
        N_PHOTONS: Number of photons to simulate.

    Returns:
        tuple: A tuple containing simulated values of escaped photons:
            - photon_energies: Array of escaped photon energies.
            - photon_wave_vector: Array of escaped photon direction 4-vectors.
            - photon_position: Array of escaped photon position 4-vectors.
            - Q: Array of Q Stokes parameters.
            - U: Array of U Stokes parameters.
            - final_iteration: Iteration index when each photon escaped.
            - photon_theta: Polar angles of the escaped direction.
            - photon_phi: Azimuthal angles of the escaped direction.
    """
    photon_energies = SD.GetPairAnnihilationRejectionMethod(N_PHOTONS, THETA_R, a=1.0e-6, b=30.0)

    photon_wave_vector = cp.concatenate([cp.ones((N_PHOTONS, 1), dtype=DTYPE), SD.GetIsotropicDirection(N_PHOTONS, THETA_J)], axis=1)
    photon_position = (INITIAL_RADIUS + 0.0) * cp.concatenate([cp.zeros((N_PHOTONS, 1), dtype=DTYPE), photon_wave_vector[:, 1:4]], axis=1)

    emession_frame = cp.zeros((N_PHOTONS, 4), dtype=DTYPE)
    emession_frame[:, 0] = BULK_LORENTZ

    ######################################### FOR LOOP ######################################################
    alive = cp.ones(N_PHOTONS, dtype=cp.bool_)
    final_iteration = cp.full(N_PHOTONS, -1, dtype=cp.int32)
    distance_rel_cork_inner = cp.zeros(N_PHOTONS, dtype=DTYPE)

    Q = cp.empty(N_PHOTONS, dtype=DTYPE)
    U = cp.empty(N_PHOTONS, dtype=DTYPE)
    photon_polarization_vector = cp.empty(photon_wave_vector.shape, dtype=DTYPE)
    electrons = cp.empty(emession_frame.shape, dtype=DTYPE)

    for iteration in range(MAX_ITERATIONS):
        if not alive.any():
            break

        idx = cp.where(alive)[0]
        emession_frame[idx, 1:4] = photon_position[idx, 1:4] / cp.linalg.norm(photon_position[idx, 1:4], axis=1)[:, None]

        electrons[idx] = cp.concatenate([SD.GetElectronGamma(cp.full(idx.size, THETA_E, dtype=DTYPE))[:, None], SD.GetIsotropicDirection(idx.size)], axis=1)

        compton_cross_section = IC.CalcPhotonKNCrossection(photon_energies[idx] * SC.GetLorentzTransform(SC.GetLorentzTransform(photon_wave_vector[idx], emession_frame[idx]), electrons[idx])[:, 0])
        mean_free_path = cp.clip(GetMeanFreePath(COMOVING_DENSITY_COEFF, emession_frame[idx, 0], cp.linalg.norm(photon_position[idx, 1:4], axis=1), cp.sum(photon_wave_vector[idx, 1:4] * emession_frame[idx, 1:4], axis=1), compton_cross_section), 1.0, 1.0e7)
        photon_position[idx, 0:4] = fma_kernel(mean_free_path[:, None], photon_wave_vector[idx, 0:4], photon_position[idx, 0:4])
        emession_frame[idx, 1:4] = photon_position[idx, 1:4] / cp.linalg.norm(photon_position[idx, 1:4], axis=1)[:, None]

        distance_rel_cork_inner[idx] += mean_free_path * (cp.sum(emession_frame[idx, 1:4] * photon_wave_vector[idx, 1:4], axis=1) - (cp.sqrt(emession_frame[idx, 0]**2 - 1) / emession_frame[idx, 0]))

        if iteration >= 0:
            out_of_cork = distance_rel_cork_inner[idx] < 0.0

            if out_of_cork.any():
                dead_idx = idx[out_of_cork]
                alive[dead_idx] = False
                final_iteration[dead_idx] = iteration
                idx = idx[~out_of_cork]
            if idx.size == 0:
                break

        photon_wave_vector[idx] = SC.GetLorentzTransform(photon_wave_vector[idx], emession_frame[idx])
        photon_energies[idx] *= photon_wave_vector[idx, 0]
        photon_wave_vector[idx] = SC.NormalizeFourVector(photon_wave_vector[idx])

        if iteration == 0:
            photon_polarization_vector = IC.CalcRandomPolarization(photon_wave_vector)
        photon_energies[idx], photon_wave_vector[idx], photon_polarization_vector[idx] = IC.CalcICScattering(photon_energies[idx], photon_wave_vector[idx], photon_polarization_vector[idx], electrons[idx])

        photon_wave_vector[idx] = SC.GetLorentzTransform(photon_wave_vector[idx], emession_frame[idx], BACKWARD)
        photon_energies[idx] *= photon_wave_vector[idx, 0]
        photon_wave_vector[idx] = SC.NormalizeFourVector(photon_wave_vector[idx])
        photon_polarization_vector_temp = IC.LorentzTransformPolarizationVector(photon_wave_vector[idx], photon_polarization_vector[idx], emession_frame[idx], BACKWARD)
        Q[idx], U[idx] = IC.CalcStokes(photon_wave_vector[idx], photon_polarization_vector_temp)

    del mean_free_path, compton_cross_section
    cp.get_default_memory_pool().free_all_blocks()

    print("error on 0 iteration count: ", len(final_iteration[final_iteration == 0]))

    alive = cp.logical_or(alive, photon_energies * NATURAL_TO_KEV < 1.0)

    photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[~alive])

    return photon_energies[~alive], photon_wave_vector[~alive], photon_position[~alive], Q[~alive], U[~alive], final_iteration[~alive], photon_theta, photon_phi
