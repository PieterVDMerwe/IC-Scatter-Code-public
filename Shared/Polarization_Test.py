"""Testing and verification script for polarization calculations under Compton Drag.

This module simulates monoenergetic photon polarization evolution and scattering
within a simplified reference frame setup, verifying the validity of Stokes
parameters and scattering dynamics.
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

E_THRESHOLD = 1.0 / NATURAL_TO_KEV
MAX_ITERATIONS = 1

FORWARD = False
BACKWARD = True


def RunComptonDrag(THETA_R, THETA_E, THETA_J, BULK_LORENTZ_0, Z_0, Z_STAR, Z_T, Z_MAX, E_F, N_PHOTONS):
    """Runs a polarization-tracking simulation for verification.

    Initializes monoenergetic photons aligned on specific axes, simulates
    scattering with stationary/directed target electrons, and outputs
    polarization vector updates alongside Stokes parameter verification.

    Args:
        THETA_R: Target radiation field temperature (unused in monoenergetic initialization).
        THETA_E: Electron temperature parameter.
        THETA_J: Opening semi-angle constraint.
        BULK_LORENTZ_0: Base bulk Lorentz factor.
        Z_0: Start radius.
        Z_STAR: Transition radius.
        Z_T: Termination radius.
        Z_MAX: Outer boundary of initial generation.
        E_F: Energy flux parameter.
        N_PHOTONS: Number of photons.

    Returns:
        tuple: Simulation verification outputs:
            - photon_energies: Scattered photon energies.
            - photon_wave_vector: Direction 4-vectors.
            - photon_position: Position 4-vectors.
            - Q: Q Stokes parameters.
            - U: U Stokes parameters.
            - final_iteration: Escape indices.
            - theta_scatter: Polar angles.
            - photon_phi: Azimuthal angles.
    """
    photon_energies = SD.GetMonoenergeticPhotons(N_PHOTONS, 1000.0 / NATURAL_TO_KEV)
    print(photon_energies)

    photon_position = cp.concatenate([cp.zeros((N_PHOTONS, 1), dtype=DTYPE), SD.GetIsotropicDirection(N_PHOTONS, THETA_J)], axis=1)

    photon_wave_vector = cp.zeros((N_PHOTONS, 4), dtype=DTYPE)
    photon_wave_vector[:, 0] = 1.0
    photon_wave_vector[:, 3] = 1.0

    emession_frame = cp.zeros((N_PHOTONS, 4), dtype=DTYPE)
    emession_frame[:, 0] = 1.0
    emession_frame[:, 3] = 1.0

    ######################################### FOR LOOP ######################################################
    alive = cp.ones(N_PHOTONS, dtype=cp.bool_)
    final_iteration = cp.full(N_PHOTONS, -1, dtype=cp.int32)

    Q = cp.empty(N_PHOTONS, dtype=DTYPE)
    U = cp.empty(N_PHOTONS, dtype=DTYPE)
    photon_polarization_vector = cp.empty(photon_wave_vector.shape, dtype=DTYPE)
    electrons = cp.empty(emession_frame.shape, dtype=DTYPE)

    electrons[:, 0] = 1.0
    electrons[:, 3] = 1.0

    photon_wave_vector[:] = SC.GetLorentzTransform(photon_wave_vector[:], emession_frame[:])
    photon_energies[:] *= photon_wave_vector[:, 0]
    photon_wave_vector[:] = SC.NormalizeFourVector(photon_wave_vector[:])

    photon_polarization_vector = IC.CalcRandomPolarization(photon_wave_vector)
    photon_polarization_vector[:, 0] = 0.0
    photon_polarization_vector[:, 1] = 1.0
    photon_polarization_vector[:, 2] = 0.0
    photon_polarization_vector[:, 3] = 0.0

    photon_energies[:], photon_wave_vector[:], photon_polarization_vector[:], theta_scatter = IC.CalcICScattering(photon_energies[:], photon_wave_vector[:], photon_polarization_vector[:], electrons[:], send_theta_sc=True)

    Q[:], U[:] = IC.CalcStokes(photon_wave_vector[:], photon_polarization_vector[:])

    photon_wave_vector[:] = SC.GetLorentzTransform(photon_wave_vector[:], emession_frame[:], BACKWARD)
    photon_energies[:] *= photon_wave_vector[:, 0]
    photon_wave_vector[:] = SC.NormalizeFourVector(photon_wave_vector[:])

    photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[alive])
    return photon_energies[alive], photon_wave_vector[alive], photon_position[alive], Q[alive], U[alive], final_iteration[alive], theta_scatter, photon_phi
