"""Main execution script for multi-scatter Compton Drag jet shell simulation.

This script sets up physical constants and parameters (e.g. temperatures, radii,
jet angle constraints, comoving density coefficients), runs multiple Monte Carlo
simulation iterations for Compton Drag with multiple scattering events, buffers
the output data, and flushes it to HDF5.
"""

import Shared.Distributions as SD
import Shared.Core_IC as IC
import Shared.Common as SC
import Shared.Compton_Drag_Multi_Scatter as CD

from numba import cuda
import numba
import numpy as np
import cupy as cp
import math as m
from numba.cuda import libdevice as cudalib
import scipy.constants as scpc
from numba.cuda.random import create_xoroshiro128p_states as cStates
from numba.cuda.random import xoroshiro128p_uniform_float64 as uniform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import h5py as h5
import argparse

# import cupy as cp
import matplotlib.pyplot as plt

############################################################## CONST ##########################################################################

DTYPE = cp.float64
N_PHOTONS = 4000000
N_PHOTONS_Larger = 100000000

JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e * scpc.c**2
NATURAL_TO_KEV = M_EC2 * JOULE_TO_KEV
SIGMA_T = 6.6524587158e-29  # Thomson cross-section in m^2
THETA_R_PLANCK_CONST = 1.0e8 * scpc.k / M_EC2
THETA_E = 0.1
INITIAL_RADIUS = m.sqrt(1.0e21)
M_DOT = 1.0e18  # kg/s
BULK_LORENTZ_0 = 100.0
THETA_J = 5.0 / BULK_LORENTZ_0
COMOVING_DENSITY_COEFF = SIGMA_T * M_DOT / (np.pi * scpc.m_p * scpc.c * THETA_J**2)

THETA_R_STAR = 3.0e5 * scpc.k / M_EC2
Z_0 = 1.0e7
Z_STAR = 1.0e11
PARAMETER_B = 0.5
PARAMETER_G = 2.0
E_F = 1.0e44
Z_T = np.sqrt((SIGMA_T * E_F) / (np.pi * scpc.m_p * BULK_LORENTZ_0 * (scpc.c * THETA_J)**2))
Z_MAX = 5.0 * Z_T

DISTRIBUTIONS = ["PairAnnihilation", "Planck", "Gamma", "Isotropic"]

FORWARD = False
BACKWARD = True

NUMBER_OF_RUNS = 20
WRITE_THRESHOLD = 4

FILE_PATH = "Compton_Drag_Multi_Scatter_Data/Compton_Drag_Multi_Scatter_Data.h5"

memory_buffer = {
    "photon_energies": [],
    "photon_wave_vector": [],
    "photon_position": [],
    "Q": [],
    "U": [],
    "final_iteration": [],
    "photon_theta": [],
    "photon_phi": []
}

first_write = True

for i in range(NUMBER_OF_RUNS):
    photon_energies, photon_wave_vector, photon_position, Q, U, final_iteration, photon_theta, photon_phi = CD.RunComptonDrag(THETA_R_STAR, THETA_E, THETA_J, BULK_LORENTZ_0, Z_0, Z_STAR, Z_T, Z_MAX, E_F, N_PHOTONS, COMOVING_DENSITY_COEFF)

    memory_buffer["photon_energies"].append(cp.asnumpy(photon_energies))
    memory_buffer["photon_wave_vector"].append(cp.asnumpy(photon_wave_vector))
    memory_buffer["photon_position"].append(cp.asnumpy(photon_position))
    memory_buffer["Q"].append(cp.asnumpy(Q))
    memory_buffer["U"].append(cp.asnumpy(U))
    memory_buffer["final_iteration"].append(cp.asnumpy(final_iteration))
    memory_buffer["photon_theta"].append(cp.asnumpy(photon_theta))
    memory_buffer["photon_phi"].append(cp.asnumpy(photon_phi))

    if i + 1 % WRITE_THRESHOLD == 0:
        SC.FlushBufferToDisk(memory_buffer, first_write, FILE_PATH)
        first_write = False
        print(f"Flush To Disk ################################### {i+1}/{NUMBER_OF_RUNS}")
    else:
        print(f"################################################# {i+1}/{NUMBER_OF_RUNS}")

if any(len(v) > 0 for v in memory_buffer.values()):
    SC.FlushBufferToDisk(memory_buffer, first_write, FILE_PATH)
