"""Script to plot polarization testing simulation results.

Loads data from 'Polarizations_Test.h5' and plots azimuthal phi counts
along with polarization vs theta distributions for verification.
"""

import Shared.Plot_Functions as SP
import h5py as h5

import scipy.constants as scpc
import math as m

############################################################## CONST ##########################################################################

N_PHOTONS = 4000000
N_PHOTONS_Larger = 100000000

THETA_R_PAIR = 3.0  # Temperature for Pair Annihilation spectrum where Theta_r = (kT_e)/(m_e c**2)
JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e * scpc.c**2
NATURAL_TO_KEV = M_EC2 * JOULE_TO_KEV
SIGMA_T = 6.6524587158e-29  # Thomson cross-section in m^2
THETA_R_PLANCK_CONST = 500.0 / (M_EC2 * JOULE_TO_EV)
THETA_E = 1.4
THETA_J = 0.1
INITIAL_RADIUS = m.sqrt(1.0e21)
M_DOT = 1.0e30  # kg/s
BULK_LORENTZ = 100.0

VIEWING_ANGLES = [3.0 / BULK_LORENTZ, 5.0 / BULK_LORENTZ, 7.0 / BULK_LORENTZ]

######################################################### MAIN ######################################################################

# Path to your HDF5 file
FILE_PATH = r"Polarizations_Test/Polarizations_Test.h5"
OUTPUT_PATH = "Polarizations_Test/PT_"

# Open the file in read mode
with h5.File(FILE_PATH, 'r') as f:
    SP.PlotPhiTest(f, [m.pi / 2.0, m.pi / 3.0, m.pi / 6.0, 0.1], OUTPUT_PATH, num_bins=100)
    SP.PlotPolarizationVsThetaTest(f, OUTPUT_PATH, num_bins=500)
