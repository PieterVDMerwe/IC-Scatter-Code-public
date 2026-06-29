"""Script to plot backscatter-dominated jet cork simulation results.

Loads data from 'Backscatter_Dominated_Cork_Data.h5' and uses SP to generate
various plots like polar angle theta distributions, energy spectrums (SED),
polarization degrees, top-down polar positions, and light curves.
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

VIEWING_ANGLES = [0.105, 0.175, 0.255]

######################################################### MAIN ######################################################################

# Path to your HDF5 file
FILE_PATH = r"Backscatter_Dominated_Cork_Data/Backscatter_Dominated_Cork_Data.h5"
OUTPUT_PATH = "Backscatter_Dominated_Cork_Data/BS_"

# Open the file in read mode
with h5.File(FILE_PATH, 'r') as f:
    SP.PlotLightCurveAssumingSymmetry(f, VIEWING_ANGLES[0:4], OUTPUT_PATH, plot_backscatter_cork_thoeretical=True)
