import Shared.Distributions as SD
import Shared.Core_IC as IC
import Shared.Common as SC
import Shared.Backscatter_Dominated_Cork as BS


from    numba             import cuda
import  numba
import  numpy as np
import  cupy as cp
import  math as m
from    numba.cuda        import libdevice as cudalib
import  scipy.constants   as scpc
from    numba.cuda.random import create_xoroshiro128p_states as cStates
from    numba.cuda.random import xoroshiro128p_uniform_float64 as uniform
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
# import  pickle
import h5py as h5
import argparse


# import cupy as cp
import matplotlib.pyplot as plt



############################################################## CONST ##########################################################################


DTYPE = cp.float64
N_PHOTONS = 4000000
N_PHOTONS_Larger = 100000000

THETA_R_PAIR = 3.0 #Temperature for Pair Annihilation spectrum where Theta_r = (kT_e)/(m_e c**2)
JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e*scpc.c**2
NATURAL_TO_KEV = M_EC2*JOULE_TO_KEV
SIGMA_T = 6.6524587158e-29  # Thomson cross-section in m^2
# THETA_R_PLANCK_CONST = 3.0e8*scpc.k/(scpc.m_e*scpc.c**2)
# THETA_E = 1.0
THETA_R_PLANCK_CONST = 500.0/(M_EC2*JOULE_TO_EV)
THETA_E = 0.06#0.25#0.3#0.4#
THETA_J = 0.1#cp.pi#0.1
INITIAL_RADIUS = m.sqrt(1.0e21)
M_DOT   = 1.0e30 # kg/s
BULK_LORENTZ = 20.0#100.0
# COMOVING_DENSITY_COEFF = SIGMA_T*M_DOT/(np.pi*scpc.m_p*scpc.c*np.sqrt(BULK_LORENTZ**2-1.0)*(np.sin(THETA_J)**2))
COMOVING_DENSITY_COEFF = SIGMA_T*M_DOT/(np.pi*scpc.m_p*scpc.c*np.sqrt(1.0-1.0/BULK_LORENTZ**2)*(np.sin(THETA_J)**2))


DISTRIBUTIONS = ["PairAnnihilation","Planck","Gamma","Isotropic"]

FORWARD = False
BACKWARD = True

NUMBER_OF_RUNS = 48
WRITE_THRESHOLD = 4

FILE_PATH = "Backscatter_Dominated_Cork_Data/Backscatter_Dominated_Cork_Data.h5"


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
    photon_energies, photon_wave_vector, photon_position, Q, U, final_iteration, photon_theta, photon_phi = BS.RunBackscatterDominatedCorkWPairAn(THETA_R_PAIR,THETA_E,THETA_J,BULK_LORENTZ,INITIAL_RADIUS,COMOVING_DENSITY_COEFF,N_PHOTONS)


    memory_buffer["photon_energies"].append(cp.asnumpy(photon_energies))
    memory_buffer["photon_wave_vector"].append(cp.asnumpy(photon_wave_vector))
    memory_buffer["photon_position"].append(cp.asnumpy(photon_position))
    memory_buffer["Q"].append(cp.asnumpy(Q))
    memory_buffer["U"].append(cp.asnumpy(U))
    memory_buffer["final_iteration"].append(cp.asnumpy(final_iteration))
    memory_buffer["photon_theta"].append(cp.asnumpy(photon_theta))
    memory_buffer["photon_phi"].append(cp.asnumpy(photon_phi))

    if i+1 % WRITE_THRESHOLD == 0:
        SC.FlushBufferToDisk(memory_buffer,first_write,FILE_PATH)
        first_write = False
        print(f"Flush To Disk ################################### {i+1}/{NUMBER_OF_RUNS}")
    else:
        print(f"################################################# {i+1}/{NUMBER_OF_RUNS}")

if any(len(v) > 0 for v in memory_buffer.values() ):
    SC.FlushBufferToDisk(memory_buffer,first_write,FILE_PATH)



















