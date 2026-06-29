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



def GetPairAnnihilation(N_photons, Theta_r):  #Takes Theta_r and random numbers in random_number and saves random_number in units of m_E c**2
    try:
        print("Trying to load Pair annihilation grid")
        with open( str(MODULE_DIR)+"epsi_p_CPU_"+str(Theta_r)+".dat",'rb') as f:
            epsi_p_CPU = pickle.load(f)
        with open( str(MODULE_DIR)+"CFD_Pair_CPU_"+str(Theta_r)+".dat",'rb') as f:
            CFD_Pair_CPU = pickle.load(f)

        epsi_p = cp.array(epsi_p_CPU)
        CFD_Pair = cp.array(CFD_Pair_CPU)

    except:
        print("Failed to load Pair annihilation grid")
        print("... re-calculating grid")

        # Energy grid
        # epsi_p = cp.arange(0.01, 2000.0, 0.001)
        epsi_p = cp.arange(1.0e-5, 30.0, 1.0e-6)
        N_interval = len(epsi_p)
        N_interval_GPU = cp.array([N_interval])

        # Initialize arrays
        spectrum = cp.zeros(N_interval)
        Trapezoids = cp.zeros(N_interval - 1)
        Integrals = cp.zeros(N_interval - 1)
        CFD_Pair = cp.zeros(N_interval - 1)

        # Calculate spectrum: exp(-(C1 * epsi_p^2) / Theta_r^2) / epsi_p
        C1 = 0.045
        spectrum = cp.exp(-(C1 * epsi_p**2) / (Theta_r**2)) / epsi_p

        # Calculate trapezoids: 0.5 * 0.001 * (spectrum[i] + spectrum[i+1])
        # Trapezoids = 0.5 * 0.001 * (spectrum[:-1] + spectrum[1:])
        Trapezoids = 0.5 * 1.0e-6 * (spectrum[:-1] + spectrum[1:])

        # Calculate cumulative integrals: sum of Trapezoids up to index i
        Integrals = cp.cumsum(Trapezoids)

        # Finalize integrals: normalize by the last integral value
        CFD_Pair = Integrals / Integrals[-1]

        with open( str(MODULE_DIR)+"epsi_p_CPU_"+str(Theta_r)+".dat",'wb') as f:
            pickle.dump(epsi_p,f)

        with open( str(MODULE_DIR)+"CFD_Pair_CPU_"+str(Theta_r)+".dat",'wb') as f:
            pickle.dump(CFD_Pair,f)

    finally:
        random_number = cp.random.random(size=N_photons,dtype=DTYPE)

        k = cp.searchsorted(CFD_Pair,random_number, side= 'right' )-1
        k = cp.clip(k,0,len(CFD_Pair)-2)
        u = (random_number-CFD_Pair[k])/(CFD_Pair[k+1]-CFD_Pair[k])
        photon_Energy = epsi_p[k] + (epsi_p[k+1]-epsi_p[k])*u
        # print(photon_Energy[photon_Energy<=0.0])

        return photon_Energy.astype(DTYPE)

import cupy as cp

def GetPairAnnihilationRejectionMethod(N_photons, Theta_r, a=1.0e-7, b=30.0):
    """
    Sample photon energies from pair annihilation distribution using rejection sampling.

    Parameters
    ----------
    N_photons : int
        Number of photon energies to draw.
    Theta_r : float
        Dimensionless temperature parameter.
        C = C1 / Theta_r^2 with C1 = 0.045.
    a, b : float, optional
        Minimum and maximum epsilon range (default 1e-7 to 30.0).

    Returns
    -------
    photon_Energy : cp.ndarray
        Photon energies in units of m_e c^2, dtype=DTYPE.
    """
    C1 = 0.045
    C = C1 / (Theta_r**2)

    # Allocate output
    photon_Energy = cp.empty(N_photons, dtype=DTYPE)

    i = 0
    batch_factor = 4
    while i < N_photons:
        # propose more to reduce rejections
        m = (N_photons - i) * batch_factor

        # proposals ~ 1/eps distribution
        U = cp.random.rand(m, dtype=DTYPE)
        eps_prop = a * (b / a) ** U

        # accept with prob exp(-C * eps^2)
        V = cp.random.rand(m, dtype=DTYPE)
        accept = V <= cp.exp(-C * eps_prop**2)

        accepted = eps_prop[accept]
        take = min(len(accepted), N_photons - i)
        if take > 0:
            photon_Energy[i:i+take] = accepted[:take]
            i += take

    return photon_Energy.astype(DTYPE)




################################################################### Planck #######################################################################


@cuda.jit(fastmath=False)
def GeneratePlanckEnergy(photon_Energy,states,Theta_r,SIZE):
    i = cuda.grid(1)
    # SIZE = SIZE_ARRAY[0]
    if i < SIZE:
        sigma1 = uniform(states,i)
        sigma2 = uniform(states,i)
        sigma3 = uniform(states,i)
        photon_Energy[i] = - ((Theta_r[i])/(GenerateAlpha(states,SIZE)))*m.log(sigma1*sigma2*sigma3)


@cuda.jit(device=True,fastmath=False)
def GenerateAlpha(states,SIZE):
    i = cuda.grid(1)
    if i < SIZE:
        alpha = 0.0
        while alpha == 0.0:
            sigma1 = 1.202*uniform(states,i)
            if sigma1 < 1.0:
                alpha = 1.0
            else:
                m = 1
                checkHigher = True
                while checkHigher:
                    m = 1 + m
                    if (Sum(m-1)<=sigma1) and (sigma1 < Sum(m)):
                        alpha = float(m)
                        if alpha == 0:
                            pass
                            #print("what")
                        checkHigher = False
                    if m > 100000:
                        #print("m very large")
                        checkHigher = False
        return alpha


def GetPlanck(N_photons,Theta_r): #Expects Theta_r to be a cupy array in units of m_e c**2

    rng_states = cStates(N_photons,seed=random.randint(0, RAND_SEED_TOPEND))
    photon_Energy = cp.zeros(N_photons,dtype=DTYPE)
    grid_size = (N_photons + BLOCK_SIZE - 1) // BLOCK_SIZE
    GeneratePlanckEnergy[grid_size,BLOCK_SIZE](photon_Energy,rng_states,Theta_r,N_photons)

    return photon_Energy.astype(DTYPE)


################################################## Maxwell_Juttner #########################################################################
def CalcGammaLow(Theta_e):
    N = Theta_e.shape[0]
    accepted = cp.zeros(N, dtype= cp.bool_)
    eta = cp.zeros(N,dtype=DTYPE)

    while not cp.all(accepted):
        idxs = cp.where(~accepted)[0]
        n = Theta_e[idxs]

        sigma1 = cp.random.random(size=idxs.shape[0])
        sigma2 = cp.random.random(size=idxs.shape[0])
        xsiPrime = -(3.0/2.0) * cp.log(sigma1)

        accept = sigma2**2 < 0.151 * ((1.0 + n * xsiPrime)**2) * xsiPrime * (2.0 + n * xsiPrime) * sigma1

        # Compute eta where accepted
        new_eta = cp.sqrt(n * xsiPrime * (2.0 + n * xsiPrime))
        eta[idxs[accept]] = new_eta[accept]
        accepted[idxs[accept]] = True

    return eta.astype(DTYPE)


def CalcGammaHigh(Theta_e):
    N = Theta_e.shape[0]
    accepted = cp.zeros(N, dtype= cp.bool_)
    eta = cp.zeros(N,dtype=DTYPE)

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
        # accept = cp.logical_and(accept, 7.0*Theta_e[idxs[accept]]<cp.sqrt(1.0+(eta1[accept])**2 ) )
        # accept = cp.logical_and(accept, 10.0*Theta_e[idxs] > cp.sqrt(1.0 + eta1**2))
        eta[idxs[accept]] = eta1[accept]
        accepted[idxs[accept]] = True

    return eta.astype(DTYPE)


def GetElectronGamma(Theta_e):#Expects Theta_e to be a cupy array, overkill for current models, but enhances futureproofing
    eta = cp.zeros_like(Theta_e,dtype=DTYPE)
    low_mask = Theta_e<=0.29
    high_mask = ~low_mask

    if cp.any(low_mask):
        eta[low_mask] = CalcGammaLow(Theta_e[low_mask])
    if cp.any(high_mask):
        eta[high_mask] = CalcGammaHigh(Theta_e[high_mask])

    return cp.sqrt(1.0+eta**2,dtype=DTYPE)



###################################################### Isotropic Direction #######################################################

def GetIsotropicDirection(N,opening_angle=cp.pi,uniform_theta = False):
    sigma_theta = cp.random.random(size=N,dtype=DTYPE)
    if uniform_theta :
        theta = sigma_theta*opening_angle
    else:
        theta = cp.arccos(1.0-sigma_theta*(1.0-cp.cos(opening_angle)))
    phi = 2.0*cp.pi*cp.random.random(size=N)
    return cp.concatenate( [ (cp.sin(theta)*cp.cos(phi))[:,None], (cp.sin(theta)*cp.sin(phi))[:,None] ,cp.cos(theta)[:,None]] ,axis=1,dtype=DTYPE)


################################################### Monoenergetic ##################################################################

def GetMonoenergeticPhotons(N_photons,photon_energy):
    return cp.full(N_photons,photon_energy,dtype=DTYPE)

def GetMonoenergeticElectrons(N_photons,lorentz_factor):
    return cp.full(N_photons,lorentz_factor,dtype=DTYPE)












