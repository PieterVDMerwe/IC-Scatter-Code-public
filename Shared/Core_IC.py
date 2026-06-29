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

def CalcPhotonKNCrossection(photon_energies):#Ephoton_energiespects photon energies in units of m_e * c**2 (naturalized units)
    return (3.0/4.0)*( ((1.0+photon_energies)/(photon_energies**3))*( (2.0*photon_energies*(1.0+photon_energies))/(1.0+2.0*photon_energies) -cp.log(1.0+2.0*photon_energies) ) + ((cp.log(1.0+2.0*photon_energies))/(2.0*photon_energies)) - (1.0+3.0*photon_energies)/(1.0+2.0*photon_energies)**2 )



@cuda.jit(device=True,fastmath=False)
def ScatteringAngle(photon_energy,mu):
    i =cuda.grid(1)
    lower = 2.0/3.0 + 2.0*photon_energy +m.log(1.0+2.0*photon_energy)/photon_energy
    upper = photon_energy*(3.0/2.0 + mu*(1.0-mu/2.0))+(1.0+mu**3)/3.0 - ( m.log(1.0+photon_energy*(1.0-mu)) - m.log(1.0+2.0*photon_energy) )/photon_energy
    return (upper/lower)

@cuda.jit(fastmath=False)
def CalcScatMu(photon_energies,xi_theta_sc,mu):
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
            zlz = zlz+1


        mu[i] = mu_mid


def CalcScEnergies(photon_energies,scattered_mu):
    return photon_energies/(1.0+photon_energies*(1.0-scattered_mu ))



@cuda.jit(device=True,fastmath=False)
def PhiSc(pNu,scattered_mu,pNuSc,phiSc):
    i = cuda.grid(1)
    y = 1.0-scattered_mu**2
    epe = pNuSc/pNu
    eep = pNu/pNuSc
    return phiSc-(y*cudalib.sin(phiSc)*cudalib.cos(phiSc))/(epe+eep-y)


@cuda.jit(fastmath=False)
def CalcScPhi(xi_phi,photon_energies,scattered_mu,scattered_photon_energies,phiSc):
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
            phi_val = PhiSc(photon_energies[i], scattered_mu[i], scattered_photon_energies[i], phi_mid)  # Evaluate PhiSc at phi_mid

            if (phi_high - phi_low) < 1e-9:
                Not_Found = False
            elif phi_val > target:
                phi_high = phi_mid
            else:
                phi_low = phi_mid

        # Convergence check


    phiSc[i] =  phi_mid



def CalcScDir(thetaSc,phiSc,pDirec,pPolarization):
    pDirecSc = cp.empty(pDirec.shape,dtype=DTYPE)
    ctSc = cp.cos(thetaSc)
    stcpSc = cp.sin(thetaSc)*cp.cos(phiSc)
    stspSc = cp.sin(thetaSc)*cp.sin(phiSc)

    pDirecSc[:,1] = pDirec[:,1]*ctSc + pPolarization[:,1]*stcpSc+stspSc*(pDirec[:,2]*pPolarization[:,3]-pDirec[:,3]*pPolarization[:,2])
    pDirecSc[:,2] = pDirec[:,2]*ctSc + pPolarization[:,2]*stcpSc+stspSc*(pDirec[:,3]*pPolarization[:,1]-pDirec[:,1]*pPolarization[:,3])
    pDirecSc[:,3] = pDirec[:,3]*ctSc + pPolarization[:,3]*stcpSc+stspSc*(pDirec[:,1]*pPolarization[:,2]-pDirec[:,2]*pPolarization[:,1])
    pDirecSc[:,0] = 1.0

    pDirecSc[:,1:] = pDirecSc[:,1:]/cp.sqrt(pDirecSc[:,1]**2+pDirecSc[:,2]**2+pDirecSc[:,3]**2)[:,None]

    return pDirecSc

# def CalcScDir(thetaSc,phiSc,pDirec,pPolarization): ###########Pol Independent
#     pDirecSc = cp.empty(pDirec.shape,dtype=DTYPE)
#
#     # normalize incoming photon direction
#     k = pDirec[:,1:4]
#     k = k / cp.sqrt(cp.sum(k**2, axis=1))[:,None]
#
#     # choose a reference vector not parallel to k
#     ref = cp.where(cp.abs(k[:,0:1])<0.9,
#                    cp.array([1.0,0.0,0.0],dtype=DTYPE),
#                    cp.array([0.0,1.0,0.0],dtype=DTYPE))
#
#     # construct orthonormal basis e1,e2 perpendicular to k
#     e1 = cp.cross(ref, k)
#     e1 = e1 / cp.sqrt(cp.sum(e1**2, axis=1))[:,None]
#     e2 = cp.cross(k, e1)
#
#     # scattered direction
#     ctSc = cp.cos(thetaSc)[:,None]
#     stSc = cp.sin(thetaSc)[:,None]
#     cosphi = cp.cos(phiSc)[:,None]
#     sinphi = cp.sin(phiSc)[:,None]
#
#     pDirecSc[:,1:4] = (stSc*cosphi)*e1 + (stSc*sinphi)*e2 + ctSc*k
#     pDirecSc[:,0] = 1.0
#
#     # normalize result
#     pDirecSc[:,1:4] /= cp.sqrt(cp.sum(pDirecSc[:,1:4]**2,axis=1))[:,None]
#
#     return pDirecSc


def CalcPD(pNu, thetaSc, pNuSc, phiSc):
    temp = (cp.sin(thetaSc) ** 2) * (cp.cos(phiSc) ** 2)
    epe = pNuSc / pNu
    eep = pNu / pNuSc
    denominator = epe + eep - 2.0 * temp
    return 2.0 * ((1.0 - temp) / denominator)

import cupy as cp

def CalcPolarization(pDirecSc, thetaSc, phiSc, pPolarization):
    """
    Vectorized update of pPolarizationSc using CuPy.

    All inputs must be CuPy arrays of shape (N,) or (N, 4) as appropriate.
    This function modifies pPolarizationSc in-place.
    """
    sin_theta_cos_phi = cp.sin(thetaSc) * cp.cos(phiSc)  # shape (N,)

    pPolarizationSc = cp.zeros(pPolarization.shape,dtype=DTYPE)

    # Update polarization vector components (x, y, z = indices 1, 2, 3)
    pPolarizationSc[:, 1] = pDirecSc[:, 1] * sin_theta_cos_phi - pPolarization[:, 1]
    pPolarizationSc[:, 2] = pDirecSc[:, 2] * sin_theta_cos_phi - pPolarization[:, 2]
    pPolarizationSc[:, 3] = pDirecSc[:, 3] * sin_theta_cos_phi - pPolarization[:, 3]

    pPolarizationSc[:,1:] = pPolarizationSc[:,1:]/cp.sqrt(pPolarizationSc[:,1]**2+pPolarizationSc[:,2]**2+pPolarizationSc[:,3]**2)[:,None]

    return pPolarizationSc

import cupy as cp

def CalcRandomPolarization( direc):
    """
    Vectorized version of CalcInitPol using CuPy.

    Parameters:
        pi (cp.ndarray): Scalar array (length 1) containing pi[0].
        direc (cp.ndarray): CuPy array of shape (N, 4) — direction vectors (only [1,2,3] are used).

    Returns:
        pol (cp.ndarray): CuPy array of shape (N, 4), where [1,2,3] are the polarization components.
    """
    N = direc.shape[0]
    pol = cp.zeros((N, 4), dtype=cp.float64)  # Output array

    # Sample random alpha_r values
    alpha_r = 2.0 * cp.pi* cp.random.random(size=N)

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

    # --- Special-case: if direction is ~parallel to z-axis, pick xy-plane polarization ---
    mask = cp.abs(dz) > 1.0 - 1e-12
    if cp.any(mask):
        pol[mask, 1] = cp.cos(alpha_r[mask])
        pol[mask, 2] = cp.sin(alpha_r[mask])
        pol[mask, 3] = 0.0

    return pol




def CalculateComptonScattering(photon_energies,direction,polarization,send_theta_sc = False):
    xi_theta = cp.random.random(photon_energies.size,dtype=DTYPE)
    mu_scatter = cp.empty(photon_energies.size,dtype=DTYPE)

    N = len(photon_energies)
    GRID_SIZE = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    CalcScatMu[GRID_SIZE, BLOCK_SIZE](photon_energies,xi_theta,mu_scatter)

    scattered_photon_energies = CalcScEnergies(photon_energies,mu_scatter)
    # cp.clip(scattered_photon_energies ,0.0,1.0,out=scattered_photon_energies )
    # print("scattered energies",(scattered_photon_energies-photon_energies)[(scattered_photon_energies-photon_energies)>0.0])

    # print(scattered_photon_energies[scattered_photon_energies-photon_energies > 1.0])

    phi_scatter = cp.empty(photon_energies.size,dtype=DTYPE)
    xi_phi = cp.random.random(photon_energies.size,dtype=DTYPE)
    CalcScPhi[GRID_SIZE, BLOCK_SIZE](xi_phi,photon_energies,mu_scatter,scattered_photon_energies,phi_scatter)

    theta_scatter = cp.arccos(mu_scatter)

    scattered_direction = CalcScDir(theta_scatter,phi_scatter,direction,polarization)

    PD = CalcPD(photon_energies,theta_scatter,scattered_photon_energies,phi_scatter)
    xi_PD = cp.random.random(photon_energies.size,dtype=DTYPE)

    accepted = xi_PD < PD

    scattered_polarization = cp.empty(polarization.shape,dtype=DTYPE)

    scattered_polarization[accepted] = CalcPolarization(scattered_direction[accepted],theta_scatter[accepted],phi_scatter[accepted],polarization[accepted])
    scattered_polarization[~accepted] = CalcRandomPolarization(scattered_direction[~accepted])
    scattered_polarization[:,0] = 0.0

    if send_theta_sc :
        return scattered_photon_energies, scattered_direction, scattered_polarization, theta_scatter
    else:
        return scattered_photon_energies, scattered_direction, scattered_polarization

def LorentzTransformPolarizationVector(photon_direction,polarization_vector,new_frame,Inverse = False):
    temp_polarization = SC.GetLorentzTransform(polarization_vector,new_frame,Inverse)
    new_polarization = cp.empty(temp_polarization.shape,dtype=DTYPE)
    for k in [1,2,3]:
        new_polarization[:,k] = temp_polarization[:,k]-temp_polarization[:,0]*photon_direction[:,k]/photon_direction[:,0]
    new_polarization = SC.NormalizeFourVector(new_polarization,0.0)
    return new_polarization


def CalcICScattering(photon_energies,photon_wave_vectors,polarization_vector,electron_frame,send_theta_sc = False):
    # print("photon direction befor scattering: ", photon_wave_vectors[0:10,:])
    electron_frame_wave_vectors         = SC.GetLorentzTransform(photon_wave_vectors,electron_frame)
    electron_frame_photon_energies      = photon_energies*electron_frame_wave_vectors[:,0]
    electron_frame_polarization_vector  = LorentzTransformPolarizationVector(electron_frame_wave_vectors,polarization_vector,electron_frame)
    electron_frame_wave_vectors         = SC.NormalizeFourVector(electron_frame_wave_vectors)
    # electron_frame_polarization_vector  = LorentzTransformPolarizationVector(electron_frame_wave_vectors,polarization_vector,electron_frame)
    if send_theta_sc:
        scattered_photon_energies, scattered_direction, scattered_polarization, theta_scatter = CalculateComptonScattering(electron_frame_photon_energies,electron_frame_wave_vectors,electron_frame_polarization_vector,send_theta_sc=send_theta_sc)
    else:
        scattered_photon_energies, scattered_direction, scattered_polarization = CalculateComptonScattering(electron_frame_photon_energies,electron_frame_wave_vectors,electron_frame_polarization_vector)
    # scattered_photon_energies   = electron_frame_photon_energies
    # scattered_direction         = electron_frame_wave_vectors
    # scattered_polarization      = electron_frame_polarization_vector
    # print(scattered_photon_energies[cp.where(scattered_photon_energies > 1.0)])
    # print(electron_frame_photon_energies[cp.where(scattered_photon_energies > 1.0)])
    # print(cp.linalg.norm(scattered_direction[0:10,1:4],axis=1)[:,None])

    # print("Before",electron_frame_photon_energies[0:10])

    emission_frame_wave_vectors         = SC.GetLorentzTransform(scattered_direction,electron_frame,BACKWARD)
    emission_frame_photon_energies      = scattered_photon_energies*emission_frame_wave_vectors[:,0]
    emission_frame_polarization_vector  = LorentzTransformPolarizationVector(emission_frame_wave_vectors,scattered_polarization,electron_frame,BACKWARD)
    emission_frame_wave_vectors         = SC.NormalizeFourVector(emission_frame_wave_vectors)
    # print("photon direction after scattering: ", emission_frame_wave_vectors[0:10,:])
    # print("electron direction: ", electron_frame[0:10,:])
    # emission_frame_polarization_vector  = LorentzTransformPolarizationVector(emission_frame_wave_vectors,scattered_polarization,electron_frame,BACKWARD)

    # print("After",emission_frame_photon_energies[0:10])

    if send_theta_sc:
        return emission_frame_photon_energies,emission_frame_wave_vectors,emission_frame_polarization_vector, theta_scatter
    else:
        return emission_frame_photon_energies,emission_frame_wave_vectors,emission_frame_polarization_vector


def CalcStokes(pDirec, pPolarization):
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

        Um = cp.stack([ UCoeff * (d2[~mask] - d1[~mask] * d3[~mask]),
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
############# implementation without treating special case ##########################################
    # d = pDirec[:, 1:4]
    # p = pPolarization[:, 1:4]
    # # print("d",d[0:10,:])
    # # print("p",p[0:10,:])
    #
    # d1, d2, d3 = d[:, 0], d[:, 1], d[:, 2]
    # one_minus_d3sq = 1.0 - d3**2
    #
    # QCoeff = 1.0 / cp.sqrt(one_minus_d3sq)
    # UCoeff = 1.0 / cp.sqrt(2.0 * one_minus_d3sq)
    #
    # # Q+ and Q-
    # Qp = cp.stack([-QCoeff * d2, QCoeff * d1, cp.zeros_like(d1)], axis=1)
    #
    # Qm = cp.stack([-QCoeff * d1 * d3,-QCoeff * d2 * d3, QCoeff * (1.0 - d3**2)], axis=1)
    #
    # # U+ and U-
    # Up = cp.stack([-UCoeff * (d2 + d1 * d3),UCoeff * (d1 - d2 * d3), UCoeff * (1.0 - d3**2)], axis=1)
    #
    # Um = cp.stack([UCoeff * (d2 - d1 * d3),-UCoeff * (d1 + d2 * d3), UCoeff * (1.0 - d3**2)], axis=1)
    #
    # # Dot products
    # dot_Qp = cp.sum(Qp * p, axis=1)
    # dot_Qm = cp.sum(Qm * p, axis=1)
    # dot_Up = cp.sum(Up * p, axis=1)
    # dot_Um = cp.sum(Um * p, axis=1)
    #
    # # Calculate and return Q and U with dtype float64
    # Q = (dot_Qp**2 - dot_Qm**2).astype(cp.float64)
    # U = (dot_Up**2 - dot_Um**2).astype(cp.float64)
    #
    # return Q, U




























