from Shared import Distributions as SD
from Shared import Core_IC as IC
from Shared import Common as SC

import cupy as cp
import  pickle
import  numpy as np
import  scipy.constants   as scpc

import matplotlib.pyplot as plt



############################################################## CONST ##########################################################################


DTYPE = cp.float64
# N_PHOTONS = 5000000
# N_PHOTONS_Larger = 100000000

# THETA_R_PAIR = 3.0 #Temperature for Pair Annihilation spectrum where Theta_r = (kT_e)/(m_e c**2)
JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e*scpc.c**2
NATURAL_TO_KEV = M_EC2*JOULE_TO_KEV

PARAMETER_B = 0.5
PARAMETER_G = 2.0
PARAMETER_A = 7.56*10**(-16)# in si

E_THRESHOLD = 1.0/NATURAL_TO_KEV
# SIGMA_T = scpc.sigma_T
# THETA_R_PLANCK_CONST = 3.0e8*scpc.k/(scpc.m_e*scpc.c**2)
# THETA_E = 1.0
# THETA_R_PLANCK_CONST = 500.0/(M_EC2*JOULE_TO_EV)
# THETA_E = 50000.0/(M_EC2*JOULE_TO_EV)

# BULK_LORENTZ = 20.0
MAX_ITERATIONS = 1

FORWARD = False
BACKWARD = True



def RunComptonDrag(THETA_R,THETA_E,THETA_J,BULK_LORENTZ_0,Z_0,Z_STAR,Z_T,Z_MAX,E_F,N_PHOTONS):


    photon_energies = SD.GetMonoenergeticPhotons(N_PHOTONS,1000.0/NATURAL_TO_KEV)
    print(photon_energies)

    photon_position = cp.concatenate([cp.zeros((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS,THETA_J)],axis=1)


    photon_wave_vector = cp.zeros((N_PHOTONS,4),dtype=DTYPE)#cp.concatenate([cp.ones((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS)],axis=1)#
    # photon_wave_vector = cp.concatenate([cp.ones((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS,uniform_theta=True)],axis=1)#cp.zeros((N_PHOTONS,4),dtype=DTYPE)#
    # print(photon_wave_vector[:10,:])
    photon_wave_vector[:,0] = 1.0
    photon_wave_vector[:,3] = 1.0

    emession_frame = cp.zeros((N_PHOTONS,4),dtype=DTYPE)
    emession_frame[:,0] = 1.0
    emession_frame[:,3] = 1.0


    ######################################### FOR LOOP ######################################################
    alive = cp.ones(N_PHOTONS,dtype=cp.bool_)

    final_iteration = cp.full(N_PHOTONS,-1,dtype=cp.int32)

    Q = cp.empty(N_PHOTONS,dtype=DTYPE)
    U = cp.empty(N_PHOTONS,dtype=DTYPE)
    photon_polarization_vector = cp.empty(photon_wave_vector.shape,dtype=DTYPE)
    electrons = cp.empty(emession_frame.shape,dtype=DTYPE)

    # for iteration in range(MAX_ITERATIONS):
    #     if not alive.any():
    #         break

    electrons[:,0] = 1.0
    electrons[:,3] = 1.0
    # electrons[:,1:4] = SD.GetIsotropicDirection(N_PHOTONS)#,uniform_theta=True)


    # energy_above_1keV_mask = 16.0*photon_energies*(electrons[:,0]**2)*(emession_frame[:,0]**2) <= E_THRESHOLD
    # # print(cp.where(energy_above_1keV_mask))
    # # print("E_THRESHOLD: ",E_THRESHOLD)
    # # print("Epsi: ",cp.min(16.0*photon_energies*(electrons[:,0]**2)*(emession_frame[:,0]**2)))
    #
    # while cp.any(energy_above_1keV_mask):
    #     photon_energies[energy_above_1keV_mask] = SD.GetPlanck(int(energy_above_1keV_mask.sum().get()),GetT(initial_radius[energy_above_1keV_mask],Z_STAR,THETA_R))
    #     energy_above_1keV_mask[energy_above_1keV_mask] = 16.0*photon_energies[energy_above_1keV_mask]*(electrons[energy_above_1keV_mask,0]**2)*(emession_frame[energy_above_1keV_mask,0]**2) <= E_THRESHOLD
    #
    # # print("epsi_min: ",NATURAL_TO_KEV*cp.min( photon_energies/(4.0*emession_frame[:,0]*electrons[:,0]) ))


    # compton_cross_section = IC.CalcPhotonKNCrossection(photon_energies[:]*SC.GetLorentzTransform(SC.GetLorentzTransform(photon_wave_vector[:],emession_frame[:]),electrons[:])[:,0])
    # alive = cp.random.random(N_PHOTONS,dtype=DTYPE) < compton_cross_section
    # print(photon_energies[:])
    # mean_free_path = GetMeanFreePath(COMOVING_DENSITY_COEFF,emession_frame[:,0],cp.linalg.norm(photon_position[:,1:4],axis=1),photon_wave_vector[:,3],compton_cross_section)
    # mean_free_path = GetMeanFreePath(COMOVING_DENSITY_COEFF,emession_frame[:,0],cp.linalg.norm(photon_position[:,1:4],axis=1),cp.sum(photon_wave_vector[:,1:4]*emession_frame[:,1:4],axis=1),compton_cross_section)
    # photon_position[:,0:4] += photon_wave_vector[:,0:4]*mean_free_path[:,None]
    # print(len(mean_free_path))
    # print(cp.linalg.norm(photon_position[0,1:4]),cp.sum(photon_wave_vector[0,1:4]*emession_frame[0,1:4]),emession_frame[0,0],compton_cross_section[0])

    # print(cp.linalg.norm(photon_position[:,1:4],axis=1) - (INITIAL_RADIUS + photon_position[:,0]*cp.sqrt(1.0-1.0/emession_frame[:,0]**2)))
    # out_of_cork = cp.logical_or(cp.linalg.norm(photon_position[:,1:4],axis=1) < (INITIAL_RADIUS + photon_position[:,0]*cp.sqrt(1.0-1.0/emession_frame[:,0]**2)),
    #                              cp.arccos(photon_position[:,3]/cp.linalg.norm(photon_position[:,1:4],axis=1)) > THETA_J )
    # print(cp.arccos(photon_position[:,3]/cp.linalg.norm(photon_position[:,1:4],axis=1))[:20])

    # out_of_cork = cp.random.random(N_PHOTONS,dtype=DTYPE) > compton_cross_section
    #
    # if out_of_cork.any():
    #     dead_: = :[out_of_cork]
    #     alive[dead_:] = False
    #     final_iteration[dead_:] = 0 #iteration
    #     : = :[~out_of_cork]
    # # if :.size == 0:
    # #     break
    #
    #
    # # print("Before",photon_wave_vector[0,0],cp.linalg.norm(photon_wave_vector[0,1:4]))
    photon_wave_vector[:] = SC.GetLorentzTransform(photon_wave_vector[:],emession_frame[:])
    photon_energies[:] *= photon_wave_vector[:,0]
    photon_wave_vector[:] = SC.NormalizeFourVector(photon_wave_vector[:])
    # # print("Mid",photon_wave_vector[0,0],cp.linalg.norm(photon_wave_vector[0,1:4]))
    #
    # # if iteration == 0:
    photon_polarization_vector = IC.CalcRandomPolarization(photon_wave_vector)
    # # # photon_polarization_vector[:,0] = photon_polarization_vector[0,0]
    # # # photon_polarization_vector[:,1] = photon_polarization_vector[0,1]
    # # # photon_polarization_vector[:,2] = photon_polarization_vector[0,2]
    # # # photon_polarization_vector[:,3] = photon_polarization_vector[0,3]
    photon_polarization_vector[:,0] = 0.0
    photon_polarization_vector[:,1] = 1.0
    photon_polarization_vector[:,2] = 0.0
    photon_polarization_vector[:,3] = 0.0
    # for ii in [0,1,2,3]:
    #     print("Polarization vector ",ii,":",len(cp.unique(photon_polarization_vector[:,ii])))
    # print(photon_polarization_vector[0:10,:])

    photon_energies[:], photon_wave_vector[:], photon_polarization_vector[:], theta_scatter = IC.CalcICScattering(photon_energies[:],photon_wave_vector[:],photon_polarization_vector[:],electrons[:],send_theta_sc=True)



    Q[:], U[:] = IC.CalcStokes(photon_wave_vector[:],photon_polarization_vector[:])


    #######################################################################################################################
    #
    photon_wave_vector[:] = SC.GetLorentzTransform(photon_wave_vector[:],emession_frame[:],BACKWARD)
    photon_energies[:] *= photon_wave_vector[:,0]
    photon_wave_vector[:] = SC.NormalizeFourVector(photon_wave_vector[:])
    # # print("After",photon_wave_vector[0,0],cp.linalg.norm(photon_wave_vector[0,1:4]))
    #
    # del compton_cross_section, #mean_free_path
    # cp.get_default_memory_pool().free_all_blocks()

    # alive = cp.logical_or(alive,final_iteration==0)

    ############################### Just for debugging ################################################
    # photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[alive])
    # alive = ~alive
    ###################################################################################################
    # photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[~alive])
    photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[alive])
    # return photon_energies[~alive], photon_wave_vector[~alive], photon_position[~alive], Q[~alive], U[~alive], final_iteration[~alive], photon_theta, photon_phi
    return photon_energies[alive], photon_wave_vector[alive], photon_position[alive], Q[alive], U[alive], final_iteration[alive], theta_scatter, photon_phi#photon_theta



#     electrons = cp.concatenate([ SD.GetElectronGamma(cp.full(N_PHOTONS,THETA_E,dtype=DTYPE))[:,None] ,SD.GetIsotropicDirection(N_PHOTONS)],axis=1)
#
#
#     compton_cross_section = IC.CalcPhotonKNCrossection(photon_energies*SC.GetLorentzTransform(SC.GetLorentzTransform(photon_wave_vector,emession_frame),electrons)[:,0])
#     mean_free_path = GetMeanFreePath(COMOVING_DENSITY_COEFF,emession_frame[:,0],cp.linalg.norm(photon_position[:,1:4],axis=1),photon_wave_vector[:,3],compton_cross_section)
#     photon_position[:,1:4] += photon_wave_vector[:,1:4]*mean_free_path[:,None]
#
#     photon_wave_vector = SC.GetLorentzTransform(photon_wave_vector,emession_frame)
#     photon_energies *= photon_wave_vector[:,0]
#     SC.NormalizeFourVector(photon_wave_vector)
#
#     photon_polarization_vector = IC.CalcRandomPolarization(photon_wave_vector)
#
#     photon_energies, photon_wave_vector, photon_polarization_vector = IC.CalcICScattering(photon_energies,photon_wave_vector,photon_polarization_vector,electrons)
#
#
#     Q, U = IC.CalcStokes(photon_wave_vector,photon_polarization_vector)
#
#
#     #######################################################################################################################
#
#     photon_wave_vector = SC.GetLorentzTransform(photon_wave_vector,emession_frame,BACKWARD)
#     photon_energies *= photon_wave_vector[:,0]
#     SC.NormalizeFourVector(photon_wave_vector)
#
#
#     return photon_energies, photon_wave_vector, Q, U































