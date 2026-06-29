from Shared import Distributions as SD
from Shared import Core_IC as IC
from Shared import Common as SC

import cupy as cp
import  pickle
import  numpy as np
import  scipy.constants   as scpc



############################################################## CONST ##########################################################################


DTYPE = cp.float64
# N_PHOTONS = 5000000
# N_PHOTONS_Larger = 100000000

# THETA_R_PAIR = 3.0 #Temperature for Pair Annihilation spectrum where Theta_r = (kT_e)/(m_e c**2)
JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e*scpc.c**2
NATURAL_TO_KEV = M_EC2*JOULE_TO_KEV
# SIGMA_T = scpc.sigma_T
# THETA_R_PLANCK_CONST = 3.0e8*scpc.k/(scpc.m_e*scpc.c**2)
# THETA_E = 1.0
# THETA_R_PLANCK_CONST = 500.0/(M_EC2*JOULE_TO_EV)
# THETA_E = 50000.0/(M_EC2*JOULE_TO_EV)

# BULK_LORENTZ = 20.0
MAX_ITERATIONS = 26

FORWARD = False
BACKWARD = True

# def GetMeanFreePath(COMOVING_DENSITY_COEFF,BL,current_radius,mu_lab,cross_section):
#     return (-cp.log(1.0-cp.random.random(size=current_radius.size,dtype=DTYPE)))/( (COMOVING_DENSITY_COEFF/current_radius**2)*cross_section*(BL-cp.sqrt(BL**2-1.0)*mu_lab ) )#
def GetMeanFreePath(COMOVING_DENSITY_COEFF,BL,current_radius,mu_lab,cross_section):
    return (-cp.log(1.0-cp.random.random(size=current_radius.size,dtype=DTYPE)))/( (COMOVING_DENSITY_COEFF*cross_section/current_radius**2) )#*(BL-cp.sqrt(BL**2-1.0)*mu_lab )

fma_kernel = cp.ElementwiseKernel(
    "float64 a, float64 b, float64 c",
    "float64 out",
    "out = fma(a, b, c);",
    "fma_kernel"
)

def RunBackscatterDominatedCorkWPairAn(THETA_R,THETA_E,THETA_J,BULK_LORENTZ,INITIAL_RADIUS,COMOVING_DENSITY_COEFF,N_PHOTONS):
    # photon_position = (INITIAL_RADIUS+1.0)*cp.concatenate([cp.zeros((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS,THETA_J)],axis=1)


    # photon_energies = SD.GetPlanck(N_PHOTONS,cp.full(N_PHOTONS,THETA_R,dtype=DTYPE))GetMonoenergeticPhotons(N_photons,photon_energy)
    photon_energies = SD.GetPairAnnihilationRejectionMethod(N_PHOTONS, THETA_R,a=1.0e-6,b=30.0)#GetPlanck(N_PHOTONS,cp.full(N_PHOTONS,THETA_R,dtype=DTYPE))
    # photon_energies_original = photon_energies.copy()
    # photon_energies = SD.GetPairAnnihilation(N_PHOTONS, THETA_R)#GetPlanck(N_PHOTONS,cp.full(N_PHOTONS,THETA_R,dtype=DTYPE))
    # photon_energies = SD.GetMonoenergeticPhotons(N_PHOTONS,1.0)#3000.0/NATURAL_TO_KEV)
    # print("photon_energies",photon_energies)
    # print("cp.max(photon_energies)",cp.max(photon_energies))

    photon_wave_vector = cp.concatenate([cp.ones((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS,THETA_J)],axis=1)
    photon_position = (INITIAL_RADIUS+0.0)*cp.concatenate([cp.zeros((N_PHOTONS,1),dtype=DTYPE),photon_wave_vector[:,1:4]],axis=1)
    # photon_position = (INITIAL_RADIUS+0.0)*cp.concatenate([cp.zeros((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS,THETA_J)],axis=1)
    # print(photon_wave_vector[:10,:])
    # photon_wave_vector = cp.concatenate([cp.ones((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS,THETA_J)],axis=1)

    emession_frame = cp.zeros((N_PHOTONS,4),dtype=DTYPE)
    emession_frame[:,0] = BULK_LORENTZ
    # emession_frame[:,3] = 1.0



    # photon_wave_vector = cp.concatenate([cp.ones((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS)],axis=1)
    # photon_wave_vector = SC.GetLorentzTransform(photon_wave_vector,emession_frame,BACKWARD)
    # # photon_energies[idx] *= photon_wave_vector[idx,0]
    # photon_wave_vector = SC.NormalizeFourVector(photon_wave_vector)


    ######################################### FOR LOOP ######################################################
    alive = cp.ones(N_PHOTONS,dtype=cp.bool_)
    ignore = cp.zeros(N_PHOTONS,dtype=cp.bool_)

    final_iteration = cp.full(N_PHOTONS,-1,dtype=cp.int32)

    distance_rel_cork_inner = cp.zeros(N_PHOTONS,dtype=DTYPE)

    Q = cp.empty(N_PHOTONS,dtype=DTYPE)
    U = cp.empty(N_PHOTONS,dtype=DTYPE)
    photon_polarization_vector = cp.empty(photon_wave_vector.shape,dtype=DTYPE)
    electrons = cp.empty(emession_frame.shape,dtype=DTYPE)

    for iteration in range(MAX_ITERATIONS):
        if not alive.any():
            break

        idx = cp.where(alive)[0]
        emession_frame[idx,1:4] = photon_position[idx,1:4]/cp.linalg.norm(photon_position[idx,1:4],axis=1)[:,None]
        # if iteration == 0:
        #     photon_energies[idx] *= SC.GetLorentzTransform(photon_wave_vector[idx],emession_frame[idx],BACKWARD)[idx,0]


        electrons[idx] = cp.concatenate([ SD.GetElectronGamma(cp.full(idx.size,THETA_E,dtype=DTYPE))[:,None] ,SD.GetIsotropicDirection(idx.size)],axis=1)
        # print("average electron lorentz factor:",cp.sum(electrons[idx,0])/len(electrons[idx,0]))
        # print("Min electronlorentz factor:",cp.min(electrons[idx,0]))
        # print("Max electronlorentz factor:",cp.max(electrons[idx,0]))
        # electrons[idx] = cp.concatenate([ 10.0*cp.ones((idx.size,1),dtype=DTYPE),cp.zeros((idx.size,1),dtype=DTYPE),cp.zeros((idx.size,1),dtype=DTYPE),cp.ones((idx.size,1),dtype=DTYPE) ],axis=1)

        # if iteration > 0:
        compton_cross_section = IC.CalcPhotonKNCrossection(photon_energies[idx]*SC.GetLorentzTransform(SC.GetLorentzTransform(photon_wave_vector[idx],emession_frame[idx]),electrons[idx])[:,0])
        # print(photon_energies[idx])
        # mean_free_path = GetMeanFreePath(COMOVING_DENSITY_COEFF,emession_frame[idx,0],cp.linalg.norm(photon_position[idx,1:4],axis=1),photon_wave_vector[idx,3],compton_cross_section)
        mean_free_path = cp.clip(GetMeanFreePath(COMOVING_DENSITY_COEFF,emession_frame[idx,0],cp.linalg.norm(photon_position[idx,1:4],axis=1),cp.sum(photon_wave_vector[idx,1:4]*emession_frame[idx,1:4],axis=1),compton_cross_section), 1.0,1.0e7 )
        # if iteration > 0:
        # photon_position[idx,0:4] += photon_wave_vector[idx,0:4]*mean_free_path[:,None]

        # photon_position[idx,0:4] += photon_wave_vector[idx,0:4]*mean_free_path[:,None]
        photon_position[idx, 0:4] = fma_kernel(mean_free_path[:, None],photon_wave_vector[idx, 0:4],photon_position[idx, 0:4])
        emession_frame[idx,1:4] = photon_position[idx,1:4]/cp.linalg.norm(photon_position[idx,1:4],axis=1)[:,None]

        distance_rel_cork_inner[idx] += mean_free_path*(cp.sum(emession_frame[idx,1:4]*photon_wave_vector[idx, 1:4],axis=1)-(cp.sqrt(emession_frame[idx,0]**2-1)/emession_frame[idx,0] ))

        # print(len(mean_free_path))
        # print(cp.linalg.norm(photon_position[0,1:4]),cp.sum(photon_wave_vector[0,1:4]*emession_frame[0,1:4]),emession_frame[0,0],compton_cross_section[0])

        # # print(cp.linalg.norm(photon_position[idx,1:4],axis=1) - (INITIAL_RADIUS + photon_position[idx,0]*cp.sqrt(1.0-1.0/emession_frame[idx,0]**2)))
        if iteration >= 0:
            # out_of_cork = cp.logical_or(cp.linalg.norm(photon_position[idx,1:4],axis=1) < (INITIAL_RADIUS + photon_position[idx,0]*cp.sqrt(1.0-1.0/emession_frame[idx,0]**2)), cp.arccos(photon_position[idx,3]/cp.linalg.norm(photon_position[idx,1:4],axis=1)) > THETA_J )


            out_of_cork = distance_rel_cork_inner[idx]<0.0

            # out_of_cork = cp.linalg.norm(photon_position[idx,1:4],axis=1) < (INITIAL_RADIUS + photon_position[idx,0]*cp.sqrt(1.0-1.0/emession_frame[idx,0]**2))
            # out_of_cork = cp.logical_or(photon_position[idx,3] < (INITIAL_RADIUS*(photon_position[idx,3]/cp.linalg.norm(photon_position[idx,1:4],axis=1)) + photon_position[idx,0]*cp.sqrt(1.0-1.0/emession_frame[idx,0]**2)), cp.arccos(photon_position[idx,3]/cp.linalg.norm(photon_position[idx,1:4],axis=1)) > THETA_J )
            # print(cp.arccos(photon_position[idx,3]/cp.linalg.norm(photon_position[idx,1:4],axis=1))[:20])
            if out_of_cork.any():
                dead_idx = idx[out_of_cork]

                # Transform the photon directions to the emission frame
                # photon_dirs_in_emission_frame = SC.GetLorentzTransform(photon_wave_vector[dead_idx], emession_frame[dead_idx])[:, 1:4]
                # print("Escaped photon directions in emission frame:\n", photon_dirs_in_emission_frame[ photon_dirs_in_emission_frame[:,2]>0.0 ])
                alive[dead_idx] = False
                # ignore[dead_idx[(photon_dirs_in_emission_frame[:,2])>0.0]]
                final_iteration[dead_idx] = iteration
                idx = idx[~out_of_cork]
            if idx.size == 0:
                break


        # emession_frame[idx,1:4] = photon_position[idx,1:4]/cp.linalg.norm(photon_position[idx,1:4],axis=1)[:,None]


        # print("Before",photon_wave_vector[0,0],cp.linalg.norm(photon_wave_vector[0,1:4]))
        # print("before: ",photon_energies[idx])
        photon_wave_vector[idx] = SC.GetLorentzTransform(photon_wave_vector[idx],emession_frame[idx])
        photon_energies[idx] *= photon_wave_vector[idx,0]
        photon_wave_vector[idx] = SC.NormalizeFourVector(photon_wave_vector[idx])
        # print("after: ",photon_energies[idx])
        # print("Mid",photon_wave_vector[0,0],cp.linalg.norm(photon_wave_vector[0,1:4]))

        if iteration == 0:
            photon_polarization_vector = IC.CalcRandomPolarization(photon_wave_vector)
        # idx_temp = cp.random.random(size=compton_cross_section[idx].size,dtype=DTYPE) < compton_cross_section[idx]
        # photon_energies[idx[idx_temp]], photon_wave_vector[idx[idx_temp]], photon_polarization_vector[idx[idx_temp]] = IC.CalcICScattering(photon_energies[idx[idx_temp]],photon_wave_vector[idx[idx_temp]],photon_polarization_vector[idx[idx_temp]],electrons[idx[idx_temp]])
        # photon_energies_temp = photon_energies[idx]
        photon_energies[idx], photon_wave_vector[idx], photon_polarization_vector[idx] = IC.CalcICScattering(photon_energies[idx],photon_wave_vector[idx],photon_polarization_vector[idx],electrons[idx])
        # print("Energy Diff",( (photon_energies_temp-photon_energies[idx]) )[electrons[idx,0]+(photon_energies_temp-photon_energies[idx])<0.0] )
        # print("Gamma",electrons[idx,0] )


        # Q[idx], U[idx] = IC.CalcStokes(photon_wave_vector[idx],photon_polarization_vector[idx])


        #######################################################################################################################

        photon_wave_vector[idx] = SC.GetLorentzTransform(photon_wave_vector[idx],emession_frame[idx],BACKWARD)
        photon_energies[idx] *= photon_wave_vector[idx,0]
        photon_wave_vector[idx] = SC.NormalizeFourVector(photon_wave_vector[idx])
        # print("Diff in energies before and after transforms", (cp.abs(photon_energies_original-photon_energies)) )
        # print("Max diff in energies before and after transforms", cp.max(cp.abs(photon_energies_original-photon_energies)) )
        # # print("After",photon_wave_vector[0,0],cp.linalg.norm(photon_wave_vector[0,1:4]))
        photon_polarization_vector_temp = IC.LorentzTransformPolarizationVector(photon_wave_vector[idx],photon_polarization_vector[idx],emession_frame[idx],BACKWARD)
        Q[idx], U[idx] = IC.CalcStokes(photon_wave_vector[idx],photon_polarization_vector_temp)

    del mean_free_path, compton_cross_section
    cp.get_default_memory_pool().free_all_blocks()

    print("error on 0 iteration count: ",len(final_iteration[final_iteration==0]))

    # alive = cp.logical_or(alive,final_iteration==0)
    # alive = cp.logical_or(alive,ignore)
    # alive = ~alive
    alive = cp.logical_or(alive,photon_energies*NATURAL_TO_KEV < 1.0)

    # photon_energies = SD.GetPairAnnihilation(N_PHOTONS, THETA_R)###########################################################

    photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[~alive])
    ############################### Just for debugging ################################################
    # photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[alive])
    # alive = ~alive
    ###################################################################################################
    return photon_energies[~alive], photon_wave_vector[~alive], photon_position[~alive], Q[~alive], U[~alive], final_iteration[~alive], photon_theta, photon_phi



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































