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
# SIGMA_T = scpc.sigma_T
# THETA_R_PLANCK_CONST = 3.0e8*scpc.k/(scpc.m_e*scpc.c**2)
# THETA_E = 1.0
# THETA_R_PLANCK_CONST = 500.0/(M_EC2*JOULE_TO_EV)
# THETA_E = 50000.0/(M_EC2*JOULE_TO_EV)

# BULK_LORENTZ = 20.0
MAX_ITERATIONS = 100

FORWARD = False
BACKWARD = True

def GetMeanFreePath(COMOVING_DENSITY_COEFF,BL,current_radius,mu_lab,cross_section):
    return (-cp.log(1.0-cp.random.random(size=current_radius.size,dtype=DTYPE)))/( (COMOVING_DENSITY_COEFF/(BL*current_radius**2))*cross_section*(BL-cp.sqrt(BL**2-1.0)*mu_lab ) )

def GetInitialRadius(Z_0,Z_MAX,N_PHOTONS):
    return Z_0*(Z_MAX/Z_0)**cp.random.random(N_PHOTONS,dtype=DTYPE)
    # return Z_0*Z_MAX/(Z_MAX-(Z_MAX-Z_0)*cp.random.random(N_PHOTONS,dtype=DTYPE))

def GetT(initial_radius,Z_STAR,initial_T): # returns T in units of m_e c**2
    return initial_T*(initial_radius/Z_STAR)**(-PARAMETER_B)

def GetT0(Z_0,Z_STAR,T_STAR):
    return T_STAR*(Z_0/Z_STAR)**(-PARAMETER_B)

def GetGammaBelowZStar(initial_radius,BL_0,THETA_J,T_0,Z_0,E_F):
    return BL_0/( 1.0+2.0*cp.pi*(THETA_J**2)*PARAMETER_A*(T_0**4)*(BL_0**2)*(Z_0**2)*(initial_radius-Z_0)/E_F )

def GetGammaBelowZT(initial_radius,BL_0,BL_STAR,THETA_J,T_STAR,Z_STAR,E_F):
    return BL_STAR/( 1.0+2.0*cp.pi*(THETA_J**2)*PARAMETER_A*(T_STAR**4)*(BL_0*BL_STAR)*(Z_STAR**2)*(initial_radius-Z_STAR)/E_F )

def GetGamma(initial_radius,BL_0,THETA_J,Theta_star,Z_0,Z_STAR,Z_T,E_F):
    gamma = cp.empty(initial_radius.size,dtype=DTYPE)
    T_STAR = Theta_star*M_EC2/scpc.k
    T_0    = GetT0(Z_0,Z_STAR,T_STAR)
    BL_STAR = GetGammaBelowZStar(Z_STAR,BL_0,THETA_J,T_0,Z_0,E_F)
    below_Z_STAR = initial_radius <= Z_STAR
    gamma[below_Z_STAR] = GetGammaBelowZStar(initial_radius[below_Z_STAR],BL_0,THETA_J,T_0,Z_0,E_F)
    below_Z_T = cp.logical_and(~below_Z_STAR,initial_radius<Z_T)
    gamma[below_Z_T] = GetGammaBelowZT(initial_radius[below_Z_T],BL_0,BL_STAR,THETA_J,T_STAR,Z_STAR,E_F)
    above_Z_T = initial_radius >= Z_T
    gamma[above_Z_T] = GetGammaBelowZT(Z_T,BL_0,BL_STAR,THETA_J,T_STAR,Z_STAR,E_F)
    return gamma

def GetTimeIntegral(BL_A,B,u_A,u_i):
    return (BL_A/B)*(cp.arcsin(u_i/BL_A)-cp.arcsin(u_A/BL_A))


def GetInitialTime(initial_radius,gamma,BL_0,THETA_J,Theta_star,Z_0,Z_STAR,Z_T,E_F):
    time = cp.empty(initial_radius.size,dtype=DTYPE)
    T_STAR = Theta_star*M_EC2/scpc.k
    T_0    = GetT0(Z_0,Z_STAR,T_STAR)
    BL_STAR = GetGammaBelowZStar(Z_STAR,BL_0,THETA_J,T_0,Z_0,E_F)
    B_1 = (2.0*cp.pi*PARAMETER_A*(THETA_J*BL_0*Z_0*T_0**2)**2)/E_F
    B_2 = (2.0*cp.pi*PARAMETER_A*BL_0*BL_STAR*(THETA_J*Z_STAR*T_STAR**2)**2)/E_F
    u_A = 1.0
    below_Z_STAR = initial_radius <= Z_STAR
    u_star_i = B_1*(initial_radius[below_Z_STAR]-Z_0)+1.0
    time[below_Z_STAR] = GetTimeIntegral(BL_0,B_1,u_A,u_star_i)
    u_star = B_1*(Z_STAR-Z_0)+1.0
    t_1 = GetTimeIntegral(BL_0,B_1,u_A,u_star)

    below_Z_T = cp.logical_and(~below_Z_STAR,initial_radius<Z_T)
    u_T_i = B_2*(initial_radius[below_Z_T]-Z_STAR)+1.0
    time[below_Z_T] = GetTimeIntegral(BL_STAR,B_2,u_A,u_T_i)+t_1
    u_T = B_2*(Z_T-Z_STAR)+1.0
    t_2 = GetTimeIntegral(BL_STAR,B_2,u_A,u_T)

    # print("t_1, t_2",t_1/scpc.c,t_2/scpc.c)

    above_Z_T = initial_radius >= Z_T
    time[above_Z_T] = t_1+t_2+(initial_radius[above_Z_T] - Z_T)/cp.sqrt(1.0-1.0/(gamma[above_Z_T]**2) )

    return time







def RunComptonDrag(THETA_R,THETA_E,THETA_J,BULK_LORENTZ_0,Z_0,Z_STAR,Z_T,Z_MAX,E_F,N_PHOTONS,COMOVING_DENSITY_COEFF):
    initial_radius = GetInitialRadius(Z_0,Z_MAX,N_PHOTONS)
    # print(initial_radius)

    # hist, edges = cp.histogram(cp.log10(initial_radius),bins=50)
    # centers = (edges[1:]+edges[:-1])/2.0
    # plt.plot(centers.get(),hist.get())
    # plt.show()

    photon_energies = SD.GetPlanck(N_PHOTONS,GetT(initial_radius,Z_STAR,THETA_R))

    photon_position = (initial_radius[:,None])*cp.concatenate([cp.zeros((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS,THETA_J)],axis=1)


    photon_wave_vector = cp.concatenate([cp.ones((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS)],axis=1)
    # print(photon_wave_vector[:10,:])

    emession_frame = cp.zeros((N_PHOTONS,4),dtype=DTYPE)
    emession_frame[:,0] = GetGamma(initial_radius,BULK_LORENTZ_0,THETA_J,THETA_R,Z_0,Z_STAR,Z_T,E_F)

    photon_position[:,0] = GetInitialTime(initial_radius,emession_frame[:,0],BULK_LORENTZ_0,THETA_J,THETA_R,Z_0,Z_STAR,Z_T,E_F)
    # print("############### Time min, max:",cp.min(photon_position[:,0]),cp.max(photon_position[:,0]))
    # print("############### R min, max:",cp.min(cp.linalg.norm(photon_position[:,1:4],axis=1)),cp.max(cp.linalg.norm(photon_position[:,1:4],axis=1)))
    # plt.scatter(initial_radius.get(),(photon_position[:,0]-initial_radius).get()/scpc.c)
    # plt.show()
    # print(emession_frame[:,0])
    # emession_frame[:,3] = 1.0
    ######################################### FOR LOOP ######################################################
    alive = cp.ones(N_PHOTONS,dtype=cp.bool_)

    final_iteration = cp.full(N_PHOTONS,-1,dtype=cp.int32)

    Q = cp.empty(N_PHOTONS,dtype=DTYPE)
    U = cp.empty(N_PHOTONS,dtype=DTYPE)
    photon_polarization_vector = cp.empty(photon_wave_vector.shape,dtype=DTYPE)
    electrons = cp.empty(emession_frame.shape,dtype=DTYPE)
    temp_position = cp.empty(photon_position.shape,dtype=DTYPE)

    for iteration in range(MAX_ITERATIONS):
        if not alive.any():
            break

        idx = cp.where(alive)[0]
        emession_frame[idx,1:4] = photon_position[idx,1:4]/cp.linalg.norm(photon_position[idx,1:4],axis=1)[:,None]
        emession_frame[idx,0] = GetGamma(cp.linalg.norm(photon_position[idx,1:4],axis=1),BULK_LORENTZ_0,THETA_J,THETA_R,Z_0,Z_STAR,Z_T,E_F)


        electrons[idx] = cp.concatenate([ SD.GetElectronGamma(cp.full(idx.size,THETA_E,dtype=DTYPE))[:,None] ,SD.GetIsotropicDirection(idx.size)],axis=1)


        compton_cross_section = IC.CalcPhotonKNCrossection(photon_energies[idx]*SC.GetLorentzTransform(SC.GetLorentzTransform(photon_wave_vector[idx],emession_frame[idx]),electrons[idx])[:,0])
        mean_free_path = GetMeanFreePath(COMOVING_DENSITY_COEFF,emession_frame[idx,0],cp.linalg.norm(photon_position[idx,1:4],axis=1),cp.sum(photon_wave_vector[idx,1:4]*emession_frame[idx,1:4],axis=1),compton_cross_section)
        if iteration > 0:
            # photon_position[idx,0:4] += photon_wave_vector[idx,0:4]*mean_free_path[:,None]
            temp_position[idx,:] = photon_position[idx,0:4] + photon_wave_vector[idx,0:4]*mean_free_path[:,None]
            out_of_cork = cp.logical_or(cp.linalg.norm(temp_position[idx,1:4],axis=1) > Z_T, cp.arccos(temp_position[idx,3]/cp.linalg.norm(temp_position[idx,1:4],axis=1)) > THETA_J )

            if out_of_cork.any():
                dead_idx = idx[out_of_cork]
                alive[dead_idx] = False
                final_iteration[dead_idx] = iteration+1
                idx = idx[~out_of_cork]
                # if iteration == 0:
                #     print(mean_free_path)
            if idx.size == 0:
                break
            photon_position[idx,0:4] += photon_wave_vector[idx,0:4]*mean_free_path[(~out_of_cork),None]
        # print("##################### mean free path Comparison ############################")
        # for ll in range(10):
        #     print(initial_radius[ll],mean_free_path[ll])
        # print("################## mean free path Comparison end ###########################")



        # print("Before",photon_wave_vector[0,0],cp.linalg.norm(photon_wave_vector[0,1:4]))
        photon_wave_vector[idx] = SC.GetLorentzTransform(photon_wave_vector[idx],emession_frame[idx])
        photon_energies[idx] *= photon_wave_vector[idx,0]
        photon_wave_vector[idx] = SC.NormalizeFourVector(photon_wave_vector[idx])
        # print("Mid",photon_wave_vector[0,0],cp.linalg.norm(photon_wave_vector[0,1:4]))

        if iteration == 0:
            photon_polarization_vector = IC.CalcRandomPolarization(photon_wave_vector)

        photon_energies[idx], photon_wave_vector[idx], photon_polarization_vector[idx] = IC.CalcICScattering(photon_energies[idx],photon_wave_vector[idx],photon_polarization_vector[idx],electrons[idx])


        Q[idx], U[idx] = IC.CalcStokes(photon_wave_vector[idx],photon_polarization_vector[idx])


        #######################################################################################################################

        photon_wave_vector[idx] = SC.GetLorentzTransform(photon_wave_vector[idx],emession_frame[idx],BACKWARD)
        photon_energies[idx] *= photon_wave_vector[idx,0]
        photon_wave_vector[idx] = SC.NormalizeFourVector(photon_wave_vector[idx])
        # print("After",photon_wave_vector[0,0],cp.linalg.norm(photon_wave_vector[0,1:4]))

        out_of_cork = cp.logical_or(cp.linalg.norm(photon_position[idx,1:4],axis=1) > Z_T, cp.arccos(photon_position[idx,3]/cp.linalg.norm(photon_position[idx,1:4],axis=1)) > THETA_J )

        if out_of_cork.any():
            dead_idx = idx[out_of_cork]
            alive[dead_idx] = False
            final_iteration[dead_idx] = iteration+1
            idx = idx[~out_of_cork]
            # if iteration == 0:
            #     print(mean_free_path)
        if idx.size == 0:
            break
        if iteration%10 == 0:
            print("================================== ",iteration,"/",MAX_ITERATIONS)

    del compton_cross_section, #mean_free_path
    cp.get_default_memory_pool().free_all_blocks()

    # alive = cp.logical_or(alive,final_iteration==0)

    ############################### Just for debugging ################################################
    # photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[alive])
    # alive = ~alive
    ###################################################################################################
    # photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[~alive])
    alive = ~alive
    photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector[alive])
    # return photon_energies[~alive], photon_wave_vector[~alive], photon_position[~alive], Q[~alive], U[~alive], final_iteration[~alive], photon_theta, photon_phi
    return photon_energies[alive], photon_wave_vector[alive], photon_position[alive], Q[alive], U[alive], final_iteration[alive], photon_theta, photon_phi



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































