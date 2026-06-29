import Shared.Distributions as SD
import Shared.Core_IC as IC
import Shared.Common as SC


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
import  pickle
import argparse


# import cupy as cp
import matplotlib.pyplot as plt



############################################################## CONST ##########################################################################


DTYPE = cp.float64
N_PHOTONS = 5000000
N_PHOTONS_Larger = 100000000

THETA_R_PAIR = 3.0 #Temperature for Pair Annihilation spectrum where Theta_r = (kT_e)/(m_e c**2)
JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e*scpc.c**2
NATURAL_TO_KEV = M_EC2*JOULE_TO_KEV
# THETA_R_PLANCK_CONST = 3.0e8*scpc.k/(scpc.m_e*scpc.c**2)
# THETA_E = 1.0
THETA_R_PLANCK_CONST = 500.0/(M_EC2*JOULE_TO_EV)
THETA_E = 50000.0/(M_EC2*JOULE_TO_EV)

BULK_LORENTZ = 10.0

DISTRIBUTIONS = ["PairAnnihilation","Planck","Gamma","Isotropic"]

FORWARD = False
BACKWARD = True

##################################################### Lorentz Tranform Tests ############################################################

def test_lorentz_transform_plot():
    N = 10000

    # 4-vectors at rest: (E = 1, p = 0)
    original = cp.zeros((N, 4), dtype=DTYPE)
    original[:, 0] = 1.0

    # Define boost velocity
    beta_vec = cp.array([0.9, 0.0, 0.0], dtype=DTYPE)
    beta_mag = cp.linalg.norm(beta_vec)
    gamma = 1.0 / cp.sqrt(1.0 - beta_mag**2)
    n_vec = beta_vec / beta_mag

    # Construct new_frame (N, 4): gamma + direction
    gamma_array = cp.full((N, 1), gamma, dtype=DTYPE)
    direction_array = cp.tile(n_vec, (N, 1))
    new_frame = cp.concatenate([gamma_array, direction_array], axis=1)

    # Apply transformation
    transformed = SC.GetLorentzTransform(original, new_frame)

    # Extract energy and spatial momentum magnitude
    E_before = cp.asnumpy(original[:, 0])
    p_before = cp.asnumpy(cp.linalg.norm(original[:, 1:], axis=1))

    E_after = cp.asnumpy(transformed[:, 0])
    p_after = cp.asnumpy(cp.linalg.norm(transformed[:, 1:], axis=1))

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(p_before, E_before, s=2, label="Before Boost", alpha=0.5)
    plt.scatter(p_after, E_after, s=2, label="After Boost", alpha=0.5)
    plt.xlabel("Momentum Magnitude")
    plt.ylabel("Energy")
    plt.title("Lorentz Transformation Test")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_isotropic_photon_lorentz_transform():
    # Define photon_energies and photon_directions externally
    # photon_energies: (N,) array of photon energies in natural units
    # photon_directions: (N,3) array of unit vectors
    photon_energies = SD.GetMonoenergeticPhotons(N_PHOTONS,1.0 ) #500000.0/(M_EC2*JOULE_TO_EV))
    photon_directions = SD.GetIsotropicDirection(N_PHOTONS)
    # print(photon_directions.dtype)
    # print(photon_energies.dtype)

    N = photon_energies.shape[0]

    # Build 4-vectors of the photons (shape: N x 4)
    # 4-vector: (E, px, py, pz) with |p| = E for photons (massless)
    momenta = photon_energies[:, None] * photon_directions  # shape (N,3)
    photon_4vectors = cp.concatenate([photon_energies[:, None], momenta], axis=1)  # shape (N,4)
    # print(photon_4vectors[0,:])

    # Define bulk motion in +z direction with gamma = 10
    gamma_bulk = 10.0 # +z

    # Create new_frame array of shape (N,4): (gamma, beta_x, beta_y, beta_z)
    beta_vec =  cp.array([0.0, 0.0, 1.0],dtype=DTYPE)# shape (3,)
    beta_vec_tiled = cp.tile(beta_vec, (N, 1))  # shape (N,3)
    gamma_arr = cp.full((N,1), gamma_bulk)
    new_frame = cp.concatenate([gamma_arr, beta_vec_tiled], axis=1)  # shape (N,4)

    # Lorentz transform
    transformed = SC.GetLorentzTransform(photon_4vectors, new_frame)
    # print(transformed[0,:])
    transformed_back = SC.GetLorentzTransform(transformed, new_frame,True)

    # Get transformed directions (normalize momentum)
    transformed_momentum = transformed[:, 1:]
    transformed_dirs = transformed_momentum / cp.linalg.norm(transformed_momentum, axis=1)[:, None]

    # Sample for plotting (if large)
    max_points = 5000
    if transformed_dirs.shape[0] > max_points:
        indices = cp.random.choice(transformed_dirs.shape[0], size=max_points, replace=False)
        dirs_to_plot = transformed_dirs[indices].get()
    else:
        dirs_to_plot = transformed_dirs.get()

    # Plot 3D directions
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dirs_to_plot[:,0], dirs_to_plot[:,1], dirs_to_plot[:,2], alpha=0.6, s=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Transformed Photon Directions in Jet Frame (γ = 10)")
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    SED_Energies, SED_Bins = cp.histogram(500000.0*transformed[:,0],bins=501)
    SED_Energies_original, SED_Bins_original = cp.histogram(500000.0*photon_4vectors[:,0],bins=501)
    SED_Energies_Back, SED_Bins_Back = cp.histogram(500000.0*transformed_back[:,0],bins=501)

    SED_Energies = SED_Energies/cp.max(SED_Energies)
    SED_Energies_original = SED_Energies_original/cp.max(SED_Energies_original)
    SED_Energies_Back = SED_Energies_Back/cp.max(SED_Energies_Back)

    ax2.plot(SED_Bins.get()[:-1],SED_Energies.get(),label="Transformed")
    ax2.plot(SED_Bins_original.get()[:-1],SED_Energies_original.get(),label="Original")
    ax2.plot(SED_Bins_Back.get()[:-1],SED_Energies_Back.get(),'x',label="Transformed Back")
    ax2.legend()
    plt.show()



########################################## IC Scattering Test Functions ########################

def RunKNCrossectionTest():
    energies = cp.arange(0.001,100.0,0.001)
    plt.plot(energies.get(),IC.CalcPhotonKNCrossection(energies).get())
    plt.show()


def compton_theta_pdf(mu, epsilon):
    lower = 2.0/3.0 + 2.0*epsilon + cp.log(1.0+2.0*epsilon)/epsilon
    term1 = epsilon * (1 - mu)
    term2 = mu**2
    term3 = 1 / (1 + epsilon * (1 - mu))
    return (term1 + term2 + term3) / lower


def RunScatterMuTest():
    # Sample setup
    N = 10**8
    E = 10.0  # photon energy in natural units
    xi_vals = cp.random.random(size=N)
    photon_energies = cp.full(N, E)
    mu_vals = cp.empty(N)

    # Launch your kernel
    threads_per_block = 128
    blocks = (N + threads_per_block - 1) // threads_per_block
    IC.CalcScatMu[blocks, threads_per_block](photon_energies, xi_vals, mu_vals)
    # theta = cp.arccos(mu_vals)

    # Histogram
    counts, bins = cp.histogram(mu_vals, bins=100, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    theoretical = compton_pdf(bin_centers,E)


    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(cp.arccos(bin_centers).get(), counts.get(), label="Monte Carlo")
    plt.plot(cp.arccos(bin_centers).get(), theoretical.get(), label="Theoretical", linestyle='--')
    plt.xlabel("Scattering angle θ [rad]")
    plt.ylabel("Normalized probability density")
    plt.legend()
    plt.title(f"Compton Scattering Angle Distribution (E = {E})")
    plt.grid(True)
    plt.show()



# --- Launch your Numba kernel with CuPy arrays ---
def run_CalcScPhi_kernel_CuPy(CalcScPhi, xi_phi, photon_energies, mu, scattered_energies):
    N = len(photon_energies)

    # Output array
    phiSc = cp.zeros(N, dtype=cp.float64)

    # Define thread/block config
    threads_per_block = 128
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # Launch kernel with CuPy device arrays
    IC.CalcScPhi[blocks_per_grid, threads_per_block](xi_phi, photon_energies, mu, scattered_energies, phiSc)

    return phiSc

# --- Analytic polarization PDF ---
def polarization_pdf(phi, mu, epe, eep):
    y = 1.0 - mu**2
    denom = epe + eep - y
    return 1.0 - (y * np.cos(2.0 * phi)) / denom

# --- Main test function ---
def test_phi_sampling_with_cupy_and_kernel():
    N = 10**8
    rng = cp.random.default_rng(42)

    # Incident photon energy (uniform for simplicity)
    pNu = cp.ones(N)

    # Cosine of scattering angle: uniform in [-1, 1]
    mu = 2.0 * rng.random(N) - 1.0

    # Compute scattered photon energy
    pNuSc = pNu / (1.0 + pNu * (1.0 - mu))

    # Random target values for inverse CDF sampling
    xi_phi = rng.random(N)

    # Run CUDA kernel (from your IC module)
    phiSc = run_CalcScPhi_kernel_CuPy(IC.CalcScPhi, xi_phi, pNu, mu, pNuSc)


    counts, bins = cp.histogram(phiSc, bins=5000, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    counts = counts/cp.max(counts)


    # --- Plot analytic PDF using mean μ and energies ---
    phi_vals = np.linspace(0, 2 * np.pi, 500)
    mu_avg = cp.mean(mu).item()
    epe = cp.mean(pNuSc / pNu).item()
    eep = cp.mean(pNu / pNuSc).item()
    pdf_vals = polarization_pdf(phi_vals, mu_avg, epe, eep)

    plt.plot(phi_vals, pdf_vals/max(pdf_vals), 'r-', label="Analytic Polarization PDF")
    plt.plot(bin_centers.get(), counts.get(), label="drawn")
    plt.xlabel("φ (radians)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Comparison: Kernel Sampled φ vs. Analytic PDF")
    plt.grid(True)
    plt.show()


def RunIsotropicICScatteringTest():
    fig, ax = plt.subplots()

    photon_energies = SD.GetPlanck(N_PHOTONS,cp.full(N_PHOTONS,THETA_R_PLANCK_CONST,dtype=DTYPE))
    photon_wave_vector = cp.concatenate([cp.ones((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS)],axis=1)
    photon_polarization_vector = IC.CalcRandomPolarization(photon_wave_vector)
    # print(photon_polarization_vector[0:5,:])
    # print(cp.sum(photon_polarization_vector[0:5,1:]*photon_wave_vector[0:5,1:],axis=1))
    electrons = cp.concatenate([ SD.GetElectronGamma(cp.full(N_PHOTONS,THETA_E,dtype=DTYPE))[:,None] ,SD.GetIsotropicDirection(N_PHOTONS)],axis=1)

    hist, bin_edges = cp.histogram(photon_energies, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_plot = hist*bin_centers**2
    hist_plot /= cp.max(hist_plot)
    ax.plot((bin_centers*NATURAL_TO_KEV).get(),hist_plot.get(),label="Isotropic, Thermal seed photons")

    photon_energies, photon_wave_vector, photon_polarization_vector = IC.CalcICScattering(photon_energies,photon_wave_vector,photon_polarization_vector,electrons)
    # print(cp.sum(photon_polarization_vector[0:5,1:]*photon_wave_vector[0:5,1:],axis=1))

    hist, bin_edges = cp.histogram(photon_energies, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_plot = hist*bin_centers**2
    hist_plot /= cp.max(hist_plot)
    ax.plot((bin_centers*NATURAL_TO_KEV).get(),hist_plot.get(),label="Scattered photons")


    ax.set_ylabel(r"Normalized $\nu F\nu$")
    ax.set_xlabel("Photon energies (keV)")
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend()
    plt.show()

    Q, U = IC.CalcStokes(photon_wave_vector,photon_polarization_vector)
    photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector)

    fig, ax = plt.subplots()
    I, bin_edges = cp.histogram(photon_theta, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    I = I.astype(DTYPE)
    # I /= cp.max(I)
    ax.plot(bin_centers.get(),I.get())
    plt.show()

    Q_binned,_ = cp.histogram(photon_theta, bins=200,weights=Q)
    U_binned,_ = cp.histogram(photon_theta, bins=200,weights=U)
    PD = cp.sqrt(Q_binned**2+U_binned**2)/I

    fig, ax = plt.subplots()
    ax.plot(bin_centers.get(),PD.get()*100.0)
    ax.set_ylabel("PD (%)")
    ax.set_xlabel("viewing angle (radians)")
    plt.show()

def RunBeamedIsotropicICScatteringTest():
# def BeamedIsotropicICScatteringTest():
    fig, ax = plt.subplots()

    photon_energies = SD.GetPlanck(N_PHOTONS,cp.full(N_PHOTONS,THETA_R_PLANCK_CONST,dtype=DTYPE))
    photon_wave_vector = cp.concatenate([cp.ones((N_PHOTONS,1),dtype=DTYPE),SD.GetIsotropicDirection(N_PHOTONS)],axis=1)



    hist, bin_edges = cp.histogram(photon_energies, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_plot = hist*bin_centers**2
    hist_plot /= cp.max(hist_plot)
    ax.plot((bin_centers*NATURAL_TO_KEV).get(),hist_plot.get(),label="Isotropic, Thermal seed photons")

    ##############################################################################################################

    emession_frame = cp.zeros((N_PHOTONS,4),dtype=DTYPE)
    emession_frame[:,0] = BULK_LORENTZ
    emession_frame[:,3] = 1.0

    photon_wave_vector = SC.GetLorentzTransform(photon_wave_vector,emession_frame)
    photon_energies *= photon_wave_vector[:,0]
    SC.NormalizeFourVector(photon_wave_vector)


    ##############################################################################################################
    photon_polarization_vector = IC.CalcRandomPolarization(photon_wave_vector)
    # print(photon_polarization_vector[0:5,:])
    # print(cp.sum(photon_polarization_vector[0:5,1:]*photon_wave_vector[0:5,1:],axis=1))
    electrons = cp.concatenate([ SD.GetElectronGamma(cp.full(N_PHOTONS,THETA_E,dtype=DTYPE))[:,None] ,SD.GetIsotropicDirection(N_PHOTONS)],axis=1)

    photon_energies, photon_wave_vector, photon_polarization_vector = IC.CalcICScattering(photon_energies,photon_wave_vector,photon_polarization_vector,electrons)
    # print(cp.sum(photon_polarization_vector[0:5,1:]*photon_wave_vector[0:5,1:],axis=1))


    Q, U = IC.CalcStokes(photon_wave_vector,photon_polarization_vector)


    #######################################################################################################################

    photon_wave_vector = SC.GetLorentzTransform(photon_wave_vector,emession_frame,BACKWARD)
    photon_energies *= photon_wave_vector[:,0]
    SC.NormalizeFourVector(photon_wave_vector)


    #######################################################################################################################
    # print(cp.min(photon_energies)*NATURAL_TO_KEV)
    hist, bin_edges = cp.histogram(photon_energies, bins=1000)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_plot = hist*bin_centers**2
    hist_plot /= cp.max(hist_plot)
    ax.plot((bin_centers*NATURAL_TO_KEV).get(),hist_plot.get(),label="Scattered photons")


    ax.set_ylabel(r"Normalized $\nu F\nu$")
    ax.set_xlabel("Photon energies (keV)")
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend()
    plt.show()

    photon_theta, photon_phi = SC.CalcAngularDirections(photon_wave_vector)

    fig, ax = plt.subplots()
    I, bin_edges = cp.histogram(photon_theta, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    I = I.astype(DTYPE)
    # I /= cp.max(I)
    ax.plot(bin_centers.get(),I.get())
    plt.show()

    Q_binned,_ = cp.histogram(photon_theta, bins=100,weights=Q)
    U_binned,_ = cp.histogram(photon_theta, bins=100,weights=U)
    Q_binned /= I
    U_binned /= I
    PD = cp.sqrt(Q_binned**2+U_binned**2)#/I

    fig, ax = plt.subplots()
    ax.plot(bin_centers.get(),PD.get()*100.0)
    ax.set_ylabel("PD (%)")
    ax.set_xlabel("viewing angle (radians)")
    ax.grid(True)
    plt.show()

    # print(U_binned[0:10])
    # print(Q_binned[0:10])

    PA = cp.arctan2(U_binned, Q_binned)
    # print(PA[PA<0])
    # PA[PA<0] += cp.pi

    PA = 0.5*(PA+cp.pi)

    # PA = 0.5*((cp.arctan2(U_binned, Q_binned) % (cp.pi))+cp.pi)
    fig, ax = plt.subplots()
    ax.plot(bin_centers.get(),PA.get())
    ax.set_ylabel("PA (radians)")
    ax.set_xlabel("viewing angle (radians)")
    ax.grid(True)
    plt.show()

# def RunBeamedIsotropicICScatteringTest():



    # CalcICScattering(photon_energies,photon_wave_vector,photon_polarization_vector,electrons)

################################################ Main #################################################################

# test_lorentz_transform_plot()

# test_isotropic_photon_lorentz_transform()

# RunKNCrossectionTest()

# RunScatterMuTest()

# test_phi_sampling_with_cupy_and_kernel()

# RunIsotropicICScatteringTest()

RunBeamedIsotropicICScatteringTest()
