import Shared.Distributions as sd


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



############################################################## CONST ##########################################################################


DTYPE = cp.float64
N_PHOTONS = 5000000
N_PHOTONS_Larger = 50000000

THETA_R_PAIR = 3.0 #Temperature for Pair Annihilation spectrum where Theta_r = (kT_e)/(m_e c**2)
JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e*scpc.c**2
THETA_R_PLANCK_CONST = 3.0e5*scpc.k/(scpc.m_e*scpc.c**2)

DISTRIBUTIONS = ["PairAnnihilation","Planck","Gamma","Isotropic"]

################################################################### FUNC ##########################################################################

def photon_number_density(f):
    # c0 = 2.0e40
    c1 = 0.045
    Theta_r = 3.0
    ff = f
    # print("exp(",( c1*ff*ff )/( Theta_r*Theta_r ),")")
    return np.exp(-( c1*ff*ff )/( Theta_r*Theta_r ))/ff

def RunPairAnnihilationTest(num_bins=1000):
    num_bins += 1  # keep original logic

    # --- Draw samples using rejection method ---
    photon_Energy = sd.GetPairAnnihilationRejectionMethod(N_PHOTONS_Larger, THETA_R_PAIR)
    photon_Energy = photon_Energy * scpc.m_e * (scpc.c**2) * JOULE_TO_EV  # convert to eV

    # --- Compute histogram using geometric bins ---
    bin_edges = np.geomspace(1.0e-7, 30.0, num_bins) * scpc.m_e * (scpc.c**2) * JOULE_TO_EV
    hist, _ = cp.histogram(cp.array(photon_Energy), bins=bin_edges, density=True)
    hist_np = cp.asnumpy(hist)
    bin_centers_np = cp.asnumpy(0.5 * (bin_edges[:-1] + bin_edges[1:]))

    del photon_Energy  # free GPU memory

    # --- Draw samples using grid/inverse method ---
    photon_Energy_Larger = sd.GetPairAnnihilation(N_PHOTONS_Larger, THETA_R_PAIR)
    photon_Energy_Larger = photon_Energy_Larger * scpc.m_e * (scpc.c**2) * JOULE_TO_EV

    # --- Compute histogram using same geometric bins ---
    hist2, _ = cp.histogram(cp.array(photon_Energy_Larger), bins=bin_edges, density=True)
    hist_np2 = cp.asnumpy(hist2)
    bin_centers_np2 = cp.asnumpy(0.5 * (bin_edges[:-1] + bin_edges[1:]))

    del photon_Energy_Larger  # free GPU memory

    # --- Theoretical curve evaluated at bin centers ---
    bin_centers_natural = bin_centers_np / (scpc.m_e * (scpc.c**2) * JOULE_TO_EV)
    theo = photon_number_density(bin_centers_natural)
    theo /= np.trapezoid(theo, bin_centers_np)  # normalize area under curve

    # --- Plotting ---
    plt.plot(bin_centers_np/1000.0, theo, label="Theoretical")
    plt.plot(bin_centers_np/1000.0, hist_np, '--', label=f"Drawn Rejection, $N_{{phot}}=$ {N_PHOTONS_Larger:.0e}")
    plt.plot(bin_centers_np2/1000.0, hist_np2, label=f"Drawn Inverse, $N_{{phot}}=$ {N_PHOTONS_Larger:.0e}")

    plt.xlabel('Photon energy (keV)')
    plt.ylabel('Probability density')
    plt.title(r'Comparison of photon number spectrum for $\Theta_{r} = $'+f'{THETA_R_PAIR:.2f}')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig("PairAnhiComparison.pdf")
    plt.show()


# def RunPairAnnihilationTest(num_bins = 100000):
#     num_bins += 1
#     # photon_Energy = sd.GetPairAnnihilation(N_PHOTONS,THETA_R_PAIR)
#     #
#     # photon_Energy = photon_Energy*scpc.m_e*(scpc.c**2)*JOULE_TO_EV
#     photon_Energy = sd.GetPairAnnihilationRejectionMethod(N_PHOTONS_Larger,THETA_R_PAIR)
#
#     photon_Energy = photon_Energy*scpc.m_e*(scpc.c**2)*JOULE_TO_EV
#     # print(photon_Energy[cp.where(photon_Energy<0.0)])
#
#
#     # Compute histogram using cupy.histogram
#     hist, bin_edges = cp.histogram(photon_Energy, bins=num_bins,density=True)
#
#     # Convert to numpy for plotting
#     hist_np = cp.asnumpy(hist)
#     bin_centers_np = cp.asnumpy(0.5 * (bin_edges[:-1] + bin_edges[1:]))
#
#     del photon_Energy
#
#
#     photon_Energy_Larger = sd.GetPairAnnihilation(N_PHOTONS_Larger,THETA_R_PAIR)
#
#     photon_Energy_Larger = photon_Energy_Larger*scpc.m_e*(scpc.c**2)*JOULE_TO_EV
#     # print(photon_Energy_Larger[cp.where(photon_Energy_Larger<0.0)])
#
#
#     # Compute histogram using cupy.histogram
#     hist2, bin_edges2 = cp.histogram(photon_Energy_Larger, bins=num_bins,density=True)
#
#     # Convert to numpy for plotting
#     hist_np2 = cp.asnumpy(hist2)
#     bin_centers_np2 = cp.asnumpy(0.5 * (bin_edges2[:-1] + bin_edges2[1:]))
#
#     del photon_Energy_Larger
#
#     # Adjust as needed
#
#
#     # Plot the histogram
#     # plt.bar(bin_edges_np[:-1], hist_np, width=bin_edges_np[1]-bin_edges_np[0])
#     theoretical_energies = np.geomspace(1.0e-7,30.0,1000)
#     theo = photon_number_density(theoretical_energies)
#     theoretical_energies *= scpc.m_e*(scpc.c**2)*JOULE_TO_EV
#     theo /= np.trapezoid(theo, theoretical_energies)  # normalize to unit area
#     # print(theo)
#     plt.plot(bin_centers_np/1000.0, hist_np,'--', label=r"Drawn Rejection, $N_{phot}=$ " + f"{(N_PHOTONS_Larger):.0e} ")#, label="Drawn")#
#     plt.plot(bin_centers_np2/1000.0,hist_np2, label=r"Drawn Inverse, $N_{phot}=$" + f" {N_PHOTONS_Larger:.0e} ")
#     plt.plot(theoretical_energies/1000.0,theo,label="Theoretical")#,label="Theoretical")#
#     plt.xlabel('Photon energy (keV)')
#     plt.ylabel('Number of photons')
#     plt.title(r'Comparison of photon number spectrum for $\Theta_{r} = $'+f'{THETA_R_PAIR:.2f}')
#     plt.yscale('log')
#     plt.legend()
#     plt.savefig("PairAnhiComparison.pdf")
#     plt.show()








def planck_spectrum_natural_cupy(E, Theta_r):
    """
    Computes the unnormalized Planck photon number distribution dN/dE
    using CuPy arrays. E and Theta_r are in units of m_e*c^2.

    Parameters
    ----------
    E : cupy.ndarray or float
        Photon energy in units of m_e*c^2
    Theta_r : float
        Temperature in units of m_e*c^2

    Returns
    -------
    cupy.ndarray or float
        Unnormalized photon number density per unit energy (dN/dE)
    """
    exponential = cp.exp(E / Theta_r)
    spectrum = cp.where(E > 0, E**2 / (exponential - 1.0), 0.0)
    spectrum = cp.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)
    return spectrum

def RunPlanckTest():
    Theta_r = cp.full(N_PHOTONS,THETA_R_PLANCK_CONST,dtype=DTYPE)
    photon_Energy = sd.GetPlanck(N_PHOTONS,Theta_r)



    num_bins = 1000  # Adjust as needed

    # Compute histogram using cupy.histogram
    hist, bin_edges = cp.histogram(photon_Energy, bins=num_bins)

    # Convert to numpy for plotting
    hist_np = cp.asnumpy(hist)
    bin_edges_np = cp.asnumpy(bin_edges)

    # Plot the histogram
    # plt.bar(bin_edges_np[:-1], hist_np, width=bin_edges_np[1]-bin_edges_np[0])
    theo = planck_spectrum_natural_cupy(bin_edges[:-1], THETA_R_PLANCK_CONST)
    # print(theo)
    plt.plot((JOULE_TO_EV*scpc.m_e*scpc.c**2)*bin_edges_np[:-1]/1000.0, hist_np/max(hist_np), label="Drawn")
    plt.plot((JOULE_TO_EV*scpc.m_e*scpc.c**2)*bin_edges_np[:-1]/1000.0,cp.asnumpy(theo/max(theo)),label="Theoretical")
    plt.xlabel('Photon energy (keV)')
    plt.ylabel('Number of photons')
    plt.title('Comparison of photon number spectrum for T_r = %.3f keV' % (THETA_R_PLANCK_CONST*JOULE_TO_KEV*M_EC2))
    plt.legend()
    plt.savefig("TComparison.pdf")
    plt.show()






def maxwell_juttner_theoretical(gamma, theta_e):
    """
    Theoretical Maxwell-Jüttner distribution (unnormalized) as a function of Lorentz factor gamma and theta_e = kT / m_e c^2.
    """
    return (gamma * cp.sqrt(-1.0+gamma**(2)) ) * cp.exp(-gamma / theta_e)

def RunGammaTest():
    """
    Compares sampled Lorentz factors to theoretical Maxwell-Jüttner distribution.

    Parameters:
    - draw_lorentz_factors_fn: function(theta_e, N) → CuPy array of Lorentz factors
    - theta_e_values: list of dimensionless temperatures (theta_e = kT / m_e c^2)
    - N: number of samples per theta_e
    - bins: number of histogram bins
    """
    theta_e_values = [0.008,0.017,0.06,0.169,0.675,1.4]
    bins = 1000
    colors = ['blue','red','green','orange','purple','teal']

    plt.figure(figsize=(10, 6))
    plt.grid(True)

    for i, theta_e in enumerate(theta_e_values):
        # Sample gamma values using your CuPy-based function


        # Histogram on GPU
        hist_vals, bin_edges = cp.histogram(sd.GetElectronGamma(cp.full(N_PHOTONS,theta_e,dtype=DTYPE)), bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_vals = hist_vals/cp.max(hist_vals)

        # Evaluate theoretical distribution at bin centers
        theoretical_vals = maxwell_juttner_theoretical(bin_centers, theta_e)
        theoretical_vals = theoretical_vals / cp.max(theoretical_vals)

        # Transfer to CPU for plotting
        plt.plot(bin_centers.get(), hist_vals.get(), 'x', markersize=2.0, label=f"Sampled (θₑ={theta_e})",color=colors[i])
        plt.plot(bin_centers.get(), theoretical_vals.get(), '--', label=f"Theory (θₑ={theta_e})",color=colors[i])

    plt.xlabel('Lorentz Factor γ')
    plt.ylabel('Probability Density')
    plt.title('Sampled vs. Theoretical Maxwell-Jüttner Distribution')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


###################################################### Isotropic directions #############################################################

def RunIsotropicTest(opening_angle = cp.pi):
    # Transfer to CPU for plotting
    directions_np = cp.asnumpy(sd.GetIsotropicDirection(N_PHOTONS,opening_angle))
    indices = cp.random.choice(N_PHOTONS,size=10000, replace=False)
    x, y, z = directions_np[indices.get(), 0], directions_np[indices.get(), 1], directions_np[indices.get(), 2]

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, alpha=0.5)

    # Set equal aspect ratio and axis limits
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Isotropic Direction Sampling')

    plt.show()


################################################### MAIN #####################################################

parser = argparse.ArgumentParser(description="Show plots testing drawing functions")

parser.add_argument("distribution", type=str)

args = parser.parse_args()

if args.distribution == DISTRIBUTIONS[0]:
    RunPairAnnihilationTest()
elif args.distribution == DISTRIBUTIONS[1]:
    RunPlanckTest()
elif args.distribution == DISTRIBUTIONS[2]:
    RunGammaTest()
elif args.distribution == DISTRIBUTIONS[3]:
    RunIsotropicTest()
    RunIsotropicTest(0.1)
else:
    print("No valid distribution selected. Please select from:")
    for dist in DISTRIBUTIONS:
        print("- "+dist)







