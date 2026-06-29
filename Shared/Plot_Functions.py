"""Plotting and visualization functions for Inverse Compton simulation results.

This module contains various plotting utilities to visualize energy spectrums,
polarizations, angular distributions, light curves, and top-down polar positions
of simulated photons.
"""

import Shared.Distributions as SD
import Shared.Core_IC as IC
import Shared.Common as SC


import  numpy as np
import  cupy as cp
import  scipy.constants   as scpc
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import matplotlib.ticker as mticker
import argparse
import h5py as h5
import  math as m
# from    numba             import cuda
# import  numba
# import  math as m
# from    numba.cuda        import libdevice as cudalib
# from    numba.cuda.random import create_xoroshiro128p_states as cStates
# from    numba.cuda.random import xoroshiro128p_uniform_float64 as uniform
# import  pickle


# import cupy as cp
# import matplotlib.pyplot as plt

############################################################## CONST ##########################################################################


DTYPE = cp.float64
N_PHOTONS = 4000000
N_PHOTONS_Larger = 100000000

THETA_R_PAIR = 3.0 #Temperature for Pair Annihilation spectrum where Theta_r = (kT_e)/(m_e c**2)
JOULE_TO_EV = (6.242e18)
JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e*scpc.c**2
NATURAL_TO_KEV = M_EC2*JOULE_TO_KEV#511.0#
SIGMA_T = 6.6524587158e-29  # Thomson cross-section in m^2
# THETA_R_PLANCK_CONST = 3.0e8*scpc.k/(scpc.m_e*scpc.c**2)
# THETA_E = 1.0
THETA_R_PLANCK_CONST = 500.0/(M_EC2*JOULE_TO_EV)
THETA_E = 1.4#50000.0/(M_EC2*JOULE_TO_EV)
THETA_J = 0.1
INITIAL_RADIUS = m.sqrt(1.0e21)
M_DOT   = 1.0e30 # kg/s
BULK_LORENTZ = 20.0
Z_0 = 1.0e7

################################################################################## Digitized Data #############################################################################################

################################## Vyas Pair Anni ##################################

Vyas_pai_105_x = np.array([1.0758928571428572, 1.2723214285714286, 1.4553571428571428, 1.6473214285714284, 1.8571428571428572, 2.0446428571428568, 2.174107142857143, 2.375, 2.602678571428571, 2.8035714285714284, 2.9732142857142856, 3.151785714285714, 3.361607142857143, 3.549107142857143, 3.7098214285714284, 3.892857142857143, 4.080357142857142, 4.25, 4.4017857142857135, 4.5892857142857135, 4.700892857142857, 4.8125, 4.9330357142857135, 5.013392857142857, 5.102678571428571, 5.160714285714286, 5.191964285714286])
Vyas_pai_105_y = np.array([-2.613573407202216, -2.4473684210526314, -2.3725761772853184, -2.24792243767313, -2.1398891966759, -2.090027700831025, -2.0069252077562325, -1.8988919667590025, -1.7409972299168972, -1.6329639889196672, -1.4916897506925206, -1.3337950138504153, -1.1592797783933513, -0.9930747922437666, -0.8268698060941828, -0.6855955678670353, -0.5609418282548475, -0.49445983379501346, -0.49445983379501346, -0.5692520775623269, -0.7188365650969528, -0.9432132963988913, -1.2340720221606647, -1.5914127423822713, -1.9404432132963985, -2.272853185595568, -2.5637119113573403])

Vyas_pai_175_x = np.array([1.0357142857142858, 1.1473214285714286, 1.28125, 1.4598214285714286, 1.5580357142857144, 1.7276785714285714, 1.9330357142857142, 2.125, 2.258928571428571, 2.4107142857142856, 2.5669642857142856, 2.674107142857143, 2.8125, 2.9955357142857144, 3.151785714285714, 3.267857142857143, 3.3794642857142856, 3.4598214285714284, 3.526785714285714, 3.571428571428571])
Vyas_pai_175_y = np.array([-4.624653739612189, -4.566481994459834, -4.483379501385041, -4.325484764542936, -4.217451523545706, -4.042936288088643, -3.8518005540166205, -3.6689750692520775, -3.4861495844875345, -3.3614958448753463, -3.2534626038781163, -3.1786703601108033, -3.095567867036011, -3.128808864265928, -3.236842105263158, -3.4529085872576175, -3.5941828254847645, -3.8850415512465375, -4.134349030470914, -4.408587257617729])

Vyas_pai_255_x = np.array([1.09375, 1.2142857142857142, 1.3660714285714286, 1.5446428571428572, 1.7142857142857142, 1.8392857142857142, 2.03125, 2.196428571428571, 2.3571428571428568, 2.4598214285714284, 2.571428571428571, 2.6517857142857144, 2.7410714285714284, 2.8080357142857144, 2.852678571428571, 2.875])
Vyas_pai_255_y = np.array([-5.696675900277008, -5.580332409972299, -5.405817174515235, -5.198060941828254, -5.023545706371191, -4.849030470914127, -4.740997229916897, -4.666204986149585, -4.6828254847645425, -4.757617728531856, -4.882271468144044, -5.0484764542936285, -5.198060941828254, -5.447368421052632, -5.646814404432133, -5.862880886426593])


################################# Vyas Mono Energetic #######################################

vyas_mono_055_x = np.array([19.27826178143417, 23.49853157267199, 28.942661247167518, 31.458467096143647, 36.77992672932109, 42.11476572391717, 51.87189650900193, 65.23484547824461, 79.5156271601822, 106.45089600223824, 144.0028345159894, 200.98674685430507, 239.933283688842, 283.4579799256692, 321.2087226324017, 391.5256155232898, 477.23581836325843, 581.7091329374364, 762.6985859023451, 1021.0564985595798, 1425.1026703029993, 1989.0354978101464, 2449.854385312311, 2924.578940787514])
vyas_mono_055_y = np.array([143.30125702369628, 332.5638199557432, 897.6871324473142, 1412.537544622754, 3651.741272548377, 7286.181745132274, 16909.27549681091, 31622.776601683792, 57876.19883491209, 87221.79096569099, 120572.98145138739, 152888.56344257703, 117998.09609496166, 68785.99123088068, 38403.87080444004, 17655.295872853927, 8474.713008888086, 4731.512589614803, 2475.9963492497727, 1268.0167779065703, 570.4925797046321, 220.673406908459, 117.99809609496178, 68.78599123088074])

vyas_mono_105_x = np.array([10.645089600223818, 13.111339374215643, 15.489779975481962, 18.110004241795057, 21.39521928721579, 25.808615404180753, 31.13240486761074, 39.15256155232898, 50.80218046913023, 70.90530557898195, 103.17504349513905, 159.81587227004837, 220.74526805465248, 274.7350234352208, 314.5846709614363, 356.4808453999243, 412.4626382901352, 546.457714104356, 731.5657987147581, 882.4728594001011, 1169.158262445337, 1381.2474625902555, 1868.5004753443568, 2399.33283688842])
vyas_mono_105_y = np.array([36.78112427301061, 97.16279515771058, 262.27083564681885, 649.3816315762114, 1295.686697517019, 3006.9416454654556, 6978.305848598663, 14855.08017172775, 26039.04187398692, 40973.21098135413, 68785.99123088068, 85359.13392913649, 81752.30379436494, 50845.20468600936, 28387.35964758755, 15179.239032493822, 7943.282347242814, 3349.654391578276, 1830.206106311055, 1090.184492385128, 582.9415347136074, 347.23620411959837, 152.8885634425769, 78.2978794189102])

vyas_mono_155_x = np.array([8.37677640068292, 10, 12.316795693236793, 15.013107289081734, 17.552698432008217, 23.013938607532253, 31.787944305861828, 41.24626382901352, 57.56798062171025, 83.76776400682924, 120.62795458049872, 153.29230761089096, 194.801707899231, 237.44641168265974, 280.51998106045033, 331.40723928616063, 383.4514701934921, 477.23581836325843, 593.9578904572273, 820.4028290795054, 1086.9237914057524, 1366.9310426441996, 1927.826178143418, 2349.8531572672014])
vyas_mono_155_y = np.array([59.13913935511893, 140.24100004218948, 378.551524925863, 878.516656407272, 2175.204034019522, 5503.322960851888, 13050.435363128214, 19247.52976905648, 29006.8119869315, 37583.74042884443, 45643.08599328049, 44668.35921509626, 36781.124273010595, 22387.21138568338, 12232.0711904993, 6540.711597277758, 3981.071705534969, 2271.1718672797642, 1382.3722273578996, 663.5520572797897, 305.05278902670256, 152.8885634425769, 83.53625469578262, 44.6683592150963])

vyas_mono_305_x = np.array([2.5808615404180753, 3.755438320351845, 5.133425218824524, 6.660846290809159, 9.010546478929266, 12.316795693236793, 16.83621037899174, 21.173461092214783, 27.188743668779452, 31.787944305861828, 39.56262211548852, 47.72358183632584, 61.92345841387512, 80.34842638354104, 106.45089600223824, 141.0331697796709, 192.78261781434182, 252.7638327613546, 321.2087226324017, 421.14765723917196])
vyas_mono_305_y = np.array([5.746124082964725, 18.040559400917857, 87.221790965691, 305.05278902670256, 937.2921937193711, 2175.204034019522, 3422.748588242436, 3573.7568434088284, 3072.557365267445, 2175.204034019522, 1021.8214144264955, 663.5520572797897, 354.8133892335753, 225.48881277351515, 110.59869434355589, 55.43065366083504, 26.6072505979881, 17.655295872853916, 10.29200527194428, 6.130557921498207])


################################################################################################################################################################################################




def PlotTheta(h5file,num_bins = 100):
    """Plots a histogram of polar angles (theta) of escaped photons.

    Args:
        h5file: Opened h5py.File object containing simulation datasets.
        num_bins: Number of histogram bins. Defaults to 100.
    """
    theta_photon = (cp.array(h5file['photon_theta'][:]))
    # energy_mask = (cp.array( h5file['photon_energies'][:])*NATURAL_TO_KEV) >= 1.0
    # theta_photon_mask = cp.logical_and(theta_photon*BULK_LORENTZ < 40.0,energy_mask)
    # theta_photon = theta_photon[theta_photon_mask]
    hist, edges = cp.histogram(theta_photon, bins=num_bins+1)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    # plt.plot(bin_centers.get()*BULK_LORENTZ,hist.get())
    plt.plot(bin_centers.get(),hist.get())
    plt.xlabel('Theta (radians)')
    plt.ylabel('Counts')
    # plt.title('Histogram of Photon Theta')
    plt.show()

def BB(epsi,theta):
    """Calculates the scaling factor for Compton scattering energy change.

    Args:
        epsi: Photon energy parameter.
        theta: Polar scattering angle.

    Returns:
        float: Computed scaling factor.
    """
    return 1.0/(1.0+ epsi*(1.0-np.cos(theta)))

def PlotThetaTest(h5file,output_path,num_bins = 100,theory_adapted = True):
    """Plots photon theta distribution and compares it with the analytical formula.

    Saves the output plot to the specified path.

    Args:
        h5file: Opened h5py.File object.
        output_path: Directory path where the output PDF will be saved.
        num_bins: Number of histogram bins. Defaults to 100.
        theory_adapted: If True, uses the adapted theoretical formula.
            Defaults to True.
    """
    theta_photon = (cp.array(h5file['photon_theta'][:]))
    hist, edges = cp.histogram(theta_photon, bins=num_bins+1,density = True)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    if theory_adapted:
        plt.plot(bin_centers.get(),(hist).get(),label=f"code")
    else:
        plt.plot(bin_centers.get(),0.5*(hist/cp.abs(2.0*cp.pi*cp.diff(cp.cos(edges)) )).get(),label=f"code")
    # print(cp.diff(cp.cos(edges)))
    # print(edges)

    theta_theoretical = np.arange(0.0,np.pi,0.01)

    # plt.plot(theta_theoretical,(1.0-0.5*(np.sin(theta_theoretical)**2))  ,label=f"theoretical")
    epsi = 1.0/NATURAL_TO_KEV
    if theory_adapted:
        plt.plot(theta_theoretical,(3.0/8.0)*(BB(epsi,theta_theoretical)**2)*(BB(epsi,theta_theoretical)+1.0/BB(epsi,theta_theoretical) - np.sin(theta_theoretical)**2)*np.sin(theta_theoretical)  ,label=f"theoretical")
    else:
        plt.plot(theta_theoretical,(BB(epsi,theta_theoretical)**2)*(BB(epsi,theta_theoretical)+1.0/BB(epsi,theta_theoretical) - np.sin(theta_theoretical)**2)  ,label=f"theoretical")



    plt.xlabel('Theta (radians)')
    plt.ylabel('Counts')
    plt.legend()
    # plt.title('Histogram of Photon Theta')
    if theory_adapted:
        plt.savefig(output_path + "Theta_distribution_theoretical_adapted.pdf")
    else:
        plt.savefig(output_path + "Theta_distribution_data_adapted.pdf")
    plt.show()

def PlotPhiTest(h5file,viewing_angles, output_path,num_bins = 100):
    """Plots azimuthal angle (phi) distributions for various viewing angles.

    Args:
        h5file: Opened h5py.File object.
        viewing_angles: List/array of viewing angles (theta) to filter by.
        output_path: Path to save the resulting plot.
        num_bins: Number of histogram bins. Defaults to 100.
    """
    for viewing_theta in viewing_angles:
        phi_photon = (cp.array(h5file['photon_phi'][:]))
        theta_photon = (cp.array(h5file['photon_theta'][:]))
        theta_mask = cp.logical_and( theta_photon > viewing_theta-0.01  , theta_photon < viewing_theta+0.01 )
        phi_photon = phi_photon[theta_mask]
        hist, edges = cp.histogram(phi_photon, bins=num_bins+1,density=True)
        # hist = hist.astype(DTYPE)
        # hist /= cp.max(hist)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        plt.plot(bin_centers.get(),hist.get(), label=f"θ={viewing_theta:.3f}")
    plt.xlabel(r'$\phi$')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(output_path + "counts_phi.pdf")
    # plt.title('Histogram of Photon Theta')
    plt.show()


def PlotPolarizationVsThetaTest(h5file, output_path, num_bins = 50):
    """Plots polarization degree and angle as a function of theta for verification.

    Args:
        h5file: Opened h5py.File object.
        output_path: Path to save the resulting plot.
        num_bins: Number of histogram bins. Defaults to 50.
    """
    fig, (ax_pd, ax_pa) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios':[4,1],'hspace':0.0})


    theta_photon = (cp.array(h5file['photon_theta'][:]))


    Q = cp.array(h5file['Q'][:])
    U = cp.array(h5file['U'][:])
    # print(Q)
    # print(U)

    # Histograms
    num_bins += 1
    I_hist, edges = cp.histogram(theta_photon, bins=num_bins)
    Q_hist, _ = cp.histogram(theta_photon, bins=num_bins, weights=Q)
    U_hist, _ = cp.histogram(theta_photon, bins=num_bins, weights=U)


    bin_centers = (edges[1:]+edges[:-1])/2.0

    # Avoid divide-by-zero
    mask = I_hist > 0
    pd = cp.zeros_like(I_hist, dtype=DTYPE)
    pa = cp.zeros_like(I_hist, dtype=DTYPE)

    pd[mask] = cp.sqrt(Q_hist[mask]**2 + U_hist[mask]**2) / I_hist[mask]
    # pd[mask] = Q_hist[mask] / I_hist[mask]
    pa[mask] = 0.5 * (cp.arctan2(U_hist[mask], Q_hist[mask]) )/ cp.pi  # in units of π
    pa += 0.5


    # --- Error bars ---
    err = cp.zeros_like(I_hist, dtype=DTYPE)
    err[mask] = 1.0 / cp.sqrt(I_hist[mask])

    # Apply filter: remove bins where error > 0.3
    good = (mask & (err <= 0.3))


    # Plot

    eb = ax_pd.errorbar(bin_centers[good].get(), pd[good].get(), yerr=err[good].get(), fmt='o')
    # ax_pa.errorbar(bin_centers[good].get()*BULK_LORENTZ, pa[good].get(), yerr=(err[good]/(2.0*pd[good])).get(), fmt='o', color = eb[0].get_color())
    ax_pa.errorbar(bin_centers[good].get(), pa[good].get(), yerr=(err[good]/(2.0)).get(), fmt='o', color = eb[0].get_color())


    # ax2.plot(bin_centers*NATURAL_TO_KEV,I_hist)
    ax_pd.plot(bin_centers[good].get(), pd[good].get(), color = eb[0].get_color())
    ax_pa.plot(bin_centers[good].get(), pa[good].get(), color = eb[0].get_color())

    # Basic shape / unique checks
    print("shapes:", theta_photon.shape, Q.shape, U.shape)
    print("Q unique (first 5):", cp.unique(Q)[:5].get())
    print("U unique (first 5):", cp.unique(U)[:5].get())

    # single-photon PD and global PD from raw arrays
    single_pd = cp.sqrt(Q[0]**2 + U[0]**2).get()
    global_Q = cp.sum(Q).get()
    global_U = cp.sum(U).get()
    N = theta_photon.size
    print("single PD:", single_pd)
    print("global_PD_from_sums:", m.sqrt(global_Q**2 + global_U**2) / N)
    print("global_Q:", global_Q )
    print("global_U:", global_U )

    print("I_hist sum:", int(I_hist.sum().get()))
    print("Q_hist sum:", float(Q_hist.sum().get()))
    print("U_hist sum:", float(U_hist.sum().get()))


    # ax_pd.set_xscale('log')
    # ax_pa.set_xscale('log')
    #
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')


    # Define tick positions in units of π (e.g., from -0.5 to 0.5)
    ticks = np.array([ 0, 0.25, 0.5,0.75,1.0])

    # Set ticks and labels like -½π, -¼π, 0, ¼π, ½π
    ax_pa.set_yticks(ticks)
    ax_pa.set_yticklabels([r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$",r"$\frac{3}{4}\pi$", r" "])


    ax_pd.set_ylabel("Polarization Degree")
    ax_pa.set_ylabel("Polarization Angle (× π)")
    ax_pa.set_xlabel(r"$\theta$")
    ax_pa.set_ylim(0.0,1.0)
    ax_pd.set_ylim(bottom = 0.0)

    # ax_pd.legend()
    plt.tight_layout()
    plt.savefig(output_path + "polarization_theta.pdf")
    plt.show()

# def PlotEnergy(h5file,viewing_angles,output_path):
#     fig, ax = plt.subplots()
#     for viewing_angle in viewing_angles:
#         photons_in_viewing_window = np.logical_and(h5file['photon_theta'][:] > viewing_angle-0.01,h5file['photon_theta'][:] < viewing_angle+0.01)
#         photon_energies = h5file['photon_energies'][photons_in_viewing_window]
#         hist, edges = np.histogram(photon_energies, bins=1001)
#         bin_centers = (edges[:-1] + edges[1:]) / 2.0
#         ax.plot(bin_centers*NATURAL_TO_KEV,hist*(bin_centers*NATURAL_TO_KEV)**2)
#     plt.show()





def PlotEnergy(h5file, viewing_angles, output_path, num_bins = 500, figname="energy_sed.pdf",plot_backscatter_cork_pair_thoeretical=False,plot_backscatter_cork_mono_thoeretical=False):
    """Plots the energy spectrum (SED) of escaped photons for different viewing angles.

    Args:
        h5file: Opened h5py.File object.
        viewing_angles: List/array of viewing angles (theta) to filter by.
        output_path: Path to save the resulting plot.
        num_bins: Number of histogram bins. Defaults to 500.
        figname: Output file name. Defaults to "energy_sed.pdf".
        plot_backscatter_cork_pair_thoeretical: If True, plots theoretical pair spectrum.
            Defaults to False.
        plot_backscatter_cork_mono_thoeretical: If True, plots theoretical monoenergetic spectrum.
            Defaults to False.
    """
    fig, ax = plt.subplots()
    print("1.0e8*scpc.k/M_EC2:",1.0e8*scpc.k/M_EC2)

    # Energy binning parameters
    # E_min = np.min(h5file['photon_energies'][:]*NATURAL_TO_KEV)
    # E_max = np.max(h5file['photon_energies'][:]*NATURAL_TO_KEV)

    # Log-spaced bin edges
    # edges = np.geomspace(1.0, 5.0e5, num_bins + 1)
    # # edges = np.linspace(E_min, E_max, num_bins + 1)
    # bin_centers = (edges[:-1] + edges[1:])/2.0  # geometric mean for bin center
    # bin_widths = np.diff(edges)

    for viewing_angle in viewing_angles:
        # Select photons in viewing window
        photons_in_viewing_window = np.logical_and(
            h5file['photon_theta'][:] > viewing_angle - 0.01,
            h5file['photon_theta'][:] < viewing_angle + 0.01
        )
        # photons_in_viewing_window = np.logical_and(
        # np.logical_and(
        #     np.array(h5file['photon_theta'][:]) > viewing_angle - 0.01,
        #     np.array(h5file['photon_theta'][:]) < viewing_angle + 0.01
        # ),
        # np.logical_and(
        #     np.array(h5file['photon_phi'][:]) > np.pi - 0.01/m.sin(viewing_angle),
        #     np.array(h5file['photon_phi'][:]) < np.pi + 0.01/m.sin(viewing_angle)
        # )
        # )
        # photons_in_viewing_window = np.ones(h5file['photon_theta'][:].size,dtype=np.bool)
        photon_energies = h5file['photon_energies'][photons_in_viewing_window]*NATURAL_TO_KEV
        edges = np.geomspace(np.min(photon_energies), np.max(photon_energies), num_bins + 1)
        # edges = np.linspace(np.min(photon_energies), np.max(photon_energies), num_bins + 1)
        # edges = np.linspace(E_min, E_max, num_bins + 1)
        bin_centers = (edges[:-1] + edges[1:])/2.0  # geometric mean for bin center
        bin_widths = np.diff(edges)*np.sin(viewing_angle)

        # Histogram normalized for variable bin widths
        hist, _ = np.histogram(photon_energies, bins=edges,weights=photon_energies**2)
        hist_density = hist / (bin_widths)# counts per unit energy
        err, _ = np.histogram(photon_energies, bins=edges,weights=photon_energies**4)
        err = np.sqrt(err)
        err /= (bin_widths)

        # peak_idx = np.argmax(hist_density)
        # E_peak = 330.0#bin_centers[peak_idx]
        #
        #
        # print("E_peak: ",E_peak)
        # if viewing_angle == 0.105:
        #     print("E_peak/theoretical_peak: ",E_peak/30000.0)
        # if viewing_angle == 0.175:
        #     print("E_peak/theoretical_peak: ",E_peak/1000.0)
        # if viewing_angle == 0.255:
        #     print("E_peak/theoretical_peak: ",E_peak/400.0)
        # # print("E_peak/theoretical_peak: ",E_peak/30)
        # beta_mask = photon_energies >= E_peak
        #
        # beta = len(photon_energies[beta_mask])/(np.sum( np.log( photon_energies[beta_mask]/E_peak  )  ) )
        # print("beta",beta)

        # Convert to νFν
        nuFnu = hist_density #* (bin_centers)**2

        # line1, = ax.plot(bin_centers , nuFnu, label=f"θ={viewing_angle:.3f}")
        line1 = ax.errorbar(bin_centers , nuFnu, yerr=err, label=f"θ={viewing_angle:.3f}").lines[0]
        if plot_backscatter_cork_mono_thoeretical:
            if viewing_angle == 0.055:
                ax.plot(vyas_mono_055_x , max(nuFnu)*vyas_mono_055_y/max(vyas_mono_055_y), '--', color=line1.get_color(), label=f"Vyas θ={viewing_angle:.3f}")
            if viewing_angle == 0.105:
                ax.plot(vyas_mono_105_x , max(nuFnu)*vyas_mono_105_y/max(vyas_mono_105_y), '--', color=line1.get_color(), label=f"Vyas θ={viewing_angle:.3f}")
            if viewing_angle == 0.155:
                ax.plot(vyas_mono_155_x , max(nuFnu)*vyas_mono_155_y/max(vyas_mono_155_y), '--', color=line1.get_color(), label=f"Vyas θ={viewing_angle:.3f}")
            if viewing_angle == 0.305:
                ax.plot(vyas_mono_305_x , max(nuFnu)*vyas_mono_305_y/max(vyas_mono_305_y), '--', color=line1.get_color(), label=f"Vyas θ={viewing_angle:.3f}")

        if plot_backscatter_cork_pair_thoeretical:
            if viewing_angle == 0.105:
                ax.plot(10.0**Vyas_pai_105_x , max(nuFnu)*(10.0**Vyas_pai_105_y)/max(10.0**Vyas_pai_105_y), '--', color=line1.get_color(), label=f"Vyas θ=0.105")
            if viewing_angle == 0.175:
                ax.plot(10.0**Vyas_pai_175_x , max(nuFnu)*(10.0**Vyas_pai_175_y)/max(10.0**Vyas_pai_175_y), '--', color=line1.get_color(), label=f"Vyas θ=0.175")
            if viewing_angle == 0.255:
                ax.plot(10.0**Vyas_pai_255_x , max(nuFnu)*(10.0**Vyas_pai_255_y)/max(10.0**Vyas_pai_255_y), '--', color=line1.get_color(), label=f"Vyas θ=0.255")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel(r"$\nu F_\nu$ (arbitrary units)")
    if plot_backscatter_cork_pair_thoeretical:
        ax.set_xlim(1.0e0,2.0e5)
        ax.set_ylim(2.450e3,7.5e11)
    if plot_backscatter_cork_mono_thoeretical:
        ax.set_xlim(1.1,3.6e3)
        ax.set_ylim(1.40e5,1.6e10)
    ax.legend()

    # Save and show
    fig.savefig(output_path + figname, bbox_inches='tight')
    plt.show()

def PlotEnergyLimitedScatterig(h5file, viewing_angles, output_path, num_bins = 500, figname="energy_sed_limited_scattering.pdf",plot_backscatter_cork_pair_thoeretical=False,plot_backscatter_cork_mono_thoeretical=False):
    """Plots energy spectrum (SED) for photons under limited scattering regimes.

    Args:
        h5file: Opened h5py.File object.
        viewing_angles: List/array of viewing angles (theta) to filter by.
        output_path: Path to save the resulting plot.
        num_bins: Number of histogram bins. Defaults to 500.
        figname: Output file name. Defaults to "energy_sed_limited_scattering.pdf".
        plot_backscatter_cork_pair_thoeretical: If True, plots theoretical pair spectrum.
            Defaults to False.
        plot_backscatter_cork_mono_thoeretical: If True, plots theoretical monoenergetic spectrum.
            Defaults to False.
    """
    fig, ax = plt.subplots()
    print("1.0e8*scpc.k/M_EC2:",1.0e8*scpc.k/M_EC2)

    # Energy binning parameters
    # E_min = np.min(h5file['photon_energies'][:]*NATURAL_TO_KEV)
    # E_max = np.max(h5file['photon_energies'][:]*NATURAL_TO_KEV)

    # Log-spaced bin edges
    # edges = np.geomspace(1.0, 5.0e5, num_bins + 1)
    # # edges = np.linspace(E_min, E_max, num_bins + 1)
    # bin_centers = (edges[:-1] + edges[1:])/2.0  # geometric mean for bin center
    # bin_widths = np.diff(edges)
    for viewing_angle in viewing_angles:
        scale = 0.0
        for scatter_count in [1,2,4,10]:#,12,16,20]:
            # Select photons in viewing window
            if scatter_count == 0:
                photons_in_viewing_window = np.logical_and(
                h5file['photon_theta'][:] > viewing_angle - 0.01,
                h5file['photon_theta'][:] < viewing_angle + 0.01
            )
            else:
                photons_in_viewing_window = np.logical_and(np.logical_and(
                    h5file['photon_theta'][:] > viewing_angle - 0.01,
                    h5file['photon_theta'][:] < viewing_angle + 0.01
                ),  np.array(h5file['final_iteration'][:]) == scatter_count )
                # photons_in_viewing_window = np.logical_and(np.logical_and(
                # np.logical_and(
                #     np.array(h5file['photon_theta'][:]) > viewing_angle - 0.01,
                #     np.array(h5file['photon_theta'][:]) < viewing_angle + 0.01
                # ),
                # np.logical_and(
                #     np.array(h5file['photon_phi'][:]) > np.pi - 0.01/m.sin(viewing_angle),
                #     np.array(h5file['photon_phi'][:]) < np.pi + 0.01/m.sin(viewing_angle)
                # )
                # ),  np.array(h5file['final_iteration'][:]) == scatter_count )
            # photons_in_viewing_window = np.ones(h5file['photon_theta'][:].size,dtype=np.bool)
            photon_energies = h5file['photon_energies'][photons_in_viewing_window]*NATURAL_TO_KEV
            edges = np.geomspace(np.min(photon_energies), np.max(photon_energies), num_bins + 1)
            # edges = np.linspace(np.min(photon_energies), np.max(photon_energies), num_bins + 1)
            # edges = np.linspace(E_min, E_max, num_bins + 1)
            bin_centers = np.sqrt(edges[:-1]*edges[1:])#(edges[:-1] + edges[1:])/2.0  # geometric mean for bin center
            bin_widths = np.diff(edges)#*(0.01*0.01*np.sin(viewing_angle))

            # Histogram normalized for variable bin widths
            hist, _ = np.histogram(photon_energies, bins=edges,weights=photon_energies**2)
            hist_density = hist / (bin_widths)# counts per unit energy

            # Convert to νFν
            nuFnu = hist_density #* (bin_centers)**2


            line1, = ax.plot(bin_centers , nuFnu, label=f"θ={viewing_angle:.3f}, l={scatter_count}")
            if scatter_count == 1:
                scale = max(nuFnu)
                colour = line1.get_color()
        if plot_backscatter_cork_mono_thoeretical:
            if viewing_angle == 0.055:
                ax.plot(vyas_mono_055_x , scale*vyas_mono_055_y/max(vyas_mono_055_y), '--', color=colour, label=f"Vyas θ={viewing_angle:.3f}")
        #     if viewing_angle == 0.105:
        #         ax.plot(vyas_mono_105_x , max(nuFnu)*vyas_mono_105_y/max(vyas_mono_105_y), '--', color=line1.get_color(), label=f"Vyas θ={viewing_angle:.3f}")
        #     if viewing_angle == 0.155:
        #         ax.plot(vyas_mono_155_x , max(nuFnu)*vyas_mono_155_y/max(vyas_mono_155_y), '--', color=line1.get_color(), label=f"Vyas θ={viewing_angle:.3f}")
        #     if viewing_angle == 0.305:
        #         ax.plot(vyas_mono_305_x , max(nuFnu)*vyas_mono_305_y/max(vyas_mono_305_y), '--', color=line1.get_color(), label=f"Vyas θ={viewing_angle:.3f}")
        #
        if plot_backscatter_cork_pair_thoeretical:
            if viewing_angle == 0.105:
                ax.plot(10.0**Vyas_pai_105_x , max(nuFnu)*(10.0**Vyas_pai_105_y)/max(10.0**Vyas_pai_105_y), label=f"Vyas θ=0.105")
            if viewing_angle == 0.175:
                ax.plot(10.0**Vyas_pai_175_x , max(nuFnu)*(10.0**Vyas_pai_175_y)/max(10.0**Vyas_pai_175_y), label=f"Vyas θ=0.175")
            if viewing_angle == 0.255:
                ax.plot(10.0**Vyas_pai_255_x , max(nuFnu)*(10.0**Vyas_pai_255_y)/max(10.0**Vyas_pai_255_y), label=f"Vyas θ=0.255")

    if plot_backscatter_cork_pair_thoeretical:
        photon_energies  = (SD.GetPairAnnihilationRejectionMethod(N_PHOTONS, 3.0,a=1.0e-3,b=30.0)*NATURAL_TO_KEV).get()
        edges = np.geomspace(np.min(photon_energies), np.max(photon_energies), num_bins + 1)
        # edges = np.linspace(E_min, E_max, num_bins + 1)
        bin_centers = np.sqrt(edges[:-1]*edges[1:])#(edges[:-1] + edges[1:])/2.0  # geometric mean for bin center
        bin_widths = np.diff(edges)

        # Histogram normalized for variable bin widths
        hist, _ = np.histogram(photon_energies, bins=edges,weights=photon_energies**2)
        hist_density = hist / (bin_widths)# counts per unit energy

        # Convert to νFν
        nuFnu = hist_density #* (bin_centers)**2

        line1, = ax.plot(bin_centers , nuFnu, label=f"θ={viewing_angle:.3f}, l=0")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel(r"$\nu F_\nu$ (arbitrary units)")
    ax.legend()

    # Save and show
    fig.savefig(output_path + figname, bbox_inches='tight')
    plt.show()



# def PlotPolarizationVsEnergy(h5file, viewing_angles, output_path, num_bins = 100):
#     fig, (ax_pd, ax_pa) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios':[4,1],'hspace':0.0})
#     # fig2 ,ax2 = plt.subplots()
#
#
#     # Energy binning parameters
#     # E_min = np.min(h5file['photon_energies'][:])
#     # E_max = np.max(h5file['photon_energies'][:])
#     #
#     # # Log-spaced bin edges
#     # # bins = np.geomspace(E_min, E_max, num_bins + 1)
#     # bins = np.linspace(E_min,E_max,501)
#     # bin_centers = np.sqrt(bins[:-1] + bins[1:])/2.0  # geometric mean for bin center
#     # bin_widths = np.diff(bins)
#
#     for viewing_angle in viewing_angles:
#         # Select photons in viewing window
#         photons_in_viewing_window = np.logical_and(
#             h5file['photon_theta'][:] > viewing_angle - 0.001,
#             h5file['photon_theta'][:] < viewing_angle + 0.001
#         )
#
#         energies = h5file['photon_energies'][photons_in_viewing_window]
#         Q = h5file['Q'][photons_in_viewing_window]
#         U = h5file['U'][photons_in_viewing_window]
#
#         bins = np.linspace(np.min(energies),np.max(energies),num_bins+1)
#         bin_centers = (bins[:-1] + bins[1:])/2.0  # geometric mean for bin center
#         bin_widths = np.diff(bins)
#
#         # Logarithmic bins in energy
#         # bins = np.logspace(np.log10(energies.min()), np.log10(energies.max()), num_bins)
#         # bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric centers
#
#         # Histograms
#         I_hist, _ = np.histogram(energies, bins=bins)
#         Q_hist, _ = np.histogram(energies, bins=bins, weights=Q)
#         U_hist, _ = np.histogram(energies, bins=bins, weights=U)
#
#         # Avoid divide-by-zero
#         mask = I_hist > 0
#         pd = np.zeros_like(I_hist, dtype=float)
#         pa = np.zeros_like(I_hist, dtype=float)
#
#         pd[mask] = np.sqrt(Q_hist[mask]**2 + U_hist[mask]**2) / I_hist[mask]
#         pa[mask] = 0.5 * (np.arctan2(U_hist[mask], Q_hist[mask]) )/ np.pi  # in units of π
#         pa += 0.5
#
#         # Plot
#         # ax2.plot(bin_centers*NATURAL_TO_KEV,I_hist)
#         ax_pd.plot(bin_centers*NATURAL_TO_KEV, pd, label=f"θ = {viewing_angle:.3f} rad")
#         ax_pa.plot(bin_centers*NATURAL_TO_KEV, pa, label=f"θ = {viewing_angle:.3f} rad")
#
#     ax_pd.set_xscale('log')
#     ax_pa.set_xscale('log')
#
#     # ax2.set_xscale('log')
#     # ax2.set_yscale('log')
#
#
#     # Define tick positions in units of π (e.g., from -0.5 to 0.5)
#     ticks = np.array([ 0, 0.25, 0.5,0.75,1.0])
#
#     # Set ticks and labels like -½π, -¼π, 0, ¼π, ½π
#     ax_pa.set_yticks(ticks)
#     ax_pa.set_yticklabels([r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$",r"$\frac{4}{3}\pi$", r"$\pi$"])
#
#
#     ax_pd.set_ylabel("Polarization Degree")
#     ax_pa.set_ylabel("Polarization Angle (× π)")
#     ax_pa.set_xlabel("Energy (keV)")
#
#     ax_pd.legend()
#     plt.tight_layout()
#     plt.savefig(output_path + "polarization_energy.pdf")
#     plt.show()

def PlotPolarizationVsEnergy(h5file, viewing_angles, output_path, num_bins = 100):
    """Plots polarization degree (PD) as a function of photon energy.

    Args:
        h5file: Opened h5py.File object.
        viewing_angles: List/array of viewing angles (theta) to filter by.
        output_path: Path to save the resulting plot.
        num_bins: Number of histogram bins. Defaults to 100.
    """
    fig, (ax_pd, ax_pa) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios':[4,1],'hspace':0.0})

    for viewing_angle in viewing_angles:
        # Select photons in viewing window
        photons_in_viewing_window = np.logical_and(
            h5file['photon_theta'][:] > viewing_angle - 0.001,
            h5file['photon_theta'][:] < viewing_angle + 0.001
        )
        # photons_in_viewing_window = np.logical_and(
        # np.logical_and(
        #     np.array(h5file['photon_theta'][:]) > viewing_angle - 0.01,
        #     np.array(h5file['photon_theta'][:]) < viewing_angle + 0.01
        # ),
        # np.logical_and(
        #     np.array(h5file['photon_phi'][:]) > np.pi - 0.01/m.sin(viewing_angle),
        #     np.array(h5file['photon_phi'][:]) < np.pi + 0.01/m.sin(viewing_angle)
        # )
        # )

        energies = h5file['photon_energies'][photons_in_viewing_window]
        Q = h5file['Q'][photons_in_viewing_window]
        U = h5file['U'][photons_in_viewing_window]

        # bins = np.linspace(np.min(energies),np.max(energies),num_bins+1)
        bins = np.geomspace(np.min(energies),np.max(energies),num_bins+1)
        bin_centers = (bins[:-1] + bins[1:])/2.0
        bin_widths = np.diff(bins)

        # Histograms
        I_hist, _ = np.histogram(energies, bins=bins)
        Q_hist, _ = np.histogram(energies, bins=bins, weights=Q)
        U_hist, _ = np.histogram(energies, bins=bins, weights=U)

        # i_hist /= bin_widths

        # Avoid divide-by-zero
        mask = I_hist > 0
        pd = np.zeros_like(I_hist, dtype=float)
        pa = np.zeros_like(I_hist, dtype=float)

        pd[mask] = np.sqrt(Q_hist[mask]**2 + U_hist[mask]**2) / I_hist[mask]
        pa[mask] = 0.5 * (np.arctan2(U_hist[mask], Q_hist[mask])) / np.pi
        pa += 0.5

        # pa = 0.5*np.unwrap(2.0*cp.pi*pa)
        # pa /= cp.pi
        # pa[pa<-0.0] += 1.0
        # pa[pa>1.0] += -1.0
        # pa = cp.mod(pa,cp.pi)

        # Error bars: 1/sqrt(I_hist)
        errors = np.zeros_like(I_hist, dtype=float)
        errors[mask] = 1.0 / np.sqrt(I_hist[mask])

        # Apply filter: exclude points where error > 0.3
        valid = mask & (errors <= 0.1)

        # Plot with error bars
        for element in range(0,len(pa)):
            if pa[element] >0.9:
                pa[element] = pa[element]-1.0


        eb = ax_pd.errorbar(bin_centers[valid]*NATURAL_TO_KEV, pd[valid], yerr=errors[valid], fmt='o', label=f"θ = {viewing_angle:.3f} rad")
        ax_pa.errorbar(bin_centers[valid]*NATURAL_TO_KEV, pa[valid], yerr=errors[valid]/(2.0*np.pi*pd[valid]), fmt='o', color = eb[0].get_color(), label=f"θ = {viewing_angle:.3f} rad")


        ax_pa.plot(bin_centers[valid]*NATURAL_TO_KEV, pa[valid], color = eb[0].get_color())
        ax_pd.plot(bin_centers[valid]*NATURAL_TO_KEV, pd[valid], color = eb[0].get_color())

    ax_pd.set_xscale('log')
    ax_pa.set_xscale('log')

    # Define tick positions in units of π
    ticks = np.array([0, 0.25, 0.5, 0.75, 1.0])
    ax_pa.set_yticks(ticks)
    ax_pa.set_yticklabels([r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$"])

    ax_pd.set_ylabel(r"$\Pi$")
    ax_pd.set_ylim(bottom = -0.01)
    ax_pa.set_ylabel(r"$\chi$")
    ax_pa.set_xlabel("Energy (keV)")
    ax_pa.set_ylim(-0.10,1.0)

    ax_pd.legend()
    plt.tight_layout()
    plt.savefig(output_path + "polarization_energy.pdf")
    plt.show()


# def PlotTopDownPosition(h5file, viewing_angle, output_path):
#         # Select photons in viewing window
#     photons_in_viewing_window = cp.logical_and(cp.logical_and(
#         cp.array(h5file['photon_theta'][:]) > viewing_angle - 0.01,
#         cp.array(h5file['photon_theta'][:]) < viewing_angle + 0.01
#     ),cp.logical_and(
#         cp.array(h5file['photon_phi'][:]) > cp.pi - 0.01/m.sin(viewing_angle),
#         cp.array(h5file['photon_phi'][:]) < cp.pi + 0.01/m.sin(viewing_angle)
#     ))
#     print("done extracting viewing window")
#
#     fig, ax = plt.subplots()
#     x = cp.array(h5file['photon_position'][:,1])
#     print("Done loading x")
#     y = cp.array(h5file['photon_position'][:,2])
#     print("Done loading y")
#     x = x[photons_in_viewing_window]
#     y = y[photons_in_viewing_window]
#     # print(len(x))
#     indices = SC.sample_subset(x, min([10000,len(x)]))
#
#     ax.scatter(x[indices].get(),y[indices].get())
#     plt.show()


def PlotTopDownPositionPolar(h5file, viewing_angle, output_path):
    """Plots the 2D spatial positions of escaped photons from a top-down polar perspective.

    Args:
        h5file: Opened h5py.File object.
        viewing_angle: Target viewing angle (theta).
        output_path: Path to save the resulting plot.
    """
    # Select photons in viewing window
    photons_in_viewing_window = cp.logical_and(
        cp.logical_and(
            cp.array(h5file['photon_theta'][:]) > viewing_angle - 0.001,
            cp.array(h5file['photon_theta'][:]) < viewing_angle + 0.001
        ),
        cp.logical_and(
            cp.array(h5file['photon_phi'][:]) > cp.pi - 0.001/m.sin(viewing_angle),
            cp.array(h5file['photon_phi'][:]) < cp.pi + 0.001/m.sin(viewing_angle)
        )
    )
    print("Done extracting viewing window")

    # Load positions from photon position four-vector
    x = cp.array(h5file['photon_position'][:, 1])
    y = cp.array(h5file['photon_position'][:, 2])
    z = cp.array(h5file['photon_position'][:, 3])
    print("Done loading positions")

    # Apply viewing window mask
    x = x[photons_in_viewing_window]
    y = y[photons_in_viewing_window]
    z = z[photons_in_viewing_window]

    # Compute spherical coordinates from position
    r_cartesian = cp.sqrt(x**2 + y**2 + z**2)
    theta_pos = cp.arccos(z / r_cartesian)       # Polar angle from z-axis
    phi_pos = cp.arctan2(y, x)                   # Azimuthal angle from x-axis

    # Random subset for plotting
    indices = SC.sample_subset(x, min([10000, len(x)]))

    # Polar plot: angle = phi, radius = theta
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.scatter(phi_pos[indices].get(), theta_pos[indices].get(), s=1)
    # ax.set_theta_zero_location("N")   # Optional: zero at top
    # ax.set_theta_direction(-1)        # Optional: clockwise
    ax.set_rlabel_position(225)       # Optional: radial labels position

    plt.savefig(output_path + "topdown_position_polar.pdf", bbox_inches="tight")
    plt.show()


def PlotTopDownPositionPolarHeatmap(h5file, viewing_angle, output_path, n_phi_bins=200, n_theta_bins=100,theta_j = 0.1):
    """Plots a 2D polar heatmap showing the spatial distribution of escaped photons.

    Args:
        h5file: Opened h5py.File object.
        viewing_angle: Target viewing angle (theta).
        output_path: Path to save the resulting plot.
        n_phi_bins: Number of bins along the phi coordinate. Defaults to 200.
        n_theta_bins: Number of bins along the theta coordinate. Defaults to 100.
        theta_j: Jet opening semi-angle limit. Defaults to 0.1.
    """
    # Select photons in viewing window (still using saved theta/phi for now)
    photons_in_viewing_window = cp.logical_and(
        cp.logical_and(
            cp.array(h5file['photon_theta'][:]) > viewing_angle - 0.001,
            cp.array(h5file['photon_theta'][:]) < viewing_angle + 0.001
        ),
        cp.logical_and(
            cp.array(h5file['photon_phi'][:]) > cp.pi - 0.001/m.sin(viewing_angle),
            cp.array(h5file['photon_phi'][:]) < cp.pi + 0.001/m.sin(viewing_angle)
        )
    )

    # Load positions from photon position four-vector
    x = cp.array(h5file['photon_position'][:, 1])[photons_in_viewing_window]
    y = cp.array(h5file['photon_position'][:, 2])[photons_in_viewing_window]
    z = cp.array(h5file['photon_position'][:, 3])[photons_in_viewing_window]

    # Convert to spherical coordinates from position
    r_cartesian = cp.sqrt(x**2 + y**2 + z**2)
    theta_pos = cp.arccos(z / r_cartesian)       # polar angle from z-axis
    phi_pos = cp.arctan2(y, x)                   # azimuthal angle from x-axis

    # Define bin edges for 2D histogram
    phi_edges = cp.linspace(-cp.pi, cp.pi, n_phi_bins + 1)
    theta_edges = cp.linspace(0, theta_j, n_theta_bins + 1)

    # Compute 2D histogram (GPU)
    hist2d, _, _ = cp.histogram2d(phi_pos, theta_pos, bins=[phi_edges, theta_edges])

    # Move to CPU for plotting
    hist2d = hist2d.get()
    phi_edges = phi_edges.get()
    theta_edges = theta_edges.get()

    # Create polar plot heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    PHI, THETA = np.meshgrid(phi_edges, theta_edges, indexing='ij')

    # Plot with pcolormesh
    pcm = ax.pcolormesh(PHI, THETA, hist2d, cmap='viridis')
    fig.colorbar(pcm, ax=ax, label="Photon count")

    # ax.set_theta_zero_location("N")
    # ax.set_theta_direction(-1)
    ax.set_rlabel_position(225)

    plt.savefig(output_path + "topdown_position_polar_heatmap.pdf", bbox_inches="tight")
    plt.show()



def PlotLightCurve(h5file, viewing_angles, output_path, num_bins = 100, figname="Light_Curve.pdf"):
    """Plots the light curve (intensity vs. time) of escaped photons.

    Args:
        h5file: Opened h5py.File object.
        viewing_angles: List/array of viewing angles (theta) to filter by.
        output_path: Path to save the resulting plot.
        num_bins: Number of histogram bins. Defaults to 100.
        figname: Output file name. Defaults to "Light_Curve.pdf".
    """
    fig, ax = plt.subplots()

    for viewing_angle in viewing_angles:
        # Select photons in viewing window
        print("Starting with ", viewing_angle)
        photons_in_viewing_window = cp.logical_and(
        cp.logical_and(
            cp.array(h5file['photon_theta'][:]) > viewing_angle - 0.01,
            cp.array(h5file['photon_theta'][:]) < viewing_angle + 0.01
        ),
        cp.logical_and(
            cp.array(h5file['photon_phi'][:]) > cp.pi - 0.01/m.sin(viewing_angle),
            cp.array(h5file['photon_phi'][:]) < cp.pi + 0.01/m.sin(viewing_angle)
        )
    )

        # Load positions from photon position four-vector
        # indices = cp.where(photons_in_viewing_window)[0].get()
        indices = cp.where(photons_in_viewing_window)[0]
        print("Done getting indices")
        # del photons_in_viewing_window
        # t = cp.array(h5file['photon_position'][indices, 0])
        # x = cp.array(h5file['photon_position'][indices, 1:4])
        # direction = cp.array(h5file['photon_wave_vector'][indices, 1:4])

        # del photons_in_viewing_window
        t = cp.array(h5file['photon_position'][:, 0])[indices]
        x = cp.array(h5file['photon_position'][:, 1:4])[indices]
        direction = cp.array(h5file['photon_wave_vector'][:, 1:4])[indices]
        print("done loading from file")
        plane_normal = cp.array([np.sin(viewing_angle)*np.cos(np.pi),np.sin(viewing_angle)*np.sin(np.pi),np.cos(viewing_angle)])[None,:]
        p0 = (cp.max(cp.linalg.norm(x,axis=1))+10.0)*plane_normal
        Delta_t = cp.sum( (p0-x)*plane_normal ,axis=1)/cp.sum(direction*plane_normal,axis=1)
        t += Delta_t
        print("done with calculation")

        # Histogram normalized for variable bin widths
        hist, edges = cp.histogram(t,bins=num_bins+1)
        bin_centers = (edges[1:]+edges[:-1])/2.0
        hist = hist.astype(DTYPE)
        scaling_factor = cp.max(hist)
        hist /= scaling_factor
        ax.plot(bin_centers.get()/scpc.c, hist.get(), label=f"θ={viewing_angle:.3f}, scaling factor = {scaling_factor}")
        print("Done with ", viewing_angle)
        # hist_density = hist / bin_widths  # counts per unit energy
    #
    #     # Convert to νFν
    #     nuFnu = hist_density * (bin_centers)**2
    #
    #     ax.plot(bin_centers , nuFnu, label=f"θ={viewing_angle:.3f}")
    #
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Normalized photon count (arbitrary units)")
    ax.legend()

    # Save and show
    fig.savefig(output_path + figname, bbox_inches='tight')
    plt.show()




def FOb(t,thetaObs,bulk_lorentz=100.0):
    """Calculates the observational time conversion factor.

    Args:
        t: Time array.
        thetaObs: Observation angle.
        bulk_lorentz: Jet bulk Lorentz factor. Defaults to 100.0.

    Returns:
        cupy.ndarray: Computed observational times.
    """
    print("BULK_LORENTZ",bulk_lorentz)
    ri = np.sqrt(1.0e21)
    A  = ri*thetaObs
    R  = ri*0.1
    D  = np.sqrt(2.0*ri*scpc.c*t)
    Sp = 1.0/( 1.0+np.exp(100.0*(R-A-D)) )
    Sm = 1.0/( 1.0+np.exp(-100.0*(R-A-D)) )
    f1 = (4.0*scpc.c)/( (bulk_lorentz**3)*ri*( (2.0*scpc.c*t)/ri + 1.0/bulk_lorentz**2 )**2 )
    f2 = np.pi*Sm+Sp*np.arccos( ( thetaObs**2-0.1**2+(2.0*scpc.c*t)/ri )/( 2.0*thetaObs*np.sqrt((2.0*scpc.c*t)/ri)  ) )
    return f1*f2


def PlotLightCurveAssumingSymmetry(h5file, viewing_angles, output_path, num_bins = 100, shift_t0 = False, figname="Light_Curve.pdf",plot_backscatter_cork_thoeretical = False, restrict_time=False,log_scale=False,mono_energetic=False):
    """Plots the light curve assuming azimuthal symmetry of the jet shell.

    Args:
        h5file: Opened h5py.File object.
        viewing_angles: List/array of viewing angles (theta) to filter by.
        output_path: Path to save the resulting plot.
        num_bins: Number of histogram bins. Defaults to 100.
        shift_t0: If True, shifts initial time t0. Defaults to False.
        figname: Output file name. Defaults to "Light_Curve.pdf".
        plot_backscatter_cork_thoeretical: If True, plots theoretical curve.
            Defaults to False.
        restrict_time: If True, limits the x-axis time range. Defaults to False.
        log_scale: If True, plots on a log scale. Defaults to False.
        mono_energetic: If True, handles monoenergetic spectrum characteristics.
            Defaults to False.
    """
    fig, ax = plt.subplots()

    for viewing_angle in viewing_angles:
        # Select photons in viewing window
        print("Starting with ", viewing_angle)
        photons_in_viewing_window = cp.logical_and(
            cp.array(h5file['photon_theta'][:]) > viewing_angle - 0.001,
            cp.array(h5file['photon_theta'][:]) < viewing_angle + 0.001
        )
        energy_mask = (cp.array( h5file['photon_energies'][:])*NATURAL_TO_KEV) >= 1.0
        photons_in_viewing_window = cp.logical_and(photons_in_viewing_window,energy_mask)

        # Load positions from photon position four-vector
        # indices = cp.where(photons_in_viewing_window)[0].get()
        indices = cp.where(photons_in_viewing_window)[0]
        print("Done getting indices")
        # del photons_in_viewing_window
        # t = cp.array(h5file['photon_position'][indices, 0])
        # x = cp.array(h5file['photon_position'][indices, 1:4])
        # direction = cp.array(h5file['photon_wave_vector'][indices, 1:4])

        # del photons_in_viewing_window
        t = cp.array(h5file['photon_position'][:, 0])[indices]
        x = cp.array(h5file['photon_position'][:, 1:4])[indices]
        direction = cp.array(h5file['photon_wave_vector'][:, 1:4])[indices]
        print("done loading from file")
        viewing_phi = cp.array(h5file['photon_phi'][:])[indices]
        plane_normal = cp.stack([cp.sin(viewing_angle)*cp.cos(viewing_phi),cp.sin(viewing_angle)*cp.sin(viewing_phi), cp.full_like(viewing_phi, cp.cos(viewing_angle))],axis=1,dtype=DTYPE)
        p0 = (cp.max(cp.linalg.norm(x,axis=1))+10.0)*plane_normal
        Delta_t = cp.sum( (p0-x)*plane_normal ,axis=1)/cp.sum(direction*plane_normal,axis=1)
        t += Delta_t
        if shift_t0:
            t -= cp.min(t)
        print("done with calculation")
        if restrict_time:
            t = t[t/scpc.c < 25.0]

        # Histogram normalized for variable bin widths
        hist, edges = cp.histogram(t,bins=num_bins+1)
        bin_centers = (edges[1:]+edges[:-1])/2.0
        hist = hist.astype(DTYPE)
        scaling_factor = cp.max(hist)
        hist /= scaling_factor
        if log_scale:
            bin_centers = cp.concatenate([cp.array([0.0]),bin_centers])
            hist = cp.concatenate([cp.array([0.0]),hist])
        line1, = ax.plot(bin_centers.get()/scpc.c, hist.get(), label=f"θ={viewing_angle:.3f}, scaling factor = {scaling_factor:,.0f}")
        if plot_backscatter_cork_thoeretical:
            start = 0
            if mono_energetic:
                if viewing_angle == 0.105:
                    scale = 1.0
                    start = 20
                elif viewing_angle == 0.155:
                    scale = 1.0
                    start = 15
                else:
                    scale = 1.0
                    start = 0
                theoretical_curve = FOb(bin_centers.get()[start:]/scpc.c,viewing_angle,bulk_lorentz=20.0)
            else:
                scale = 1.0
                theoretical_curve = FOb(bin_centers.get()/scpc.c,viewing_angle,bulk_lorentz=100.0)
            ax.plot(bin_centers.get()[start:]/scpc.c, scale*theoretical_curve/max(theoretical_curve), '--', color = line1.get_color(), label=f"θ={viewing_angle:.3f}, Vyas Analytical")

        print("Done with ", viewing_angle)
        # hist_density = hist / bin_widths  # counts per unit energy
    #
    #     # Convert to νFν
    #     nuFnu = hist_density * (bin_centers)**2
    #
    #     ax.plot(bin_centers , nuFnu, label=f"θ={viewing_angle:.3f}")
    #
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Normalized photon count (arbitrary units)")
    if log_scale:
        ax.set_yscale('log')
        ax.set_xlim(-5,400.0)
    ax.legend()

    # Save and show
    fig.savefig(output_path + figname, bbox_inches='tight')
    plt.show()



# def PlotPolarizationVsTime(h5file, viewing_angles, output_path, num_bins = 50):
#     fig, (ax_pd, ax_pa) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios':[4,1],'hspace':0.0})
#
#     for viewing_angle in viewing_angles:
#         # Select photons in viewing window
#         print("Starting with ", viewing_angle)
#         photons_in_viewing_window = cp.logical_and(
#             cp.array(h5file['photon_theta'][:]) > viewing_angle - 0.001,
#             cp.array(h5file['photon_theta'][:]) < viewing_angle + 0.001
#         )
#
#         # Load positions from photon position four-vector
#         # indices = cp.where(photons_in_viewing_window)[0].get()
#         indices = cp.where(photons_in_viewing_window)[0]
#
#         del photons_in_viewing_window
#
#         print("Done getting indices")
#         # del photons_in_viewing_window
#         # t = cp.array(h5file['photon_position'][indices, 0])
#         # x = cp.array(h5file['photon_position'][indices, 1:4])
#         # direction = cp.array(h5file['photon_wave_vector'][indices, 1:4])
#
#         # del photons_in_viewing_window
#         t = cp.array(h5file['photon_position'][:, 0])[indices]
#         x = cp.array(h5file['photon_position'][:, 1:4])[indices]
#         direction = cp.array(h5file['photon_wave_vector'][:, 1:4])[indices]
#         print("done loading from file")
#         viewing_phi = cp.array(h5file['photon_phi'][:])[indices]
#         plane_normal = cp.stack([cp.sin(viewing_angle)*cp.cos(viewing_phi),cp.sin(viewing_angle)*cp.sin(viewing_phi), cp.full_like(viewing_phi, cp.cos(viewing_angle))],axis=1,dtype=DTYPE)
#         p0 = (cp.max(cp.linalg.norm(x,axis=1))+10.0)*plane_normal
#         Delta_t = cp.sum( (p0-x)*plane_normal ,axis=1)/cp.sum(direction*plane_normal,axis=1)
#         t += Delta_t
#         t /= scpc.c
#         print("done with calculation")
#         del x, direction, viewing_phi
#
#         Q = cp.array(h5file['Q'][:])[indices]
#         U = cp.array(h5file['U'][:])[indices]
#
#         # Histograms
#         I_hist, edges = cp.histogram(t, bins=num_bins)
#         Q_hist, _ = cp.histogram(t, bins=num_bins, weights=Q)
#         U_hist, _ = cp.histogram(t, bins=num_bins, weights=U)
#
#         print(cp.max(Q_hist/I_hist))
#         print(cp.max(U_hist/I_hist))
#
#         bin_centers = (edges[1:]+edges[:-1])/2.0
#
#         # Avoid divide-by-zero
#         mask = I_hist > 0
#         pd = cp.zeros_like(I_hist, dtype=DTYPE)
#         pa = cp.zeros_like(I_hist, dtype=DTYPE)
#
#         pd[mask] = cp.sqrt(Q_hist[mask]**2 + U_hist[mask]**2) / I_hist[mask]
#         pa[mask] = 0.5 * (cp.arctan2(U_hist[mask], Q_hist[mask]) )/ cp.pi  # in units of π
#         pa += 0.5
#
#         # Plot
#         # ax2.plot(bin_centers*NATURAL_TO_KEV,I_hist)
#         ax_pd.plot(bin_centers.get(), pd.get(), label=f"θ = {viewing_angle:.3f} rad")
#         ax_pa.plot(bin_centers.get(), pa.get(), label=f"θ = {viewing_angle:.3f} rad")
#
#     # ax_pd.set_xscale('log')
#     # ax_pa.set_xscale('log')
#     #
#     # ax2.set_xscale('log')
#     # ax2.set_yscale('log')
#
#
#     # Define tick positions in units of π (e.g., from -0.5 to 0.5)
#     ticks = np.array([ 0, 0.25, 0.5,0.75,1.0])
#
#     # Set ticks and labels like -½π, -¼π, 0, ¼π, ½π
#     ax_pa.set_yticks(ticks)
#     ax_pa.set_yticklabels([r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$",r"$\frac{3}{4}\pi$", r"$\pi$"])
#
#
#     ax_pd.set_ylabel("Polarization Degree")
#     ax_pa.set_ylabel("Polarization Angle (× π)")
#     ax_pa.set_xlabel("Time (s)")
#
#     ax_pd.legend()
#     plt.tight_layout()
#     plt.savefig(output_path + "polarization_time.pdf")
#     plt.show()

def PlotPolarizationVsTime(h5file, viewing_angles, output_path, num_bins = 50, shift_t0 = False, restrict_time=False,BS_plot=False):
    """Plots polarization degree and angle as a function of time.

    Args:
        h5file: Opened h5py.File object.
        viewing_angles: List/array of viewing angles (theta) to filter by.
        output_path: Path to save the resulting plot.
        num_bins: Number of histogram bins. Defaults to 50.
        shift_t0: If True, shifts initial time t0. Defaults to False.
        restrict_time: If True, limits the x-axis time range. Defaults to False.
        BS_plot: If True, adjusts plots specifically for backscatter cork configurations.
            Defaults to False.
    """
    fig, (ax_pd, ax_pa) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios':[4,1],'hspace':0.0})

    for viewing_angle in viewing_angles:
        print("Starting with ", viewing_angle)
        photons_in_viewing_window = cp.logical_and(
            cp.array(h5file['photon_theta'][:]) > viewing_angle - 0.001,
            cp.array(h5file['photon_theta'][:]) < viewing_angle + 0.001
        )

        energy_mask = (cp.array( h5file['photon_energies'][:])*NATURAL_TO_KEV) >= 1.0
        photons_in_viewing_window = cp.logical_and(photons_in_viewing_window,energy_mask)

        indices = cp.where(photons_in_viewing_window)[0]
        del photons_in_viewing_window

        t = cp.array(h5file['photon_position'][:, 0])[indices]
        x = cp.array(h5file['photon_position'][:, 1:4])[indices]
        direction = cp.array(h5file['photon_wave_vector'][:, 1:4])[indices]
        viewing_phi = cp.array(h5file['photon_phi'][:])[indices]

        plane_normal = cp.stack(
            [cp.sin(viewing_angle)*cp.cos(viewing_phi),
             cp.sin(viewing_angle)*cp.sin(viewing_phi),
             cp.full_like(viewing_phi, cp.cos(viewing_angle))],
            axis=1, dtype=DTYPE
        )
        p0 = (cp.max(cp.linalg.norm(x,axis=1))+10.0)*plane_normal
        Delta_t = cp.sum((p0-x)*plane_normal,axis=1)/cp.sum(direction*plane_normal,axis=1)
        t += Delta_t
        if shift_t0:
            t -= cp.min(t)
        t /= scpc.c
        t_restrict = t<20.0
        if restrict_time:
            t = t[t_restrict]
        del x, direction, viewing_phi

        Q = cp.array(h5file['Q'][:])[indices]
        U = cp.array(h5file['U'][:])[indices]
        if restrict_time:
            Q = Q[t_restrict]
            U = U[t_restrict]

        # Histograms
        I_hist, edges = cp.histogram(t, bins=num_bins)
        Q_hist, _ = cp.histogram(t, bins=num_bins, weights=Q)
        U_hist, _ = cp.histogram(t, bins=num_bins, weights=U)

        bin_centers = (edges[1:]+edges[:-1])/2.0

        # Avoid divide-by-zero
        mask = I_hist > 0
        pd = cp.zeros_like(I_hist, dtype=DTYPE)
        pa = cp.zeros_like(I_hist, dtype=DTYPE)

        pd[mask] = cp.sqrt(Q_hist[mask]**2 + U_hist[mask]**2) / I_hist[mask]
        pa[mask] = 0.5 * (cp.arctan2(U_hist[mask], Q_hist[mask])) / cp.pi
        pa += 0.5

        # pa = 0.5*cp.unwrap(2.0*cp.pi*pa)
        # pa /= cp.pi
        # pa[pa<-0.1] += 1.0
        # pa[pa>0.9] += -1.0
        pa = cp.mod(pa,cp.pi)

        # --- Error bars ---
        err = cp.zeros_like(I_hist, dtype=DTYPE)
        err[mask] = 1.0 / cp.sqrt(I_hist[mask])

        # Apply filter: remove bins where error > 0.3
        good = (mask & (err <= 0.1))

        for element in range(0,len(pa)):
            if pa[element] >0.9:
                pa[element] = pa[element]-1.0

        # Convert to NumPy before plotting
        bin_centers_np = bin_centers.get()
        pd_np = pd.get()
        pa_np = pa.get()
        err_np = err.get()


        eb = ax_pd.errorbar(
            bin_centers_np[good.get()],
            pd_np[good.get()],
            yerr=err_np[good.get()],
            fmt="o-", label=f"θ = {viewing_angle:.3f} rad"
        )
        ax_pa.errorbar(
            bin_centers_np[good.get()],
            pa_np[good.get()],
            yerr=err_np[good.get()]/(2.0*np.pi*pd_np[good.get()]),
            fmt="o-", color = eb[0].get_color(), label=f"θ = {viewing_angle:.3f} rad"
        )
        ax_pd.plot( bin_centers_np[good.get()], pd_np[good.get()], color = eb[0].get_color() )
        ax_pa.plot( bin_centers_np[good.get()], pa_np[good.get()], color = eb[0].get_color() )

    ticks = np.array([0, 0.25, 0.5, 0.75, 1.0])
    ax_pa.set_yticks(ticks)
    ax_pa.set_yticklabels([r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r" "])

    ax_pd.set_ylabel(r"$\Pi$")
    ax_pd.set_ylim(bottom = -0.01)
    ax_pa.set_ylabel(r"$\chi$")
    ax_pa.set_xlabel("Time (s)")
    ax_pa.set_ylim(-0.10,1.0)

    ax_pd.legend()
    plt.tight_layout()
    plt.savefig(output_path + "polarization_time.pdf")
    plt.show()



def PlotPolarizationVsTheta(h5file, output_path, num_bins = 50):
    """Plots polarization degree and angle as a function of theta.

    Args:
        h5file: Opened h5py.File object.
        output_path: Path to save the resulting plot.
        num_bins: Number of histogram bins. Defaults to 50.
    """
    fig, (ax_pd, ax_pa) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios':[4,1],'hspace':0.0})

    energy_mask = (cp.array( h5file['photon_energies'][:])*NATURAL_TO_KEV) >= 1.0

    theta_photon = (cp.array(h5file['photon_theta'][:]))
    theta_photon_mask = cp.logical_and(theta_photon*BULK_LORENTZ < 0.45,energy_mask)
    theta_photon = theta_photon[theta_photon_mask]

    Q = cp.array(h5file['Q'][:])[theta_photon_mask]
    U = cp.array(h5file['U'][:])[theta_photon_mask]
    # print(Q)
    # print(U)

    # Histograms
    num_bins += 1
    I_hist, edges = cp.histogram(theta_photon, bins=num_bins)
    Q_hist, _ = cp.histogram(theta_photon, bins=num_bins, weights=Q)
    U_hist, _ = cp.histogram(theta_photon, bins=num_bins, weights=U)


    bin_centers = (edges[1:]+edges[:-1])/2.0

    # Avoid divide-by-zero
    mask = I_hist > 0
    pd = cp.zeros_like(I_hist, dtype=DTYPE)
    pa = cp.zeros_like(I_hist, dtype=DTYPE)

    pd[mask] = cp.sqrt(Q_hist[mask]**2 + U_hist[mask]**2) / I_hist[mask]
    # pd[mask] = Q_hist[mask] / I_hist[mask]
    pa[mask] = 0.5 * (cp.arctan2(U_hist[mask], Q_hist[mask]) )/ cp.pi  # in units of π
    pa += 0.5

    # #
    # # if pa[0]<0:
    # #     pa[0] += 1.0
    # # for i in range(0,len(pa)-1):
    # #     temp1 = abs(pa[i]-pa[i+1])
    # #     temp2 = abs(pa[i]-(pa[i+1]+1.0))
    # #     if temp2 < temp1:
    # #         pa[i+1] += 1.0
    #
    #
    # pa = 0.5*cp.unwrap(2.0*cp.pi*pa)
    # pa /= cp.pi
    # pa[pa<-0.1] += 1.0

    pa = cp.mod(pa,cp.pi)


    # --- Error bars ---
    err = cp.zeros_like(I_hist, dtype=DTYPE)
    err[mask] = 1.0 / cp.sqrt(I_hist[mask])

    # Apply filter: remove bins where error > 0.3
    good = (mask & (err <= 0.3))


    for element in range(0,len(pa)):
        if pa[element] <1.0-0.95:
            pa[element] = pa[element]+1.0
    # Plot

    eb = ax_pd.errorbar(bin_centers[good].get(), pd[good].get(), yerr=err[good].get(), fmt='o')
    # ax_pa.errorbar(bin_centers[good].get()*BULK_LORENTZ, pa[good].get(), yerr=(err[good]/(2.0*pd[good])).get(), fmt='o', color = eb[0].get_color())
    ax_pa.errorbar(bin_centers[good].get(), pa[good].get(), yerr=(err[good]/(2.0*cp.pi*pd[good])).get(), fmt='o', color = eb[0].get_color())


    # ax2.plot(bin_centers*NATURAL_TO_KEV,I_hist)
    ax_pd.plot(bin_centers[good].get(), pd[good].get(), color = eb[0].get_color())
    ax_pa.plot(bin_centers[good].get(), pa[good].get(), color = eb[0].get_color())

    # ax_pd.set_xscale('log')
    # ax_pa.set_xscale('log')
    #
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')


    # Define tick positions in units of π (e.g., from -0.5 to 0.5)
    ticks = np.array([ 0, 0.25, 0.5,0.75,1.0])

    # Set ticks and labels like -½π, -¼π, 0, ¼π, ½π
    ax_pa.set_yticks(ticks)
    ax_pa.set_yticklabels([r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$",r"$\frac{3}{4}\pi$", r"$\pi$"])


    ax_pd.set_ylabel(r"$\Pi$")
    ax_pa.set_ylabel(r"$\chi$")
    ax_pa.set_xlabel(r"$\theta_{obs}$")
    ax_pa.set_ylim(0.0,1.2)
    ax_pd.set_ylim(bottom = 0.0)

    # ax_pd.legend()
    plt.tight_layout()
    plt.savefig(output_path + "polarization_theta.pdf")
    plt.show()

############################################################################# Depreciated #####################################################



def PlotEnergyUniformBins(h5file, viewing_angles, output_path, bins=10000):
    """Plots the energy spectrum of escaped photons using uniform binning.

    Args:
        h5file: Opened h5py.File object.
        viewing_angles: List/array of viewing angles (theta) to filter by.
        output_path: Path to save the resulting plot.
        bins: Number of uniform bins. Defaults to 10000.
    """
    fig, ax = plt.subplots()

    for viewing_angle in viewing_angles:
        photons_in_viewing_window = np.logical_and(
            h5file['photon_theta'][:] > viewing_angle - 0.01,
            h5file['photon_theta'][:] < viewing_angle + 0.01  # fixed '<' instead of '>'
        )
        photon_energies = h5file['photon_energies'][photons_in_viewing_window]* NATURAL_TO_KEV
        # hist, edges = np.histogram(photon_energies, bins=bins+1,weights=photon_energies)
        hist, edges = np.histogram(photon_energies, bins=bins+1,weights=photon_energies**2)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        ax.plot(bin_centers , hist)# * (bin_centers * NATURAL_TO_KEV)**2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel(r"$\nu F_\nu$ (arbitrary units)")

    plt.savefig(output_path+"energy_spectrum.pdf", format='pdf', bbox_inches='tight')

    plt.show()


def PlotPolarizationUniformBins(h5file, viewing_angles, output_path, bins=100):
    """Plots polarization degree vs. energy using uniform binning.

    Args:
        h5file: Opened h5py.File object.
        viewing_angles: List/array of viewing angles (theta) to filter by.
        output_path: Path to save the resulting plot.
        bins: Number of uniform bins. Defaults to 100.
    """
    fig, (ax_pd, ax_pa) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

    for viewing_angle in viewing_angles:
        photons_in_view = np.logical_and(
            h5file['photon_theta'][:] > viewing_angle - 0.01,
            h5file['photon_theta'][:] < viewing_angle + 0.01
        )

        energies = h5file['photon_energies'][photons_in_view]
        Q = h5file['photon_Q'][photons_in_view]
        U = h5file['photon_U'][photons_in_view]

        # Intensity histogram (I)
        I_hist, edges = np.histogram(energies, bins=bins+1)

        # Q and U histograms weighted by their respective values
        Q_hist, _ = np.histogram(energies, bins=edges, weights=Q)
        U_hist, _ = np.histogram(energies, bins=edges, weights=U)

        # Avoid division by zero
        valid_bins = I_hist > 0

        # Polarization degree
        PD = np.zeros_like(I_hist, dtype=float)
        PD[valid_bins] = np.sqrt(Q_hist[valid_bins]**2 + U_hist[valid_bins]**2) / I_hist[valid_bins]

        # Polarization angle (degrees)
        PA = np.zeros_like(I_hist, dtype=float)
        PA[valid_bins] = 0.5 * np.degrees(np.arctan2(U_hist[valid_bins], Q_hist[valid_bins]))

        # Bin centers
        bin_centers = (edges[:-1] + edges[1:]) / 2.0

        ax_pd.plot(bin_centers * NATURAL_TO_KEV, PD, label=f"θ={viewing_angle:.2f} rad")
        ax_pa.plot(bin_centers * NATURAL_TO_KEV, PA, label=f"θ={viewing_angle:.2f} rad")

    ax_pd.set_xscale('log')
    ax_pd.set_ylabel("Polarization Degree")
    ax_pd.legend()

    ax_pa.set_xscale('log')
    ax_pa.set_xlabel("Energy [keV]")
    ax_pa.set_ylabel("Polarization Angle [°]")

    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, "polarization_vs_energy.pdf")
    plt.savefig(fig_path, format='pdf', bbox_inches='tight')
    plt.show()
