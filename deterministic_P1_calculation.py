#!/usr/bin/env python3
"""Compute P_N = probability to escape after exactly N scatterings.

Applies to cold electrons, Thomson scattering, semi-infinite slab,
normal incidence mu0=1, for N=1..Nmax.

Warning: cost ~ M^N. Use small M for N>4, or increase M for better accuracy
for small N.
"""

import numpy as np
import time


def K_thomson(mu_out_arr, mu_in, nphi=512):
    """Compute K(mu_out, mu_in) = (3/(16*pi)) * ∫_0^{2π} (1 + cos^2 psi) dphi.

    Args:
        mu_out_arr: 1D array of mu_out values.
        mu_in: Scalar representing incident cosine angle.
        nphi: Angular grid size for phi integration. Defaults to 512.

    Returns:
        numpy.ndarray: 1D array of computed K values of same length as mu_out_arr.
    """
    mu_out = np.atleast_1d(mu_out_arr)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    cosphi = np.cos(phi)
    sqrt_in = np.sqrt(max(0.0, 1.0 - mu_in**2))
    sqrt_out = np.sqrt(np.maximum(0.0, 1.0 - mu_out**2))  # (M,)
    sqrt_prod = np.outer(np.sqrt(np.maximum(0.0, 1.0 - mu_out**2)), sqrt_in)  # (M,1)
    cospsi = (mu_in * mu_out)[:, None] + sqrt_prod * cosphi[None, :]
    integrand = 1.0 + cospsi**2
    int_phi = np.trapz(integrand, phi, axis=1)
    K = (3.0 / (16.0 * np.pi)) * int_phi
    return K


def compute_PN_up_to_Nmax(Nmax=6, M=60, nphi=512, verbose=True):
    """Compute P_N for N=1..Nmax on a uniform mu grid.

    Intermediate scattering directions: [-1,1] for inner scatters,
    final scattering must be in [-1,0] (escape upward).

    Args:
        Nmax: Maximum number of scatterings to calculate. Defaults to 6.
        M: Grid points for mu in [-1,1]. Defaults to 60.
        nphi: Azimuthal integration resolution. Defaults to 512.
        verbose: If True, prints progress details. Defaults to True.

    Returns:
        tuple: (mu_full, w_full, K_matrix, P_list) containing:
            - mu_full: The full mu grid in [-1, 1].
            - w_full: Quadrature weights for the full grid.
            - K_matrix: Precomputed K(mu_out, mu_in) matrix.
            - P_list: List of P_N probabilities for N=1..Nmax.
    """
    # grids
    mu_full = np.linspace(-1.0, 1.0, M)
    w_full = np.empty_like(mu_full)
    dx_full = mu_full[1] - mu_full[0]
    w_full[:] = dx_full
    w_full[0] *= 0.5
    w_full[-1] *= 0.5

    # final mu grid (escape) subset indices
    mask_final = mu_full <= 0.0
    mu_final = mu_full[mask_final]
    w_final = w_full[mask_final]

    if verbose:
        print("Precomputing K(..., mu_in) tables")
    t0 = time.time()
    K_cache = {}
    K_cache[1.0] = K_thomson(mu_full, 1.0, nphi=nphi)
    K_matrix = np.empty((M, M))
    for j, mu_in in enumerate(mu_full):
        K_matrix[j, :] = K_thomson(mu_full, mu_in, nphi=nphi)
    if verbose:
        print(f"Precompute done in {time.time()-t0:.2f} s; K_matrix shape {K_matrix.shape}")

    abs_mu_full = np.abs(mu_full)
    eps = 1e-14
    abs_mu_full_safe = np.where(abs_mu_full > eps, abs_mu_full, 1e100)

    def compute_PN_bruteforce(N):
        """Helper to calculate P_N using brute-force recursion.

        Args:
            N: The specific scattering count index to compute.

        Returns:
            float: Computed P_N.
        """
        if N == 0:
            return 0.0
        idxs = np.arange(M)
        weights = w_full

        def rec(level, prev_idx, prodK, sum_inv_abs, weight_prod):
            """Recursive helper executing nested summation over grid directions.

            Args:
                level: Current recursion depth.
                prev_idx: Index of previous scattering cosine.
                prodK: Accumulated Klein-Nishina product.
                sum_inv_abs: Accumulated sum of inverse absolute cosines.
                weight_prod: Accumulated product of weights.
            """
            if level == N:
                for iN_pos, iN in enumerate(np.where(mask_final)[0]):
                    mu_i = mu_full[iN]
                    K_factor = 0.0
                    if prev_idx is None:
                        K_factor = K_cache[1.0][iN]
                    else:
                        K_factor = K_matrix[prev_idx, iN]
                    prod = prodK * K_factor
                    sum_inv = sum_inv_abs + 1.0 / abs_mu_full_safe[iN]
                    weight = weight_prod * w_final[iN_pos]
                    total_contrib = prod * weight / (1.0 + sum_inv)
                    nonlocal_total[0] += total_contrib
                return

            if prev_idx is None:
                K_row = K_cache[1.0]
            else:
                K_row = K_matrix[prev_idx, :]

            for i_next in range(M):
                Kval = K_row[i_next]
                new_prodK = prodK * Kval
                new_sum_inv = sum_inv_abs + 1.0 / abs_mu_full_safe[i_next]
                new_weight_prod = weight_prod * weights[i_next]
                rec(level + 1, i_next, new_prodK, new_sum_inv, new_weight_prod)

        nonlocal_total = [0.0]
        rec(1, None, 1.0, 0.0, 1.0)
        return nonlocal_total[0]

    P_list = []
    t_start = time.time()
    for N in range(1, Nmax + 1):
        if verbose:
            print(f"Computing P_{N} ... (this may be slow for large N and M)")
        tN = time.time()
        PN = compute_PN_bruteforce(N)
        P_list.append(PN)
        if verbose:
            print(f"P_{N} = {PN:.10e}  (computed in {time.time()-tN:.2f}s)")
    if verbose:
        print(f"Total time: {time.time()-t_start:.2f}s")
    return mu_full, w_full, K_matrix, P_list


if __name__ == "__main__":
    Nmax = 6
    M = 48
    nphi = 384
    mu_full, w_full, K_matrix, P_list = compute_PN_up_to_Nmax(Nmax=Nmax, M=M, nphi=nphi, verbose=True)
    for i, PN in enumerate(P_list, start=1):
        print(f"P_{i} = {PN:.10e}")
    print("Sum P_1..P_Nmax =", sum(P_list))
    print("If you want P up to N=10, set Nmax=10 but reduce M (e.g. M=20) to keep runtime feasible.")


# #!/usr/bin/env python3
# """
# Compute probability A^(2) that a photon injected perpendicularly into a
# semi-infinite, purely-Thomson-scattering slab will scatter exactly twice and
# escape after the second scattering (cold electrons).
# """
# import numpy as np
#
# def K_thomson(mu_out, mu_in, nphi=512):
#     """
#     Compute K(mu_out, mu_in) = (3/(16pi)) * ∫_0^{2π} (1 + cos^2ψ) dφ
#     where cosψ = mu_in*mu_out + sqrt(1-mu_in^2)*sqrt(1-mu_out^2)*cosφ.
#     mu_out can be array, mu_in scalar.
#     """
#     mu_out = np.atleast_1d(mu_out)
#     phi = np.linspace(0.0, 2.0*np.pi, nphi, endpoint=False)
#     cosphi = np.cos(phi)
#     # vectorize: for each mu_out compute sqrt factors and cospsi array
#     sqrt_in = np.sqrt(max(0.0, 1.0 - mu_in**2))
#     sqrt_out = np.sqrt(np.maximum(0.0, 1.0 - mu_out**2))  # shape (M,)
#     # produce cospsi array shape (M, nphi)
#     sqrt_prod = np.outer(np.sqrt(np.maximum(0.0, 1.0 - mu_out**2)), sqrt_in)  # (M,1) times scalar
#     # but simpler:
#     cospsi = (mu_in * mu_out)[:,None] + (sqrt_out * sqrt_in)[:,None] * cosphi[None,:]
#     integrand = 1.0 + cospsi**2
#     int_phi = np.trapz(integrand, phi, axis=1)   # shape (M,)
#     K = (3.0 / (16.0 * np.pi)) * int_phi
#     return K
#
# def compute_A2(N_mu1=240, N_mu=240, nphi=512):
#     mu1_grid = np.linspace(-1.0, 1.0, N_mu1)
#     mu_grid = np.linspace(-1.0, 0.0, N_mu)
#
#     # K(mu1, 1) for first scattering (incident mu0=1)
#     K_mu1 = K_thomson(mu1_grid, 1.0, nphi=nphi)   # shape (N_mu1,)
#
#     # K(mu, mu1) matrix: for each mu1, compute K over mu_grid
#     K_matrix = np.empty((N_mu1, N_mu))
#     for i, mu1 in enumerate(mu1_grid):
#         K_matrix[i, :] = K_thomson(mu_grid, mu1, nphi=nphi)
#
#     # denominator matrix and integrand
#     eps = 1e-14
#     abs_mu1 = np.abs(mu1_grid)
#     abs_mu = np.abs(mu_grid)
#     denom = np.empty((N_mu1, N_mu))
#     for i in range(N_mu1):
#         for j in range(N_mu):
#             a = abs_mu1[i] if abs_mu1[i] > eps else 1e100
#             b = abs_mu[j]  if abs_mu[j]  > eps else 1e100
#             denom[i,j] = 1.0 + 1.0/a + 1.0/b
#
#     integrand = (K_mu1[:,None] * K_matrix) / denom
#
#     # integrate over mu (axis=1) then mu1 (axis=0)
#     int_mu = np.trapz(integrand, mu_grid, axis=1)
#     A2 = np.trapz(int_mu, mu1_grid)
#     return A2
#
# if __name__ == "__main__":
#     # baseline run
#     A2 = compute_A2(N_mu1=240, N_mu=240, nphi=512)
#     print("A^(2) (baseline grids) = {:.8e}".format(A2))
#
#     # quick convergence test: double resolution
#     A2_fine = compute_A2(N_mu1=360, N_mu=360, nphi=768)
#     print("A^(2) (fine grids)     = {:.8e}".format(A2_fine))
#     print("Relative difference (fine vs baseline) = {:.2e}".format((A2_fine - A2)/A2))
