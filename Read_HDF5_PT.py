"""Utility script to read and print sample data from polarization test HDF5 output.

Reads and lists datasets from 'Polarizations_Test.h5', validating Q/U values
and checking unique occurrences of Stokes vectors.
"""

import h5py
import numpy as np
import cupy as cp
import scipy.constants as scpc

JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e * scpc.c**2
NATURAL_TO_KEV = M_EC2 * JOULE_TO_KEV

# Path to your HDF5 file
file_path = r"Polarizations_Test/Polarizations_Test.h5"

# Open the file in read mode
with h5py.File(file_path, 'r') as f:
    # List all keys (datasets) in the file
    print("Keys in HDF5 file:", list(f.keys()))

    # Access a specific dataset (e.g., photon_energies)
    for key in f.keys():
        dataset = f[key]
        print(f"Dataset: {key}, Shape: {dataset.shape}, Dtype: {dataset.dtype}")

        # Read a small portion of the data (e.g., first 10 elements)
        data = dataset[:10]
        print(f"Sample data for {key}:\n", data)
    Q_device = cp.array(f['Q'][:])
    U_device = cp.array(f['U'][:])
    Q_nan = cp.isnan(Q_device)
    Q_larger = cp.logical_or(Q_device > 1.0, Q_device < -1.0)
    print("nans in Q", len(Q_device[Q_nan]))
    print("out of bounds in Q", len(Q_device[Q_larger]))

    print("Q unique (length):", len(cp.unique(Q_device)))
    print("U unique (length):", len(cp.unique(U_device)))
    for values in cp.unique(Q_device):
        print("for value ", values, "len is:", len(Q_device[Q_device == values]))

    photon_energies = cp.array(f["photon_energies"][:])

    print("amount of photons below 1 keV:", len(photon_energies[photon_energies * NATURAL_TO_KEV >= 1.0]))
