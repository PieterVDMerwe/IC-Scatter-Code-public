"""Utility script to read and print sample data from Compton Drag HDF5 simulation output.

Reads and lists datasets from 'Compton_Drag_Data.h5', performing sanity checks
on the Q Stokes parameter bounds, checking for NaNs, and printing statistics of
surviving photon energies.
"""

import h5py
import numpy as np
import cupy as cp
import scipy.constants as scpc

JOULE_TO_KEV = (6.242e15)
M_EC2 = scpc.m_e * scpc.c**2
NATURAL_TO_KEV = M_EC2 * JOULE_TO_KEV

# Path to your HDF5 file
file_path = r"Compton_Drag_Data/Compton_Drag_Data.h5"

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
    Q_nan = cp.isnan(Q_device)
    Q_larger = cp.logical_or(Q_device > 1.0, Q_device < -1.0)
    print("nans in Q", len(Q_device[Q_nan]))
    print("out of bounds in Q", len(Q_device[Q_larger]))

    photon_energies = cp.array(f["photon_energies"][:])

    print("amount of photons below 1 keV:", len(photon_energies[photon_energies * NATURAL_TO_KEV >= 1.0]))
