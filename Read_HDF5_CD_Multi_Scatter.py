"""Utility script to read and print sample data from multi-scatter Compton Drag HDF5 simulation output.

Reads and lists datasets from 'Compton_Drag_Multi_Scatter_Data.h5', validating Q
bounds and printing final scattering iteration frequencies.
"""

import h5py
import numpy as np
import cupy as cp

# Path to your HDF5 file
file_path = r"Compton_Drag_Multi_Scatter_Data/Compton_Drag_Multi_Scatter_Data.h5"

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

    for i in range(np.max(f['final_iteration'][:]) + 1):
        scatter_check = f['final_iteration'][:] == i
        print(f"{i} scatterings: ", len(f['final_iteration'][:][scatter_check]))
