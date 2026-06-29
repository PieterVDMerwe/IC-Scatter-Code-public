"""Utility script to read and print sample data from backscatter HDF5 simulation output.

Reads and lists datasets from 'Backscatter_Dominated_Cork_Data.h5', printing
shape, dtype, sample elements, and statistics of final scattering counts.
"""

import h5py
import numpy as np

# Path to your HDF5 file
file_path = r"Backscatter_Dominated_Cork_Data/Backscatter_Dominated_Cork_Data.h5"

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

    for i in range(31):
        scatter_check = f['final_iteration'][:] == i
        print(f"{i} scatterings: ", len(f['final_iteration'][:][scatter_check]) / f['final_iteration'][:].size)
