"""Common utility functions, constants, and helper operations for the simulation.

This module provides common mathematical operations, coordinate transformations,
frame changes (Lorentz transforms), and HDF5 file I/O operations used throughout
the Inverse Compton scattering simulation.
"""

from numba import cuda, float64
import numba
import numpy as np
import cupy as cp
import math as m
from numba.cuda import libdevice as cudalib
import scipy.constants as scpc
from numba.cuda.random import create_xoroshiro128p_states as cStates
from numba.cuda.random import xoroshiro128p_uniform_float64 as uniform
import random

from pathlib import Path
# import pickle
import h5py as h5
import os

############## Constants ##################
DTYPE = cp.float64
MODULE_DIR = Path(__file__).parent
RAND_SEED_TOPEND = 4294967295
BLOCK_SIZE = 256
FORWARD = False
BACKWARD = True


###################################################### Common #########################################################

@cuda.jit(device=True, fastmath=False)
def Sum(m):
    """Computes the sum of inverse cubes up to m.

    Args:
        m: The upper limit of the summation (inclusive).

    Returns:
        float: The sum of 1 / i^3 for i from 1 to m.
    """
    x = 0.0
    for i in range(1, m + 1):
        x = x + 1.0 / i**3
    return x


def sample_subset(array, sample_size):
    """Randomly samples indices from a CuPy array without replacement.

    Args:
        array: The input array or structure from which to sample.
        sample_size: The number of indices to sample.

    Returns:
        cupy.ndarray: A 1D array of randomly chosen indices.
    """
    N = array.shape[0]
    indices = cp.random.choice(N, size=sample_size, replace=False)
    return indices


def sample_subsetNumpy(array, sample_size):
    """Randomly samples indices from a NumPy array without replacement.

    Args:
        array: The input array or structure from which to sample.
        sample_size: The number of indices to sample.

    Returns:
        numpy.ndarray: A 1D array of randomly chosen indices.
    """
    N = array.shape[0]
    indices = np.random.choice(N, size=sample_size, replace=False)
    return indices


def NormalizeFourVector(vectors, zero_component=1.0):
    """Normalizes the spatial components of 4-vectors and sets the time component.

    Assumes the input shape is (N, 4). The spatial components (x, y, z) are
    normalized to unit vectors, and the time component (index 0) is set to
    the specified zero_component value.

    Args:
        vectors: A CuPy array of shape (N, 4) containing the 4-vectors.
        zero_component: The value to set for the time component (default is 1.0).

    Returns:
        cupy.ndarray: The normalized 4-vectors array (in-place modification).
    """
    # Access columns directly (assumes shape (N, 4) and normalization of spatial components only)
    x = vectors[:, 1]
    y = vectors[:, 2]
    z = vectors[:, 3]

    norms = cp.sqrt(x**2 + y**2 + z**2)  # Add epsilon to avoid div-by-zero

    vectors[:, 1] /= norms
    vectors[:, 2] /= norms
    vectors[:, 3] /= norms
    vectors[:, 0] = zero_component

    return vectors


##################### Frame Change ###############################################

def GetLorentzTransform(array_to_transform, new_frame, inverse_transform=False):
    """Performs a Lorentz boost transformation on an array of 4-vectors.

    Args:
        array_to_transform: CuPy array of shape (N, 4) containing 4-vectors
            to transform (e.g., [t, x, y, z]).
        new_frame: CuPy array of shape (N, 4) specifying the velocity/boost
            parameters of the target frame (e.g., [gamma, nx, ny, nz]).
        inverse_transform: If True, performs the inverse Lorentz boost.
            Defaults to False.

    Returns:
        cupy.ndarray: Transformed 4-vectors of shape (N, 4).
    """
    gamma = new_frame[:, 0]
    direction = new_frame[:, 1:4]

    beta = cp.sqrt(1.0 - 1.0 / (gamma**2), dtype=DTYPE)
    beta_n = beta[:, None] * direction
    # print("beta_n: ",beta_n[0])

    t = array_to_transform[:, 0]
    x = array_to_transform[:, 1:4]

    beta_dot_x = cp.sum(beta_n * x, axis=1, dtype=DTYPE)
    # print("beta_dot_x: ",beta_dot_x)
    factor = (gamma**2) / (1.0 + gamma)
    # factor = (gamma-1.0)/beta**2
    # print(factor-factor2)

    if inverse_transform:
        t_prime = gamma * (t + beta_dot_x)
        x_0 = x[:, 0] + beta_n[:, 0] * (gamma * t + factor * beta_dot_x)
        x_1 = x[:, 1] + beta_n[:, 1] * (gamma * t + factor * beta_dot_x)
        x_2 = x[:, 2] + beta_n[:, 2] * (gamma * t + factor * beta_dot_x)
    else:
        t_prime = gamma * (t - beta_dot_x)
        x_0 = x[:, 0] + beta_n[:, 0] * (-gamma * t + factor * beta_dot_x)
        x_1 = x[:, 1] + beta_n[:, 1] * (-gamma * t + factor * beta_dot_x)
        x_2 = x[:, 2] + beta_n[:, 2] * (-gamma * t + factor * beta_dot_x)
    # print("t_prime",t_prime,"beta_dot_x",beta_dot_x,"inverse_transform",inverse_transform)

    return cp.concatenate([t_prime[:, None], x_0[:, None], x_1[:, None], x_2[:, None]], axis=1)


def CalcAngularDirections(direction_4d: cp.ndarray):
    """Calculates spherical angles (theta, phi) from spatial components of 4-vectors.

    Args:
        direction_4d: CuPy array of shape (N, 4) containing the 4-vectors.

    Returns:
        tuple[cupy.ndarray, cupy.ndarray]: A tuple containing:
            - theta: Polar angle (0 to pi).
            - phi: Azimuthal angle (0 to 2*pi).
    """
    x = direction_4d[:, 1]
    y = direction_4d[:, 2]
    z = direction_4d[:, 3]

    norm = cp.sqrt(x**2 + y**2 + z**2)  # Will raise error if any norm == 0

    theta = cp.arccos(z / norm)
    phi = cp.arctan2(y, x) % (2 * cp.pi)

    return theta, phi


def SaveBatchTo_hdf5(batch, first_write, file_name):
    """Saves or appends a batch of simulation data to an HDF5 file.

    Args:
        batch: Dict of dataset name keys and NumPy arrays of data values.
        first_write: If True, creates the dataset with unlimited maxshape;
            if False, appends data by resizing the existing dataset.
        file_name: Path or file name of the target HDF5 file.
    """
    mode = 'w' if first_write else 'a'

    with h5.File(file_name, mode) as f:
        for key, data in batch.items():
            if first_write:
                # Create new dataset with unlimited rows (None), enabling resizing later
                maxshape = (None,) + data.shape[1:]
                f.create_dataset(key, data=data, maxshape=maxshape, chunks=True)
            else:
                # Append new data by resizing the dataset first
                dset = f[key]
                orig_len = dset.shape[0]
                new_len = orig_len + data.shape[0]
                dset.resize(new_len, axis=0)
                dset[orig_len:new_len] = data


def FlushBufferToDisk(memory_buffer, first_write, file_name):
    """Flushes standard lists of buffered data to disk by saving to HDF5 format.

    Args:
        memory_buffer: Dict of dataset name keys and lists containing batches.
        first_write: If True, creates new datasets; if False, appends.
        file_name: Path or file name of the target HDF5 file.
    """
    batch = {k: np.concatenate(v, axis=0) for k, v in memory_buffer.items() if v}

    for k in memory_buffer.keys():
        memory_buffer[k].clear()

    SaveBatchTo_hdf5(batch, first_write, file_name)
