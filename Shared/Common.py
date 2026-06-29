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

@cuda.jit(device=True,fastmath=False)
def Sum(m):
    x = 0.0
    for i in range(1,m+1):
        x = x + 1.0/i**3
    return x

def sample_subset(array, sample_size):
    N = array.shape[0]
    indices = cp.random.choice(N, size=sample_size, replace=False)
    return indices

def sample_subsetNumpy(array, sample_size):
    N = array.shape[0]
    indices = np.random.choice(N, size=sample_size, replace=False)
    return indices


def NormalizeFourVector(vectors,zero_component=1.0):
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

def GetLorentzTransform(array_to_transform,new_frame,inverse_transform=False):
    gamma = new_frame[:,0]
    direction = new_frame[:,1:4]

    beta = cp.sqrt(1.0-1.0/(gamma**2),dtype=DTYPE)
    beta_n = beta[:,None]*direction
    # print("beta_n: ",beta_n[0])

    t = array_to_transform[:,0]
    x = array_to_transform[:,1:4]

    beta_dot_x = cp.sum(beta_n*x,axis=1,dtype=DTYPE)
    # print("beta_dot_x: ",beta_dot_x)
    factor = (gamma**2)/(1.0+gamma)
    # factor = (gamma-1.0)/beta**2
    # print(factor-factor2)

    if inverse_transform:
        t_prime = gamma*(t+beta_dot_x)
        x_0 =   x[:,0] + beta_n[:,0]*( gamma*t + factor*beta_dot_x )
        x_1 =   x[:,1] + beta_n[:,1]*( gamma*t + factor*beta_dot_x )
        x_2 =   x[:,2] + beta_n[:,2]*( gamma*t + factor*beta_dot_x )
    else:
        t_prime = gamma*(t-beta_dot_x)
        x_0 =   x[:,0] + beta_n[:,0]*( -gamma*t + factor*beta_dot_x )
        x_1 =   x[:,1] + beta_n[:,1]*( -gamma*t + factor*beta_dot_x )
        x_2 =   x[:,2] + beta_n[:,2]*( -gamma*t + factor*beta_dot_x )
    # print("t_prime",t_prime,"beta_dot_x",beta_dot_x,"inverse_transform",inverse_transform)

    return cp.concatenate([t_prime[:,None],x_0[:,None],x_1[:,None],x_2[:,None]],axis=1)

# def GetLorentzTransformTimeComponent(t, x, new_frame, inverse_transform=False):
#     """
#     Compute only the time component of a Lorentz transformation.
#
#     Parameters
#     ----------
#     t : cupy.ndarray, shape (N,)
#         Time component in original frame.
#     x : cupy.ndarray, shape (N,3)
#         Spatial components in original frame.
#     new_frame : cupy.ndarray, shape (N,4)
#         Four-vector of the new frame's velocity: (gamma, nx, ny, nz).
#     inverse_transform : bool, default False
#         If True, performs the inverse Lorentz transformation.
#
#     Returns
#     -------
#     t_prime : cupy.ndarray, shape (N,)
#         Time component in the new frame.
#     """
#     gamma = new_frame[:, 0]
#     beta = cp.sqrt(1.0 - 1.0 / (gamma**2), dtype=t.dtype)
#     beta_dot_x = beta * (x[:, 0] * new_frame[:, 1] +
#                          x[:, 1] * new_frame[:, 2] +
#                          x[:, 2] * new_frame[:, 3])
#
#     if inverse_transform:
#         return gamma * (t + beta_dot_x)
#     else:
#         return gamma * (t - beta_dot_x)


def CalcAngularDirections(direction_4d: cp.ndarray):

    x = direction_4d[:, 1]
    y = direction_4d[:, 2]
    z = direction_4d[:, 3]

    norm = cp.sqrt(x**2 + y**2 + z**2)  # Will raise error if any norm == 0

    theta = cp.arccos(z / norm)
    phi = cp.arctan2(y, x) % (2 * cp.pi)

    return theta, phi


def SaveBatchTo_hdf5(batch,first_write,file_name):
    mode = 'w' if first_write else 'a'

    with h5.File(file_name,mode) as f:
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



def FlushBufferToDisk(memory_buffer,first_write,file_name):
    batch = {k: np.concatenate(v, axis=0) for k, v in memory_buffer.items() if v}

    for k in memory_buffer.keys():
        memory_buffer[k].clear()


    SaveBatchTo_hdf5(batch,first_write,file_name)







