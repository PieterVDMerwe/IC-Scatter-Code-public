# IC-Scatter-Code-public

A high-performance Monte Carlo single-photon tracking Inverse Compton (IC) scattering and ray-tracing codebase designed to simulate and investigate polarization signatures from Gamma-Ray Burst (GRB) models. 

Massive parallelization is achieved on GPUs by utilizing **CuPy** and **Numba CUDA** (`numba.cuda`).

---

## Key Features

- **Massive Parallelization**: Offloads photon propagation, coordinate transforms, and scattering calculations to the GPU.
- **Polarized Compton/IC Scattering**: Full implementation of polarization-dependent Klein-Nishina cross-sections, scattering angles, polarization rotations, and Stokes parameter ($Q$, $U$) evaluations.
- **Physical Sampling Distributions**:
  - Planck blackbody spectrum.
  - Pair annihilation photon spectrum (via grid-based inversion and rejection sampling).
  - Thermal electron Lorentz factors (Maxwell-Jüttner distribution).
  - Relativistic isotropic directions (with cone limits).
- **Jet Modeling Configurations**:
  - **Backscatter-Dominated Cork**: Simulates photon propagation inside relativistic jet shells where backscattering dominates.
  - **Compton Drag**: Simulates photon-electron drag and Lorentz factor evolution over single and multi-scattering paths.

---

## Codebase Structure

The project is organized as follows:

```
├── Shared/
│   ├── Common.py           # Coordinate conversions, Lorentz boosts, HDF5 I/O
│   ├── Core_IC.py          # Scattering cross-sections, Stokes parameters, and boosts
│   ├── Distributions.py    # GPU samplers for Planck, Maxwell-Jüttner, and Pair Annihilation
│   ├── Plot_Functions.py   # Plotting utility functions (SED, polarizations, light curves)
│   └── Polarization_Test.py# Polarization tracking simulation verification
│
├── Run_Backscatter_Dominated_Cork.py               # Runner for backscatter cork simulations
├── Run_Backscatter_Dominated_Cork_Mono_Energetic.py # Runner for monoenergetic backscatter simulations
├── Run_Compton_Drag.py                             # Runner for single-scatter Compton Drag
├── Run_Compton_Drag_Multi_Scatter.py               # Runner for multi-scatter Compton Drag
├── Run_Polarization_Test.py                        # Runner for polarization verification tests
│
├── Plot_BS_Data.py                                 # Plotting script for backscatter data
├── Plot_BS_Data_Mono_Energetic.py                   # Plotting script for monoenergetic data
├── Plot_CD_Data.py                                 # Plotting script for Compton Drag data
├── Plot_CD_Data_Multi_Scatter.py                   # Plotting script for multi-scatter data
├── Plot_Polarization_Test.py                       # Plotting script for polarization tests
│
├── Read_HDF5*.py                                   # Diagnostic scripts to load/sanity-check outputs
├── deterministic_P1_calculation.py                 # Numerical deterministic slab scattering solver
├── grb-montecarlo-env.yml                          # Anaconda environment configuration
└── setup_cuda_env.ps1                              # Environment setup script for CUDA paths
```

---

## Setup & Requirements

- **Requirements**: NVIDIA GPU with CUDA support, CUDA Toolkit installed, Python 3.
- **Dependencies**: `numpy`, `cupy`, `numba`, `scipy`, `h5py`, `matplotlib`.

To set up the environment using conda:

```bash
conda env create -f grb-montecarlo-env.yml
conda activate grb-montecarlo
```

Use `setup_cuda_env.ps1` in PowerShell to resolve local CUDA path references if necessary.

---

## Citation & References

If you use this code or findings from it in your research, please cite the following paper:

**Citation:**
> van der Merwe, P., & Böttcher, M. (2026). Investigating Polarization Signatures from GRB Models. *The Astrophysical Journal*, 998(2), 290. https://doi.org/10.3847/1538-4357/ae3a95

**BibTeX:**
```bibtex
@article{van der Merwe_2026,
  doi = {10.3847/1538-4357/ae3a95},
  url = {https://doi.org/10.3847/1538-4357/ae3a95},
  year = {2026},
  month = {feb},
  publisher = {The American Astronomical Society},
  volume = {998},
  number = {2},
  pages = {290},
  author = {van der Merwe, Pieter and Böttcher, Markus},
  title = {Investigating Polarization Signatures from GRB Models},
  journal = {The Astrophysical Journal},
}
```
