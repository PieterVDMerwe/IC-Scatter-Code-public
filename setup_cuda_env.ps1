# Name of the conda environment
$envName = "grb-montecarlo-env"

Write-Host "Creating conda environment '$envName'..."

# Create environment if it doesn't exist
conda create --name $envName python=3.11 numpy scipy matplotlib numba pip -y

# Activate conda base so we can then activate our target env
conda activate base

Write-Host "Activating environment '$envName'..."
conda activate $envName

# Install CuPy for CUDA 12.x (covers 12.0 to 12.3+)
Write-Host "Installing CuPy for CUDA 12.x..."
pip install cupy-cuda12x

# Set PYTHONPATH to current directory
$pwdPath = Get-Location
$env:PYTHONPATH = "$pwdPath;$env:PYTHONPATH"

Write-Host "Environment ready."
Write-Host "PYTHONPATH is set to: $env:PYTHONPATH"

