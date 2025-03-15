#!/bin/bash

# to run
# nohup bash conda_run_memdlm.sh > memdlm_outs.out 2> memdlm_errors.err &

# Set paths
HOME_LOC=/raid/sg666   # Change to your actual home directory
SCRIPT_LOC=$HOME_LOC/MeMDLM/MeMDLM/src/diffusion 
LOG_LOC=$SCRIPT_LOC  


CONDA_ENV=shrey_mdlm
PYTHON_EXECUTABLE=$(conda run -n $CONDA_ENV which python)  


mkdir -p $LOG_LOC

# Activate Conda environment
echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/bin/activate $CONDA_ENV

# Run the script using multiple GPUs and redirect output/error logs
CUDA_VISIBLE_DEVICES=1,2,3 nohup $PYTHON_EXECUTABLE $SCRIPT_LOC/main.py > "$LOG_LOC/memdlm_outs.out" 2> "$LOG_LOC/memdlm_errors.err" &

echo "Script started at $(date)"
conda deactivate
