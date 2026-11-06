#!/bin/bash
#SBATCH --job-name=cosi_classify
#SBATCH --output=logs/cosi_classify_%j.out
#SBATCH --error=logs/cosi_classify_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu

# Create results directory
mkdir -p results

# Load modules
echo "Loading modules..."
module purge
module load Python/3.11.5

# Activate COSI virtual environment
echo "Activating COSI virtual environment..."
source /pasteur/appa/homes/mcampi/envs/hf_env/bin/activate

# Set cache to avoid home disk space issues
export HF_HOME=/pasteur/appa/scratch/mcampi/COSI2/.cache
export TRANSFORMERS_CACHE=/pasteur/appa/scratch/mcampi/COSI2/.cache

# Install sentencepiece if not already installed
# pip install sentencepiece --quiet

# Load CUDA
module load cuda/11.8

# Run the classification
echo "Starting COSI classification at $(date)"
python3 classify_cosi_evolution.py
# python3 debug_cosi.py

echo "Classification completed at $(date)"

# Deactivate virtual environment
deactivate
