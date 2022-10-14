#!/bin/bash -l
#COBALT -q single-gpu
#COBALT -t 60
#COBALT -n 1
#COBALT -A ALCFAITP
#COBALT -M tp@u.northwestern.edu
#COBALT --attrs filesystems=home,grand

# Make executable with 
# chmod +x resnet34_trainer.sh

# Submit with
# qsub-gpu resnet34_trainer.sh

# Data is stored on Grand for the purposes of this class.

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series
cd /home/soc/ai-science-training-series/04_modern_neural_networks/04_homework/

# Makes things run faster on a GPU.
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
python resnet34_trainer.py -b 256 -e 1 -l 0.0001

# Instead of through the interactive 'notebook', this has to run in interactive terminal mode with
# qsub-gpu -q single-gpu -t 60 -I -n 1 -A ALCFAITP --attrs filesystems=home,grand
# or
# qsub-gpu -q full-node -t 60 -I -n 1 -A ALCFAITP --attrs filesystems=home,grand
# module load conda/2022-07-01
# conda activate
# clear; python resnet34_trainer.py
# Note that on the interactive thetaGPU, you need to enter both 'module load conda/2022-07-01' AND 'conda activate' to set the environment.