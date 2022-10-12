#!/bin/bash -l
#COBALT -t 120
#COBALT -q full-node
#COBALT -A ALCFAITP
#COBALT -n 1
#COBALT --attrs filesystems=home,grand

# Data is stored on Grand for the purposes of this class.

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series
cd /home/soc/ai-science-training-series/04_modern_neural_networks/

# Makes things run faster on a GPU.
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
python train_resnet34_self.py

# Instead of through the interactive 'notebook', this has to run in interactive terminal mode with
# qsub-gpu -q single-gpu -t 60 -I -n 1 -A ALCFAITP --attrs filesystems=home,grand
# Or use the Jupyter environment.
# Note that on the interactive thetaGPU, you need to enter both 'module load conda/2022-07-01' AND 'conda activate' to set the environment.