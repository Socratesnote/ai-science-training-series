#!/bin/bash -l
#COBALT -q single-gpu
#COBALT -t 60
#COBALT -n 1
#COBALT -A ALCFAITP
#COBALT -M tp@u.northwestern.edu
#COBALT --attrs filesystems=home,grand

# Make executable with 
# chmod +x train_resnet34_thetagpu.sh

# Submit with
# qsub-gpu train_resnet34_thetagpu.sh

# Data is stored on Grand for the purposes of this class.

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series
cd /home/soc/ai-science-training-series/05_dataPipelines/05_homework

# Makes things run faster on a GPU.
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
python train_resnet34.py -t 1 -b 1 -n single
python train_resnet34.py -t 1 -b 4 -n single
python train_resnet34.py -t 1 -b 8 -n single
python train_resnet34.py -t 1 -b 16 -n single
python train_resnet34.py -t 1 -b 32 -n single

python train_resnet34.py -t 4 -b 1 -n single
python train_resnet34.py -t 4 -b 4 -n single
python train_resnet34.py -t 4 -b 8 -n single
python train_resnet34.py -t 4 -b 16 -n single
python train_resnet34.py -t 4 -b 32 -n single

python train_resnet34.py -t 8 -b 1 -n single
python train_resnet34.py -t 8 -b 4 -n single
python train_resnet34.py -t 8 -b 8 -n single
python train_resnet34.py -t 8 -b 16 -n single
python train_resnet34.py -t 8 -b 32 -n single

python train_resnet34.py -t 16 -b 1 -n single
python train_resnet34.py -t 16 -b 4 -n single
python train_resnet34.py -t 16 -b 8 -n single
python train_resnet34.py -t 16 -b 16 -n single
python train_resnet34.py -t 16 -b 32 -n single

python train_resnet34.py -t 32 -b 1 -n single
python train_resnet34.py -t 32 -b 4 -n single
python train_resnet34.py -t 32 -b 8 -n single
python train_resnet34.py -t 32 -b 16 -n single
python train_resnet34.py -t 32 -b 32 -n single

python train_resnet34.py -t 128 -b 1 -n single
python train_resnet34.py -t 128 -b 4 -n single
python train_resnet34.py -t 128 -b 8 -n single
python train_resnet34.py -t 128 -b 16 -n single
python train_resnet34.py -t 128 -b 32 -n single

python train_resnet34.py -t 256 -b 1 -n single
python train_resnet34.py -t 256 -b 4 -n single
python train_resnet34.py -t 256 -b 8 -n single
python train_resnet34.py -t 256 -b 16 -n single
python train_resnet34.py -t 256 -b 32 -n single

# Instead of through the interactive 'notebook', this has to run in interactive terminal mode with
# qsub-gpu -q single-gpu -t 60 -I -n 1 -A ALCFAITP --attrs filesystems=home,grand
# or
# qsub-gpu -q full-node -t 150 -I -n 1 -A ALCFAITP --attrs filesystems=home,grand
# module load conda/2022-07-01
# conda activate
# clear; python ./ai-science-training-series/05_dataPipelines/05_homework/train_resnet34.py
# Note that on the interactive thetaGPU, you need to enter both 'module load conda/2022-07-01' AND 'conda activate' to set the environment.

# Start tensorboard to visualize data flow between GPU and CPU.
# tensorboard --logdir ./logdir/ --port 9993 --bind_all