#!/bin/bash -l
#COBALT -q single-gpu
#COBALT -t 60
#COBALT -n 1
#COBALT -A ALCFAITP
#COBALT -M tp@u.northwestern.edu
#COBALT --attrs filesystems=home,grand

# Alternative queues:
# -q single-gpu
# -q training-gpu

# Make executable with 
# chmod +x train_resnet34_thetagpu.sh

# Submit with
# qsub-gpu train_resnet34_thetagpu.sh

# qsub-gpu -q single-gpu -t 60 -I -n 1 -A ALCFAITP --attrs filesystems=home,grand
# or
# qsub-gpu -q full-node -t 120 -I -n 1 -A ALCFAITP --attrs filesystems=home,grand
# or
# qsub-gpu -q training-gpu -t 60 -I -n 1 -A ALCFAITP --attrs filesystems=home,grand

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series
cd /home/soc/ai-science-training-series/06_distributedTraining/

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
mpirun -np 1 python train_resnet34_hvd.py --num_steps 10
mpirun -np 2 python train_resnet34_hvd.py --num_steps 10
mpirun -np 4 python train_resnet34_hvd.py --num_steps 10
mpirun -np 8 python train_resnet34_hvd.py --num_steps 10
mpirun -np 16 python train_resnet34_hvd.py --num_steps 10
