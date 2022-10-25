#!/bin/bash -l
#COBALT -q full-node
#COBALT -t 60
#COBALT -n 1
#COBALT -A ALCFAITP
#COBALT -M tp@u.northwestern.edu
#COBALT --attrs filesystems=home,grand

# Alternative queues:
# -q single-gpu
# -q training-gpu

# Make executable with 
# chmod +x pi.sh

# Submit with
# qsub-gpu pi.sh

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

# In-class: test for yourself and see how long this code takes for differnt numbers of threads.
# 1 Time: 8.476861000061035
# 2 Time: 4.546555042266846
# 4 Time: 2.2157328128814697
# 8 Time: 1.1081163883209229
# 16 Time: 0.595123291015625
# 64: Time: 0.19693541526794434
# 128 Time: 0.31934285163879395
# At most 16 threads available for single-gpu node. Training GPU nodes go up to 128.

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
mpirun -np 1 python pi.py
mpirun -np 2 python pi.py
mpirun -np 4 python pi.py
mpirun -np 8 python pi.py
mpirun -np 16 python pi.py
mpirun -np 32 python pi.py
mpirun -np 64 python pi.py
mpirun -np 128 python pi.py 
mpirun -np 256 python pi.py
