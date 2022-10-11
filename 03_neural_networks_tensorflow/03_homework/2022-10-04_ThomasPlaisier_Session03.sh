#!/bin/sh
#COBALT -A ALCFAITP 
#COBALT -q full-node
#COBALT -n 1
#COBALT -t 60 
#COBALT -M tp@u.northwestern.edu
#COBALT --attrs filesystems=home

# Make executable with 
# chmod +x 2022-10-04_ThomasPlaisier_Session03.sh

# Submit with
# qsub-gpu 2022-10-04_ThomasPlaisier_Session03.sh

module load /lus/theta-fs0/software/datascience/conda/2021-09-22

CWD="$(pwd)"

# These are needed to make sure that theta can access (at least) the Keras API with the unverified SSL connection.
export http_proxy=theta-proxy.tmi.alcf.anl.gov:3128
export https_proxy=theta-proxy.tmi.alcf.anl.gov:3128

/lus/theta-fs0/software/datascience/conda/2021-09-22/mconda3/bin/python 2022-10-04_ThomasPlaisier_Session03.py -m adhd -o rmsprop -b 128 -e 100 -l 0.001 -s True
