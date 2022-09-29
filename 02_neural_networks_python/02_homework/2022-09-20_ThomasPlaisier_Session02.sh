#!/bin/sh

module load /lus/theta-fs0/software/datascience/conda/2021-09-22

CWD="$(pwd)"

# These are needed to make sure that theta can access (at least) the Keras API with the unverified SSL connection.
export http_proxy=theta-proxy.tmi.alcf.anl.gov:3128
export https_proxy=theta-proxy.tmi.alcf.anl.gov:3128

/lus/theta-fs0/software/datascience/conda/2021-09-22/mconda3/bin/python 2022-09-20_ThomasPlaisier_Session02.py
