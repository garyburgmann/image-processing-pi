#!/usr/bin/env bash

# install dependencies
sudo apt update && sudo apt install -y 
    libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran libgfortran5 \
    libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base \
    libblas-dev liblapack-dev  libatlas-base-dev openmpi-bin libopenmpi-dev \
    cython python-dev python3-dev python3.7-dev

DOWNLOAD_SCRIPT=tensorflow-2.3.0-cp37-cp37m-linux_aarch64_download.sh

wget https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/$DOWNLOAD_SCRIPT

chmod +x $DOWNLOAD_SCRIPT
./$DOWNLOAD_SCRIPT
ls -alh | grep tensorflow
