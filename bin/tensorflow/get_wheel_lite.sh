#!/usr/bin/env bash

# install dependencies
sudo apt update && sudo apt install -y 
    swig libjpeg-dev zlib1g-dev python3-dev python3-numpy \
    unzip wget python3-pip curl git cmake make python3.7-dev

DOWNLOAD_SCRIPT=download_tflite_runtime-2.3.0-py3-none-linux_aarch64.whl.sh

wget https://raw.githubusercontent.com/PINTO0309/TensorflowLite-bin/master/2.3.0/$DOWNLOAD_SCRIPT

chmod +x $DOWNLOAD_SCRIPT
./$DOWNLOAD_SCRIPT

ls -alh | grep tensorflow
