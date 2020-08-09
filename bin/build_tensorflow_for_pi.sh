#!/usr/bin/env bash
# run as root on Debian Host

# https://www.tensorflow.org/install/source_rpi#python-3.7
cd /tmp
git clone https://github.com/tensorflow/tensorflow.git
cd /tmp/tensorflow
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON37 \
    tensorflow/tools/ci_build/pi/build_raspberry_pi.sh

# docker
# apt-get update && apt-get install -y \
#     apt-transport-https \
#     ca-certificates \
#     curl \
#     software-properties-common \
#     gpg-agent

# curl -fsSL get.docker.com -o get-docker.sh && sh get-docker.sh

# git pull
# git checkout master

# tensorflow/tools/ci_build/ci_build.sh PI-PYTHON37 \
#     tensorflow/tools/ci_build/pi/build_raspberry_pi.sh
