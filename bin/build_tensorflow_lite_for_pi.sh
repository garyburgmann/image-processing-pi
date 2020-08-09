#!/usr/bin/env bash
# https://www.tensorflow.org/install/source#docker_linux_builds
# https://www.tensorflow.org/lite/guide/build_rpi

git checkout r2.3
git clone https://github.com/raspberrypi/tools.git /rpi_tools
./tensorflow/lite/tools/make/download_dependencies.sh
PATH=../rpi_tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin:$PATH ./tensorflow/lite/tools/make/build_rpi_lib.sh
