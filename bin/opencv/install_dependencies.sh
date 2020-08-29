#! /usr/bin/env bash

# 32 bit versions of linked docs contain diff packages, install all
# (too lazy to check for duplicates)

sudo apt-get update
sudo apt-get upgrade -y

# https://qengineering.eu/install-opencv-4.4-on-raspberry-64-os.html
sudo apt-get install -y build-essential cmake git unzip pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgtk2.0-dev libcanberra-gtk* libgtk-3-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y python3-dev python3-numpy python3-pip
sudo apt-get install -y libtbb2 libtbb-dev libdc1394-22-dev
sudo apt-get install -y libv4l-dev v4l-utils
sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install -y liblapack-dev gfortran libhdf5-dev
sudo apt-get install -y libprotobuf-dev libgoogle-glog-dev libgflags-dev
sudo apt-get install -y protobuf-compiler

# https://qengineering.eu/install-opencv-4.4-on-raspberry-pi-4.html
sudo apt-get install -y cmake gfortran
sudo apt-get install -y libjpeg-dev libtiff-dev libgif-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgtk2.0-dev libcanberra-gtk*
sudo apt-get install -y libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt-get install -y libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev
sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install -y libjasper-dev liblapack-dev libhdf5-dev
sudo apt-get install -y gcc-arm* protobuf-compiler

# other useful stuff
sudo apt-get install -y git curl wget python3.7 python3.7-venv python3.7-dev
