#! /usr/bin/env bash
# run as root
apt-get update && apt-get install -y \
    vim \
    swig \
    libjpeg-dev \
    zlib1g-dev \
    unzip \
    wget \
    curl \
    make \
    gcc \
    build-essential \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-numpy

apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    git

# https://pimylifeup.com/raspberry-pi-opencv/
apt-get install -y \
    libjpeg-dev \
    libtiff-dev \
    libjasper-dev \
    libpng-dev \
    libwebp-dev \
    libopenexr-dev

apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libdc1394-22-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev

apt-get install -y \
    libgtk-3-dev \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5

apt-get install -y \
    libatlas-base-dev \
    liblapacke-dev \
    gfortran

apt-get install -y \
    libhdf5-dev \
    libhdf5-103
