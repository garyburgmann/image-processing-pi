#! /usr/bin/env bash

# ensure all dependencies loaded by forcing user to login again
su - $(whoami)

$CODE_DIR=~/code

# get opencv source
mkdir -p $CODE_DIR
cd $CODE_DIR
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip

# setup source folders
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.4.0 opencv
mv opencv_contrib-4.4.0 opencv_contrib
cd $CODE_DIR/opencv/
mkdir $CODE_DIR/opencv/build
