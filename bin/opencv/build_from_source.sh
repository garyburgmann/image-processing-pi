#! /usr/bin/env bash

# ensure all dependencies loaded by forcing user to login again
su - $(whoami)

$CODE_DIR=~/code

cd $CODE_DIR

# build using a python venv, for python 3.7
python3.7 -m venv opencv-venv --clear
. opencv-venv/bin/activate

# prepare venv for build
pip install --upgrade pip setuptools wheel numpy

# go to source build dir
cd $CODE_DIR/opencv/build

cmake .. \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D ENABLE_NEON=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_TBB=ON \
    -D BUILD_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=OFF \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_VTK=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=TRUE \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF

make -j4

sudo make install
sudo ldconfig
sudo apt-get update

cd $CODE_DIR
ln -s /usr/local/lib/python3.7/site-packages/cv2/python-3.7/cv2.cpython-37m-aarch64-linux-gnu.so

echo "output cv2 build symlink to: ${CODE_DIR}/cv2.cpython-37m-arm-linux-gnueabihf.so"
