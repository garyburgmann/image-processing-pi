FROM arm32v7/python:3.7-buster

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libc-ares-dev \
    libeigen3-dev \
    gcc \
    git \
#     gfortran \
#     python-dev \
#     libgfortran5 \
    libatlas3-base \
    libatlas-base-dev \
    libopenblas-dev \
    libopenblas-base \
    libblas-dev \
    liblapack-dev \
#     cython \
    openmpi-bin \
    libopenmpi-dev \
    libatlas-base-dev

RUN pip install --upgrade pip setuptools
RUN pip install \
    keras_applications==1.0.8 --no-deps \
    keras_preprocessing==1.1.0 --no-deps \
    h5py==2.9.0 \
    pybind11 \
    six \
    wheel \
    mock
WORKDIR /tmp
RUN wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh"
RUN chmod +x tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh
RUN ./tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh
RUN pip install tensorflow-2.3.0-cp37-none-linux_armv7l.whl

WORKDIR /srv/app

RUN python -c "import tensorflow; tensorflow.__version__"
