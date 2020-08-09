FROM arm32v7/python:3.7-buster

RUN apt-get update && apt-get install -y \
    swig \
    libjpeg-dev \
    zlib1g-dev \
    unzip \
    wget \
    curl \
    git \
    cmake \
    make \
    # python3-dev \
    # python3-pip \
    python3-numpy

RUN pip install --upgrade pip setuptools
WORKDIR /tmp
RUN wget https://github.com/PINTO0309/TensorflowLite-bin/raw/master/2.3.0/download_tflite_runtime-2.3.0-py3-none-linux_armv7l.whl.sh
RUN chmod +x download_tflite_runtime-2.3.0-py3-none-linux_armv7l.whl.sh
RUN ./download_tflite_runtime-2.3.0-py3-none-linux_armv7l.whl.sh
RUN pip install tflite_runtime-2.3.0-py3-none-linux_armv7l.whl

WORKDIR /srv/app
