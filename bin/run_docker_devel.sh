#!/usr/bin/env bash
# https://www.tensorflow.org/install/source#docker_linux_builds
TF_VERSION=devel
docker run --rm -it \
    -w /tensorflow_src \
    -v $PWD:/mnt \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e HOST_PERMS="$(id -u):$(id -g)" \
    tensorflow/tensorflow:${TF_VERSION} \
    bash -c "git pull && bash"
