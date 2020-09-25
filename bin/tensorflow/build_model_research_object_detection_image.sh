#!/usr/bin/env bash

#### https://github.com/tensorflow/models/tree/master/research/object_detection/dockerfiles/tf2 ####
#### args 1 - PATH_TO_TENSORFLOW_MODELS == https://github.com/tensorflow/models.git ####
#### args 2 (optional) - IMAGE_NAME ####

PATH_TO_TENSORFLOW_MODELS=$1
IMAGE_NAME = model_research_object_detection
if [ $# -gt 1 ]; then
    IMAGE_NAME=$2
fi
cd $PATH_TO_TENSORFLOW_MODELS
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t $IMAGE_NAME .
