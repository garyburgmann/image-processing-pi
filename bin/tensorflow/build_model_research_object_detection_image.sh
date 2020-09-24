#!/usr/bin/env bash
PATH_TO_TENSORFLOW_MODELS=$1
cd $PATH_TO_TENSORFLOW_MODELS
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t model_research_object_detection .
