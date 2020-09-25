#!/usr/bin/env bash

#### tensorflow/serving inputs and outputs JSON #####
#### assumes models/ dir in project root #####
#### args 1 - MODEL #####
#### args 2 (optional): - PORT #####

CONTAINER_NAME=serving
IMAGE=tensorflow/serving:latest
docker pull $IMAGE

MODEL=$1
API_MODEL_NAME=default
PORT=8501
if [ $# -gt 1 ]; then
    PORT=$2
fi
# e.g.
# res = requests.post(
#     f'http://localhost:$PORT/v1/models/$API_MODEL_NAME:predict',
#     json={'instances': [input_frame.tolist()]}
# )

docker kill $CONTAINER_NAME
docker run -t --rm -p $PORT:8501 \
    -v "$PWD/models/$MODEL:/models/$API_MODEL_NAME" \
    -e MODEL_NAME=$API_MODEL_NAME \
    --name $CONTAINER_NAME \
    -d $IMAGE
docker logs -f $CONTAINER_NAME
