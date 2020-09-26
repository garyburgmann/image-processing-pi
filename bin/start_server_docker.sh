#!/usr/bin/env bash
IMAGE_NAME=image-processing-pi-server
CONTAINER_NAME=image-processing-pi-server
docker kill $CONTAINER_NAME

docker run \
    --rm \
    --name \
    $CONTAINER_NAME \
    -v $PWD/server.py:/srv/server.py \
    -v $PWD/models:/srv/models \
    -v $PWD/labels:/srv/labels \
    -v $PWD/app:/srv/app \
    -p 8000:8000 \
    -d \
    $IMAGE_NAME
docker logs -f $CONTAINER_NAME
