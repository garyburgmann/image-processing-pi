#!/usr/bin/env bash
IMAGE_NAME=image-processing-pi-server
docker build -t $IMAGE_NAME -f server.Dockerfile .
