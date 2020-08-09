#!/usr/bin/env bash

if [ $# -eq 0 ]; then
  DATA_DIR="/tmp"
else
  DATA_DIR="$1"
fi

cd ${DATA_DIR}
curl -O  https://motchallenge.net/sequenceVideos/MOT17-05-SDP-raw.mp4
curl -O  https://motchallenge.net/sequenceVideos/MOT17-06-SDP-raw.mp4

echo -e "Files downloaded to ${DATA_DIR}"
