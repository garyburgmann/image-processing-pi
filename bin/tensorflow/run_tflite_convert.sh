#!/usr/bin/env bash

#### use tf-nightly - try various tags: e.g. 2.4.0.dev20200906 #####
#### requires output from run_export_tflite_graph_tf2.sh ####

IMAGE=tensorflow/tensorflow:nightly
docker pull $IMAGE

MODEL=$1

CMD="tflite_convert"
CMD="$CMD --saved_model_dir=/mnt/models/$MODEL/tflite/saved_model"
CMD="$CMD --output_file=/mnt/models/$MODEL/tflite/saved_model/detect.tflite"
CMD="$CMD --experimental_new_converter=True"

docker run --rm -v $PWD:/mnt $IMAGE $CMD
