#!/usr/bin/env bash

#### requires build_model_research_object_detection_image.sh first ####
#### assumes models/ dir in project root #####
#### args 1 - MODEL ####
#### args 2 (optional) - IMAGE_NAME ####
MODEL=$1
IMAGE_NAME = model_research_object_detection
if [ $# -gt 1 ]; then
    IMAGE_NAME=$2
fi
# CMD="python object_detection/export_tflite_ssd_graph.py"
CMD="python object_detection/export_tflite_graph_tf2.py"
CMD="$CMD --pipeline_config_path /mnt/models/$MODEL/pipeline.config"
CMD="$CMD --trained_checkpoint_dir /mnt/models/$MODEL/checkpoint"
CMD="$CMD --output_directory /mnt/models/$MODEL/tflite"
CMD="$CMD --ssd_max_detections 50"
# CMD="$CMD --ssd_use_regular_nms=true"
# below are for TF1
# CMD="$CMD --input_shape=1,640,640,3"
# CMD="$CMD --add_postprocessing_op=true"
docker run --rm -v $PWD:/mnt $IMAGE_NAME $CMD
