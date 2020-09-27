#!/usr/bin/env bash

#### assumes models/ dir in project root #####
#### args 1 (optional) - MODELS_DIR ####

MODELS_DIR="./models"
if [ $# -gt 0 ]; then
  MODELS_DIR="$1"
fi

mkdir -p $MODELS_DIR

cd $MODELS_DIR

# as referenced - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
MODEL_FILE="efficientdet_d3_coco17_tpu-32.tar.gz"
MODEL_URI="http://download.tensorflow.org/models/object_detection/tf2/20200711/$MODEL_FILE"

curl -O $MODEL_URI
tar xzvf $MODEL_FILE
rm $MODEL_FILE

MODEL_FILE="ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
MODEL_URI="http://download.tensorflow.org/models/object_detection/tf2/20200711/$MODEL_FILE"

echo -e "TF models downloaded to $MODELS_DIR"
