#!/usr/bin/env bash

#### assumes models/ dir in project root #####
#### args 1 (optional) - MODELS_DIR ####

MODELS_DIR="./models"
if [ $# -gt 0 ]; then
  MODELS_DIR="$1"
fi

mkdir -p $MODELS_DIR

cd $MODELS_DIR

# # as referenced - https://www.tensorflow.org/lite/models/object_detection/overview
# # https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v1/1/default/1
# MODEL_URI="https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite"

# curl -O $MODEL_URI  # ssd_mobilenet_v1_1_default_1.tflite

# includes labels, but they are obselete
MODEL_FILE="coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
curl -O "http://storage.googleapis.com/download.tensorflow.org/models/tflite/$MODEL_FILE"
unzip -o $MODEL_FILE
rm $MODEL_FILE
rm labelmap.txt

# as referenced - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
MODEL_FILE="ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz"
MODEL_URI="http://download.tensorflow.org/models/object_detection/$MODEL_FILE"

curl -O $MODEL_URI
tar xzvf $MODEL_FILE
rm $MODEL_FILE

echo -e "TFLite models downloaded to $MODELS_DIR"
