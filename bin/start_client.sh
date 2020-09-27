#!/usr/bin/env bash

# ./main.py \
#     -v ~/mot_challenge/MOT17Det/train/MOT17-05/ \
#     -m models/detect.tflite \
#     -t 0.5 \
#     --class_id_offset_celery -1 \
#     --celery \
#     --num_threads 3

# ./main.py \
#     -v ~/mot_challenge/MOT17Det/train/MOT17-05/ \
#     -m models/ssd_mobilenet_v3_large_coco_2020_01_14/model.tflite \
#     -t 0.5 \
#     --class_id_offset_celery -1 \
#     --celery \
#     --num_threads 3

# ./main.py \
#     -v ~/mot_challenge/MOT17Det/train/MOT17-05/ \
#     -m models/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/model.tflite \
#     -t 0.5

# ./main.py \
#     -v ~/mot_challenge/MOT17Det/train/MOT17-05/ \
#     -m models/ssd_mobilenet_v3_large_coco_2020_01_14/model.tflite \
#     -t 0.5

./main.py \
    -v ~/mot_challenge/MOT17Det/train/MOT17-05/ \
    -m models/ssd_mobilenet_v3_small_coco_2020_01_14/model.tflite \
    -t 0.5

# ./main.py \
#     -v ~/mot_challenge/MOT17Det/train/MOT17-05/ \
#     -m models/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19/uint8/model.tflite \
#     -t 0.5
