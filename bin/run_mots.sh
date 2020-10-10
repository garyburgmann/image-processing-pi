#!/usr/bin/env bash

# for m in ssd_mobilenet_v3_small_coco_2020_01_14 ssd_mobilenet_v3_large_coco_2020_01_14
# do
#     for i in 0.5
#     do
#         ./bin/start_redis.sh
#         sleep 5
#         python run_mot.py \
#             -m ./models/$m/model.tflite \
#             --mot ~/mot_challenge/MOT17Det \
#             --attempt $i \
#             --threshold $i \
#             --num_threads 3 \
#             --celery
#     done
# done

# for m in ssd_mobilenet_v3_small_coco_2020_01_14 ssd_mobilenet_v3_large_coco_2020_01_14
# do
#     ./bin/tensorflow/run_tensorflow_serving_api.sh
#     ./bin/start_redis.sh
#     for i in 4
#     do
#        python run_mot.py \
#             -m ./models/$m/model.tflite \
#             --mot ~/mot_challenge/MOT17Det \
#             --attempt $i \
#             --threshold 0.4
#     done
# done

# for m in ssd_resnet50_v1_fpn_640x640
# do
#     ./bin/tensorflow/run_tensorflow_serving_api.sh $m
#     sleep 5
#     for i in 1 2 3
#     do
#        python run_mot.py \
#             -m ./models/$m/jedi \
#             --mot ~/mot_challenge/MOT17Det \
#             --tensorflow_serving \
#             --attempt $i \
#             --class_id_offset -1
#     done
# done

#  ssd_resnet50_v1_fpn_640x640_coco17_tpu
for m in efficientdet_d3_coco17_tpu-32
do
    for i in 0.2 0.3 0.4 0.5 0.6 0.7
    do
       python run_mot.py \
            -m ./models/$m/saved_model \
            --mot ~/mot_challenge/MOT17Det \
            --api \
            --attempt $i \
            --class_id_offset -1 \
            --server_threshold $i \
            --redis
    done
done
