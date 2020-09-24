#!/usr/bin/env bash
MODEL=$1
# CMD="python object_detection/export_tflite_ssd_graph.py"
CMD="python object_detection/export_tflite_graph_tf2.py"
CMD="$CMD --pipeline_config_path /mnt/models/$MODEL/pipeline.config"
CMD="$CMD --trained_checkpoint_dir /mnt/models/$MODEL/checkpoint"
CMD="$CMD --output_directory /mnt/models/$MODEL/tflite"
CMD="$CMD --ssd_max_detections 50"
# CMD="$CMD --ssd_use_regular_nms=true"
# CMD="$CMD --input_shape=1,640,640,3"
# CMD="$CMD --add_postprocessing_op=true"
# docker exec od $CMD
docker run --rm -v $PWD:/mnt model_research_object_detection $CMD
