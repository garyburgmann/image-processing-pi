""" module to use ObjectDetection class with pre-captured video files """
import time
import os
import argparse
import json
import copy

# import requests
import cv2
import numpy as np
from PIL import Image

from app.object_detection import ObjectDetection
from app.pre_captured_video import PreCapturedVideo
# from app.live_capture_video import LiveCaptureVideo

# these are my local working models from ./models
# adjust as necessary
MODEL_OPTS = [
    './models/ssd_mobilenet_v3_small_coco_2020_01_14/model.tflite',
    '/tmp/detect.tflite'

]
od = ObjectDetection(interpreter_path=MODEL_OPTS[1])


def run(img: Image.Image) -> int:
    np_img = np.array(img)

    res = od.exec(img)

    bbox_list = []
    conf_list = []

    CAMERA_HEIGHT, CAMERA_WIDTH = np_img.shape[:2]
    # CAMERA_HEIGHT, CAMERA_WIDTH = img.size

    if res:
        for bbox in res:
            ymin, xmin, ymax, xmax = bbox['bounding_box']

            xmin = int(xmin * CAMERA_WIDTH)
            xmax = int(xmax * CAMERA_WIDTH)
            ymin = int(ymin * CAMERA_HEIGHT)
            ymax = int(ymax * CAMERA_HEIGHT)

            confidence = bbox['score']
            label = bbox['class']

            # draw bounding box
            cv2.rectangle(np_img, (xmin, ymin), (xmax, ymax), (125, 255, 51), 2)
            # draw label above bounding box
            cv2.putText(
                np_img,
                f'{label.capitalize()} | {confidence:.2f}',
                (xmin, ymin - 20),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    if True:
        cv2.imshow('img', np_img[:, :, ::-1])

    return len(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run object detection')
    parser.add_argument(
        '-v',
        type=str,
        help='path to video file'
    )

    args = parser.parse_args()

    if args.v:
        video = PreCapturedVideo(args.v)
    else:
        raise ValueError('Missing argument for video path')
    #     video = LiveCaptureVideo

    start = time.time()
    num_frames = 1
    t = 0
    n = 0
    for frame in video.frames():
        num_detections = run(img=frame)
        n += num_detections
        t = time.time() - start
        print(
            f'Time: {int(t)}, '
            f'Frames: {num_frames}, '
            f'FPS: {int(num_frames/t)}, '
            f'Num Detections: {num_detections}, '
            f'N: {n}'
        )
        num_frames += 1
