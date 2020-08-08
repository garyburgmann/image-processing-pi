""" module to use ObjectDetection class with pre-captured video files """
import time
import os
import argparse
import json
import copy
from typing import List, Any

import cv2
import numpy as np
from PIL import Image

from app.object_detection import ObjectDetection
from app.pre_captured_video import PreCapturedVideo
# from app.live_capture_video import LiveCaptureVideo


def run(img: Image.Image, od=ObjectDetection) -> int:
    res = od.exec(img)

    bbox_list = []
    conf_list = []

    np_img = display_boxes(img, res)

    return np_img, res, len(res)


def display_boxes(img: Image.Image, res: List[Any]) -> np.ndarray:
    if res:
        np_img = np.array(img)
        CAMERA_HEIGHT, CAMERA_WIDTH = np_img.shape[:2]
        # CAMERA_HEIGHT, CAMERA_WIDTH = img.size
        for bbox in res:
            ymin, xmin, ymax, xmax = bbox['bounding_box']

            xmin = int(xmin * CAMERA_WIDTH)
            xmax = int(xmax * CAMERA_WIDTH)
            ymin = int(ymin * CAMERA_HEIGHT)
            ymax = int(ymax * CAMERA_HEIGHT)

            confidence = bbox['score']
            label = bbox['class']

            # draw bounding box
            cv2.rectangle(
                np_img,
                (xmin, ymin),
                (xmax, ymax),
                (125, 255, 51),
                2
            )
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

    return np_img


def parse_args():
    parser = argparse.ArgumentParser(description='Run object detection')
    parser.add_argument(
        '-v',
        type=str,
        help='path to video file'
    )
    parser.add_argument(
        '-m',
        type=str,
        help='path to model'
    )
    return parser.parse_args()


def get_video_source(args):
    """ load pre-captured video or use pi camera as source """
    if args.v:
        return PreCapturedVideo(args.v)
    else:
        raise ValueError('Missing argument for video path')
    #     video = LiveCaptureVideo


def get_model_source(args):
    """ load tf lite model source, default /tmp/detect.tflite """
    if args.m:
        return args.m
    return '/tmp/detect.tflite'


def log_metrics(t: int, n_frames: int, f_detections: int, t_detections: int):
    # TODO: use logging
    print(
        f'Time: {int(t)}, '
        f'Frames: {n_frames}, '
        f'FPS: {int(n_frames/t)}, '
        f'Frame Detections: {f_detections}, '
        f'N: {t_detections}'
    )


if __name__ == '__main__':
    args = parse_args()
    video = get_video_source(args)
    model_path = get_model_source(args)

    od = ObjectDetection(interpreter_path=model_path)

    start = time.time()
    num_frames = 1
    t = 0
    total_detections = 0
    for frame in video.frames():
        _, _, frame_detections = run(img=frame, od=od)
        total_detections += frame_detections
        t = time.time() - start
        log_metrics(t, num_frames, frame_detections, total_detections)
        num_frames += 1
