#! /usr/bin/env python3
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

from object_detection import ObjectDetection
from pre_captured_video import PreCapturedVideo


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
        '--model_path',
        type=str,
        help='path to model',
        default='/tmp/detect.tflite'
    )
    parser.add_argument(
        '-l',
        '--labels_path',
        type=str,
        help='path to labels for model',
        default='/tmp/labels_corrected.txt'
    )
    parser.add_argument(
        '-t',
        '--target_label',
        type=str,
        help='target labels',
        default='person'
    )
    parser.add_argument(
        '-c',
        '--confidence',
        type=float,
        help='confidence threshold',
        default=0.51
    )
    parser.add_argument(
        '-n',
        '--num_threads',
        type=int,
        help='number of threads for processing',
        default=4
    )
    parser.add_argument(
        '--lite',
        action='store_true',
        help='flag to use tflite model',
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='capture live video feed',
    )
    return parser.parse_args()


def get_video_source(args):
    """ load pre-captured video or use pi camera as source """
    if args.v:
        return PreCapturedVideo(args.v)
    elif args.live:
        from live_capture_video import LiveCaptureVideo
        # raise NotImplementedError('Camera not connected yet')
        return LiveCaptureVideo()
    else:
        raise ValueError('Missing argument for video path')


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

    print(args)
    # all args have defaults
    od = ObjectDetection(
        model_path=args.model_path,
        labels_path=args.labels_path,
        target_label=args.target_label,
        threshold=args.confidence,
        tflite_runtime=args.lite,
        num_threads=args.num_threads
    )

    # print(args)

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
