#! /usr/bin/env python3
""" module to use ObjectDetection class with pre-captured video files """
import time
import os
import argparse
import json
import copy
from typing import List, Any, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
import requests

from app.object_detection import ObjectDetection
from app.opencv_video_capture import OpenCVVideoCapture


def run_classifier(
    img: np.ndarray,
    clf: ObjectDetection
) -> Tuple[List[Dict], int]:
    """ run_classifier then return result & number of detections """
    result = clf.exec(img)
    rescaled_result = rescale_image(img, result)

    return rescaled_result, len(rescaled_result)


def rescale_image(img: np.ndarray, result: List[Dict]) -> List[Dict]:
    """ use frame's original shape to restore aspect ratio """
    if result:
        CAMERA_HEIGHT, CAMERA_WIDTH = img.shape[:2]
        # CAMERA_HEIGHT, CAMERA_WIDTH = img.size
        for bbox in result:
            ymin, xmin, ymax, xmax = bbox['bounding_box']

            xmin = int(xmin * CAMERA_WIDTH)
            xmax = int(xmax * CAMERA_WIDTH)
            ymin = int(ymin * CAMERA_HEIGHT)
            ymax = int(ymax * CAMERA_HEIGHT)

            bbox['bounding_box'] = [ymin, xmin, ymax, xmax]
    return result


def draw_boxes(
    img: np.ndarray,
    res: List[Dict],
) -> np.ndarray:
    """ use opencv to paint bounding boxes on original image """
    for bbox in res:
        ymin, xmin, ymax, xmax = bbox['bounding_box']
        confidence = bbox['score']
        label = bbox['class']

        # draw bounding box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            (125, 255, 51),
            2
        )
        # draw label above bounding box
        cv2.putText(
            img,
            f'{label.capitalize()} | {confidence:.2f}',
            (xmin, ymin - 20),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return img


def get_video_source(args):
    """ load pre-captured video or use camera as source """
    if args.v:
        return OpenCVVideoCapture(video_path=args.v)
    elif args.live:
        # raise NotImplementedError('Camera not connected yet')
        # from app.live_capture_video import LiveCaptureVideo
        # return LiveCaptureVideo()
        return OpenCVVideoCapture(live=args.live, camera=args.camera)
    else:
        raise ValueError('Missing argument for video path')


def log_metrics(
    t: int,
    n_frames: int,
    f_detections: int,
    t_detections: int,
    r: List[Dict]
):
    # TODO: use logging module
    print(
        f'Time: {int(t)}, '
        f'Frames: {n_frames}, '
        f'FPS: {int(n_frames/t)}, '
        f'Frame Detections: {f_detections}, '
        f'N: {t_detections}',
        f"Boxes: {[x['bounding_box'] for x in r]}\n\n"
    )


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
        '--camera',
        type=int,
        help='number of threads for processing',
        default=0
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
    parser.add_argument(
        '-d',
        '--display',
        action='store_true',
        help='show modified images',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    video = get_video_source(args)

    # print(args)
    # all args have defaults
    clf = ObjectDetection(
        model_path=args.model_path,
        labels_path=args.labels_path,
        target_label=args.target_label,
        threshold=args.confidence,
        tflite_runtime=args.lite,
        num_threads=args.num_threads
    )

    start = time.time()
    num_frames = 1
    t = 0
    total_detections = 0
    previous_zero = False
    for frame in video.frames():
        # if previous_zero:
        #     # skip a detection if a previous frame was zero
        #     previous_zero = False
        # else:
        results, num_boxes = run_classifier(img=frame, clf=clf)

        if args.display:
            frame = draw_boxes(frame, results)

        if args.display:
            cv2.imshow('frame', frame[:, :, ::-1])

        total_detections += num_boxes
        # if num_boxes == 0:
        #     previous_zero = True
        t = time.time() - start
        log_metrics(
            t,
            num_frames,
            num_boxes,
            total_detections,
            results
        )
        num_frames += 1
