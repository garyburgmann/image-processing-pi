""" module with shared helper funcs """
import argparse
from typing import List, Any, Tuple, Dict
import pickle
import time

import cv2
import numpy as np
from PIL import Image
import requests

from app.object_detection import ObjectDetection
from app.opencv_video_capture import OpenCVVideoCapture
from app.opencv_mot_loop import OpenCVMOTLoop
from app.config import (
    FLASK_SERVER,
    FLASK_PORT,
    FLASK_ENDPOINT,
    CLASSIFIER_INPUT_SHAPE,
    DEFAULT_ARGS
)


def run_classifier(
    img: np.ndarray,
    clf: ObjectDetection
) -> Tuple[List[Dict], int]:
    """ run_classifier then return result & number of detections """
    result = clf.exec(img)
    # rescaled_result = rescale_image(img, result)

    return result, len(result)


def rescale_image(img: np.ndarray, result: List[Dict]) -> List[Dict]:
    """ use frame's original shape to restore aspect ratio """
    if result:
        CAMERA_HEIGHT, CAMERA_WIDTH = img.shape[:2]
        # CAMERA_HEIGHT, CAMERA_WIDTH = img.size
        for bbox in result:
            ymin, xmin, ymax, xmax = bbox['bounding_box']

            xmin = xmin * CAMERA_WIDTH
            xmax = xmax * CAMERA_WIDTH
            ymin = ymin * CAMERA_HEIGHT
            ymax = ymax * CAMERA_HEIGHT

            bbox['bounding_box'] = [ymin, xmin, ymax, xmax]
    return result


def draw_boxes(
    img: np.ndarray,
    res: List[Dict],
) -> np.ndarray:
    """ use opencv to paint bounding boxes on original image """
    for bbox in res:
        ymin, xmin, ymax, xmax = [int(x) for x in bbox['bounding_box']]
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
            f'{label} | {confidence:.2f}',
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
        # return OpenCVVideoCapture(video_path=args.v)
        return OpenCVMOTLoop(dataset_dir=args.v)
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


def parse_args() -> argparse.Namespace:
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
        default=DEFAULT_ARGS.model_path
    )
    parser.add_argument(
        '-l',
        '--labels_path',
        type=str,
        help='path to labels for model',
        default=DEFAULT_ARGS.labels_path
    )
    parser.add_argument(
        '--target',
        type=str,
        help='target label',
        default=DEFAULT_ARGS.target
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        help='confidence threshold',
        default=DEFAULT_ARGS.threshold
    )
    parser.add_argument(
        '-n',
        '--num_threads',
        type=int,
        help='number of threads for processing',
        default=DEFAULT_ARGS.num_threads
    )
    parser.add_argument(
        '--camera',
        type=int,
        help='number of threads for processing',
        default=DEFAULT_ARGS.camera
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
    parser.add_argument(
        '-o',
        '--out',
        type=str,
        help='output text file in MOT format',
        default='/tmp/out.txt'
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='less output to console'
    )
    return parser.parse_args()


def get_classifier(args) -> ObjectDetection:
    """ build classifier class from args """
    return ObjectDetection(
        model_path=args.model_path,
        labels_path=args.labels_path,
        target_label=args.target,
        threshold=args.threshold,
        tflite_runtime=args.lite,
        num_threads=args.num_threads
    )


def create_bbox_dump(frame_idx: int, bbox_idx: int, bbox: Dict) -> str:
    """
    turn classifier output into:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>,
    <x>, <y>, <z>

    gt.txt appears to use int for:
    <bb_left>, <bb_top>, <bb_width>, <bb_height>
    examples from docs however format these to 3 decimal places:
    <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>
    """
    ymin, xmin, ymax, xmax = bbox['bounding_box']

    return (
        f"{frame_idx},"
        "-1,"  # ,f"{bbox_idx}," # f"{bbox['class_id']}, "
        f"{xmin:.3f},"
        f"{ymin:.3f},"
        f"{(xmax - xmin):.3f},"
        f"{(ymax - ymin):.3f},"
        f"{bbox['score']:.3f},"
        "-1,-1,-1"  # <x>, <y>, <z>
    )


def dump_results(all_results: List[List[Dict]], to_file: str) -> None:
    """
    write out .txt file in format defined by MOT Challenge
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>,
    <x>, <y>, <z>
    """
    with open(to_file, 'w+') as f:
        for frame_idx, frame in enumerate(all_results, start=1):
            for bbox_idx, bbox in enumerate(frame, start=1):
                f.write(f"{create_bbox_dump(frame_idx, bbox_idx, bbox)}\n")
        print(f'results written to: {to_file}')


def detect_api(input_frame: np.ndarray) -> Tuple[List[Dict], int]:
    """ use api for object detection """
    res = requests.post(
        f'{FLASK_SERVER}:{FLASK_PORT}/{FLASK_ENDPOINT}',
        data=pickle.dumps(input_frame)
    )
    res.raise_for_status()
    return pickle.loads(res.content)


def descale_image(frame: np.ndarray) -> np.ndarray:
    """ shrink image to size required for object detection """
    return cv2.resize(
        frame,
        (CLASSIFIER_INPUT_SHAPE.height, CLASSIFIER_INPUT_SHAPE.width)
    )
