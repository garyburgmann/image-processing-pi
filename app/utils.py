""" module with shared helper funcs """
import argparse
from typing import List, Any, Tuple, Dict
import pickle
import time
import re
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import requests

from app.object_detection import ObjectDetection
from app.opencv_video_capture import OpenCVVideoCapture
from app.opencv_mot_loop import OpenCVMOTLoop
from app.config import (
    APP_SERVER,
    CLASSIFIER_INPUT_SHAPE,
    DEFAULT_ARGS,
    THRESHOLD_CONFIG,
    TENSORFLOW_SERVING
)
from app.redis import get_redis

r = get_redis()


def run_classifier(
    frame: np.ndarray,
    clf: ObjectDetection
) -> Tuple[List[Dict], int]:
    """ run_classifier then return result & number of detections """
    predictions = clf.exec(frame)['predictions'][0]

    count = int(predictions['num_detections'])

    results, thresholds = format_results(
        frame,
        count,
        predictions['detection_boxes'],
        predictions['detection_scores'],
        predictions['detection_classes'],
        clf.labels,
        clf.target_label,
        clf.threshold,
        class_id_offset=clf.class_id_offset  # classes start at 1, not zero
    )

    return results, len(results), thresholds


def rescale_bbox(bbox: List, height: int, width: int) -> List:
    """ change detected boxes back to original size """
    # ymin, xmin, ymax, xmax = bbox
    assert len(bbox) == 4
    return [
        coord * width
        if idx in [1, 3]  # x coords
        else coord * height  # y coords
        for idx, coord
        in enumerate(bbox)
    ]


def rescale_results(img: np.ndarray, result: List[Dict]) -> List[Dict]:
    """ use frame's original shape to restore aspect ratio to bboxes """
    if result:
        height, width = img.shape[:2]
        for bbox in result:
            bbox['bounding_box'] = rescale_bbox(
                bbox['bounding_box'],
                height,
                width
            )
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
        '--server_threshold',
        type=float,
        help='confidence threshold for server only',
        default=DEFAULT_ARGS.server_threshold
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
        '--server_modulus',
        type=int,
        help=(
            'the number of images to process before requesting server '
            'feedback asynchronously with celery'
        ),
        default=DEFAULT_ARGS.server_modulus
    )
    parser.add_argument(
        '--class_id_offset',
        type=int,
        help='some models start class ids at 0, some at 1',
        default=DEFAULT_ARGS.class_id_offset
    )
    parser.add_argument(
        '--class_id_offset_celery',
        type=int,
        help='some models start class ids at 0, some at 1',
        default=DEFAULT_ARGS.class_id_offset_celery
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
        '--api',
        action='store_true',
        help='classify with external api (http)',
    )
    parser.add_argument(
        '--tensorflow_serving',
        action='store_true',
        help='flag to use tensorflow serving api for detection',
    )
    parser.add_argument(
        '--celery',
        action='store_true',
        help='classify with backend celery process',
    )
    parser.add_argument(
        '--redis',
        action='store_true',
        help='classify with redis pub/sub',
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
        num_threads=args.num_threads,
        class_id_offset=args.class_id_offset
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


def detect_api(
    input_frame: np.ndarray,
    args: argparse.Namespace
) -> Tuple[List[Dict], int]:
    """ use api for object detection with tensoflow/serving Docker image """
    predictions = None
    if args.tensorflow_serving:
        res = requests.post(
            f'{TENSORFLOW_SERVING.remote_base_url}:{TENSORFLOW_SERVING.port}'
            f'/v1/models/{TENSORFLOW_SERVING.model_name}:predict',
            json={'instances': [input_frame.tolist()]}
        )
        res.raise_for_status()
        # returns JSON
        predictions = res.json()['predictions'][0]
    elif args.api:
        res = requests.post(
            f'{APP_SERVER.remote_base_url}:{APP_SERVER.port}'
            f'/{APP_SERVER.endpoint}',
            data=pickle.dumps(input_frame)
        )
        res.raise_for_status()
        # returns bytes
        predictions = pickle.loads(res.content)['predictions'][0]

    count = int(predictions['num_detections'])

    results, thresholds = format_results(
        input_frame,
        count,
        predictions['detection_boxes'],
        predictions['detection_scores'],
        predictions['detection_classes'],
        load_labels(args.labels_path),
        args.target,
        args.server_threshold,  # can use a different threshold to client
        class_id_offset=args.class_id_offset,  # classes start at 1, not zero
        is_server=True  # override quadrant calcs to use static threshold
    )

    return results, len(results), thresholds


def descale_image(frame: np.ndarray) -> np.ndarray:
    """ shrink image to size required for object detection """
    return cv2.resize(
        frame,
        (CLASSIFIER_INPUT_SHAPE.height, CLASSIFIER_INPUT_SHAPE.width)
    )


def map_class_to_label(class_id: int, labels: Dict) -> str:
    """
    labels may not be mapped or inconsistent
    - https://tech.amikelive.com/node-718/
      what-object-categories-labels-are-in-coco-dataset/
    """
    try:
        label = labels[class_id].lower()
        # print("label: ", label)
        return labels[class_id].lower()
    except:
        print("label missing for class_id: ", class_id)
        return '???'


def get_single_result_dict(
    box: List,
    score: float,
    class_id: int,
    class_label: str
) -> Dict:
    """ return formatted dict for single detection from a frame """
    return {
        'bounding_box': box,
        'class_id': class_id,
        'class': class_label,
        'score': score
    }


def format_results(
    frame,
    count: int,
    boxes: List[List],
    scores: List,
    classes: List,
    labels: Dict,
    target_label: str,
    threshold: float,
    class_id_offset: int = DEFAULT_ARGS.class_id_offset,
    is_server: bool = False
) -> List[Dict]:
    """ declare here to share with multiple classification methods """
    target = target_label.lower()  # prevent casing typos
    thresholds = {}  # store retrieved values to lessen redis load
    height, width = frame.shape[:2]  # idx 3 == channels
    results = []
    for i in range(count):
        class_id = int(classes[i]) + class_id_offset
        class_label = map_class_to_label(class_id, labels)
        if target in [class_label, '__all__']:
            # rescale image first for get_quadrant_key 
            box = rescale_bbox(boxes[i], height, width)
            if is_server:
                # use a static threshold from args
                meets_threshold = scores[i] >= threshold  # bool
            else:
                # use quadrant technique for client
                # determine quadrant that box falls into, use to set threshold
                key = get_quadrant_key(frame, box)
                # this must be set in main.py, no need for exception handling
                thresholds[key] = thresholds.get(key, round(float(r.get(key)), 2))
                meets_threshold = scores[i] >= thresholds[key]
            if meets_threshold:
                # meets criteria, return result
                results.append(
                    get_single_result_dict(
                        box,
                        scores[i],
                        class_id,
                        class_label
                    )
                )
    
    print(f'format_results run against thresholds: {thresholds}')
    return results, thresholds  # return thresholds for async comparison
    # return [
    #     {
    #         'bounding_box': boxes[i],
    #         'class_id': int(classes[i]) + class_id_offset,
    #         'class': map_class_to_label(
    #             int(classes[i]) + class_id_offset,
    #             labels
    #         ),
    #         'score': scores[i]
    #     }
    #     for i in range(count)
    #     if target_label.lower()
    #     in [
    #         map_class_to_label(int(classes[i]) + class_id_offset, labels),
    #         '__all__'
    #     ]
    #     and scores[i] >= float(threshold)
    # ]


def load_labels(path: str) -> Dict:
    """
    Loads the labels file.
    Supports files with or without index numbers.
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def get_quadrant_key(frame: np.ndarray, bbox: List) -> str:
    """ decide on quadrant to update, assume q1 as default """
    mid_y, mid_x = get_mid_y_mid_x(frame)

    # assume min, i.e. q1, modify on falsey
    is_y_min = True
    is_x_min = True
    ymin, xmin, ymax, xmax = bbox

    if abs(ymax - mid_y) > abs(ymin - mid_y):
        # falls within q3 or q4, negates truthy
        is_y_min = False
    if abs(xmax - mid_x) > abs(xmin - mid_x):
        # falls within q2 or q4, negates truthy
        is_x_min = False

    key = 'q1'
    if is_y_min and not is_x_min:
        key = 'q2'
    elif not is_y_min and is_x_min:
        key = 'q3'
    elif not is_y_min and not is_x_min:
        key = 'q3'

    return key


def aggregate_sections(
    frame: np.ndarray,
    data: List[Dict]
) -> SimpleNamespace:
    """
    count total detections per section, defined as:

    q1, q2
    q3, q4

    equal distances belong to mininum
    e.g. a value directly between q1 and q2 belongs to q1
    """
    sections = SimpleNamespace(
        q1=0,
        q2=0,
        q3=0,
        q4=0
    )

    for d in data:  # already rescaled
        key = get_quadrant_key(frame, d['bounding_box'])
        sections.__dict__[key] += 1

    return sections


def get_quadrant_results(
    frame: np.ndarray,
    onboard: List[Dict],
    offboard: List[Dict],
    from_server: bool = False
) -> Tuple[SimpleNamespace, SimpleNamespace]:
    """ split into 4 quadrants and compare results from each """
    if not from_server:
        # pass thru rescaled images
        onboard_res = aggregate_sections(
            frame,
            onboard
            # rescale_results(frame, onboard)
        )
        # print('onboard_res: ', onboard_res.__dict__)

        offboard_res = aggregate_sections(
            frame,
            offboard
            # rescale_results(frame, offboard)
        )
        # print('offboard_res: ', offboard_res.__dict__)
    else:
        res = requests.post(
            f'{APP_SERVER.remote_base_url}:{APP_SERVER.port}'
            f'/quadrants',
            data=pickle.dumps([frame, onboard, offboard])
        )
        res.raise_for_status()
        # returns bytes
        onboard_res, offboard_res = pickle.loads(res.content)

    return onboard_res, offboard_res


def validate_threshold_divergence(
    current_value: float,
    original_value: float
) -> bool:
    """
    due to async server response, threshold may already have been adjusted

    this function is used to validate, with tolerance, permission to adjust
    """
    return abs(
        round(
            current_value - original_value,
            2
        )
    ) <= THRESHOLD_CONFIG.tolerance


def update_quadrant_threshold(
    key: str,
    original_thresholds: Dict,
    lower: bool = False,
    higher: bool = False
) -> float:
    """ assumes that key is set, as per main.py """
    global r  # redis
    val = round(float(r.get(key)), 2)  # cast bytes to float
    print(f'current val for key {key}: {val}')
    original_value = original_thresholds.get(key)  # TODO: may not exist?

    if not validate_threshold_divergence(val, original_value):
        raise Exception(
            f'current value ({val}) has diverged too far from original value '
            f'({original_value})'
        )

    if lower:
        val -= THRESHOLD_CONFIG.increment
    elif higher:
        val += THRESHOLD_CONFIG.increment

    new_val = round(val, 2)
    if 0 < new_val < 1:
        # catch async situation driving results out of range
        print(f'setting val for key {key}: {new_val}')
        r.set(key, new_val)
    else:
        # throw here so that we don't return
        raise Exception(
            f'new value ({new_val}) for key ({key}) has an invalid range, new '
            'value has not been set set'
        )

    return new_val


def compare_quadrants_from_results(
    onboard: SimpleNamespace,
    offboard: SimpleNamespace,
    original_thresholds: Dict
) -> SimpleNamespace:
    """ compare quadrants and update redis keys as required """
    updated_quadrants = SimpleNamespace()

    for key in onboard.__dict__.keys():
        val = None
        try:
            if getattr(onboard, key) > getattr(offboard, key):
                # too confident, raise threshold
                val = update_quadrant_threshold(
                    key,
                    original_thresholds,
                    higher=True
                )
            if getattr(onboard, key) < getattr(offboard, key):
                # not enough results found, lower threshold
                val = update_quadrant_threshold(
                    key,
                    original_thresholds,
                    lower=True
                )
        except Exception as exc:
            # thrown when value would be invalid - due to aysnc operation
            print(f'update_quadrant_threshold error: {exc}')
        if val:
            updated_quadrants.__dict__[key] = val

    return updated_quadrants


def get_mid_y_mid_x(frame: np.ndarray) -> Tuple[float, float]:
    """ get the shape of image array and return dimensions """
    y, x, num_channels = frame.shape

    mid_y, mid_x = y / 2, x / 2

    return mid_y, mid_x
