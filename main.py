#!/usr/bin/env python3
""" module to use ObjectDetection class with pre-captured video files """
import time
from typing import List, Dict

import cv2

from app.utils import (
    run_classifier,
    rescale_image,
    draw_boxes,
    get_video_source,
    log_metrics,
    parse_args,
    get_classifier,
    dump_results
)
from app.config import (
    FLASK_SERVER,
    FLASK_PORT,
    FLASK_ENDPOINT
)


if __name__ == '__main__':
    import requests
    import pickle
    args = parse_args()

    video = get_video_source(args)

    # print(args)
    # all args have defaults
    clf = get_classifier(args)

    start = time.time()
    num_frames = 0
    t = 0
    total_detections = 0
    previous_zero = False
    all_results: List[List[Dict]] = []
    for frame in video.frames():
        num_frames += 1
        # if previous_zero:
        #     # skip a detection if a previous frame was zero
        #     previous_zero = False
        # else:

        # send_via_socket(frame, num_frames)

        # res = requests.post(
        #     f'{FLASK_SERVER}:{FLASK_PORT}/{FLASK_ENDPOINT}',
        #     data=pickle.dumps(frame)
        # )
        # res.raise_for_status()
        # results, num_boxes = pickle.loads(res.content)

        results, num_boxes = run_classifier(img=frame, clf=clf)

        # print('results, num_boxes: ', results, num_boxes)

        # 2 dimensional by design
        all_results.append(results)

        if args.display:
            frame = draw_boxes(frame, results)

        if args.display:
            cv2.imshow(f'Frame: {num_frames}', frame[:, :, ::-1])

        total_detections += num_boxes
        # if num_boxes == 0:
        #     previous_zero = True
        t = time.time() - start

        if not args.quiet:
            log_metrics(
                t,
                num_frames,
                num_boxes,
                total_detections,
                results
            )

    with open('./results.txt', 'a+') as f:
        f.write(
            f'Dataset: {args.v}, '
            f'Time: {int(t)}, '
            f'Frames: {num_frames}, '
            f'FPS: {int(num_frames/t)}, '
            f'Total Det: {total_detections}\n'
        )
    dump_results(all_results, args.out)
