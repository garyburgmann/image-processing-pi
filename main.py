#!/usr/bin/env python3
""" module to use ObjectDetection class with pre-captured video files """
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

import cv2

from app.utils import (
    run_classifier,
    rescale_image,
    draw_boxes,
    get_video_source,
    log_metrics,
    parse_args,
    get_classifier,
    dump_results,
    detect_api,
    descale_image
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
    # executor = ThreadPoolExecutor()
    for frame in video.frames():
        num_frames += 1
        # if previous_zero:
        #     # skip a detection if a previous frame was zero
        #     previous_zero = False
        # else:

        inference = time.time()

        # print('frame.shape: ', frame.shape)
        input_frame = descale_image(frame)
        # print('input_frame.shape: ', input_frame.shape)

        # split between api and local with threads
        # if num_frames % 10 == 0:
        #     x = executor.submit(detect_api, input_frame)
        # else:
        #     x = executor.submit(run_classifier, img=input_frame, clf=clf)
        # results, num_boxes = x.result()

        # classify via api
        # results, num_boxes = detect_api(input_frame)

        # classify locally
        results, num_boxes = run_classifier(img=input_frame, clf=clf)

        # classify locally threaded
        # x = executor.submit(run_classifier, img=input_frame, clf=clf)
        # results, num_boxes = x.result()
        # print('x.result(): ', x.result())

        print('inference time: ', time.time() - inference)

        results = rescale_image(frame, results)
        # print('results, num_boxes: ', results, num_boxes)
        # 2 dimensional by design
        all_results.append(results)

        if args.display:
            frame = draw_boxes(frame, results)

        if args.display:
            # cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # below is faster
            cv2.imshow('Frame', frame[:, :, ::-1])

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
