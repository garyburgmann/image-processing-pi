#!/usr/bin/env python3
""" module to use ObjectDetection class with pre-captured video files """
import time
from typing import List, Dict
# from concurrent.futures import ThreadPoolExecutor
import pickle

import cv2

from app.utils import (
    run_classifier,
    draw_boxes,
    get_video_source,
    log_metrics,
    parse_args,
    get_classifier,
    dump_results,
    detect_api,
)
from app.tasks import detect_and_compare
from app.redis import get_redis


if __name__ == '__main__':
    args = parse_args()

    video = get_video_source(args)

    # if args.redis:
    #     r = get_redis()
    #     p = r.pubsub(ignore_subscribe_messages=True)
    #     p.subscribe('ack')
    #     r.set('threshold', args.threshold)

    # TODO: perform this conditionally
    r = get_redis()
    r.set('q1', args.threshold)
    r.set('q2', args.threshold)
    r.set('q3', args.threshold)
    r.set('q4', args.threshold)
        # print(args)
        # all args have defaults

    clf = get_classifier(args)

    total_detections = 0
    previous_zero = False
    all_results: List[List[Dict]] = []
    # executor = ThreadPoolExecutor()
    start = time.time()
    for frame_idx, frame in enumerate(video.frames(), 1):
        # if previous_zero:
        #     # skip a detection if a previous frame was zero
        #     previous_zero = False
        # else:

        inference_start = time.time()

        # print('frame.shape: ', frame.shape)
        # input_frame = descale_image(frame)
        input_frame = frame
        # print('input_frame.shape: ', input_frame.shape)

        if args.redis:
            # TODO: remove test code
            r.publish('det', input_frame)
            msg = p.get_message()
            while not msg:
                msg = p.get_message()
                if msg:
                    results, num_boxes = msg['data']

        # split between api and local with threads
        # if frame_idx % 10 == 0:
        #     x = executor.submit(detect_api, input_frame)
        # else:
        #     x = executor.submit(run_classifier, img=input_frame, clf=clf)
        # results, num_boxes, thresholds = x.result()

        if args.api or args.tensorflow_serving:
            # classify via api
            results, num_boxes, thresholds = detect_api(input_frame, args=args)
        else:
            # classify locally
            results, num_boxes, thresholds = \
                run_classifier(frame=input_frame, clf=clf)

        if frame_idx % args.server_modulus == 0 and args.celery:
            _ = detect_and_compare.delay(
                input_frame,
                args,
                results,
                thresholds,
                frame_idx
            )

        # classify locally threaded
        # x = executor.submit(run_classifier, img=input_frame, clf=clf)
        # results, num_boxes = x.result()
        # print('x.result(): ', x.result())

        print('inference time: ', time.time() - inference_start)

        # results = rescale_results(frame, results)

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
        time_from_start = time.time() - start

        if not args.quiet:
            log_metrics(
                time_from_start,
                frame_idx,
                num_boxes,
                total_detections,
                results
            )

    with open('./results.txt', 'a+') as f:
        f.write(
            f'Dataset: {args.v}, '
            f'Time: {int(time_from_start)}, '
            f'Frames: {frame_idx}, '
            f'FPS: {int(frame_idx/time_from_start)}, '
            f'Total Det: {total_detections}\n'
        )
    dump_results(all_results, args.out)
