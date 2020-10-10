#!/usr/bin/env python3
""" module to use ObjectDetection class with pre-captured video files """
import time
from typing import List, Dict
import pickle
import uuid

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

    if args.redis or args.redis_server:
        # prevents exception if this is conditional and not running redis
        r = get_redis()
        # redis server needs to be updated to match api changes before using
        # if args.redis_server:
        #     p = r.pubsub(ignore_subscribe_messages=True)
        #     p.subscribe('ack')
        #     r.set('threshold', args.threshold)
        if args.redis:
            r.set('q1', args.threshold)
            r.set('q2', args.threshold)
            r.set('q3', args.threshold)
            r.set('q4', args.threshold)

    # clf redundant
    if not (args.api or args.tensorflow_serving or args.redis_server):
        clf = get_classifier(args)

    total_detections = 0
    all_results: List[List[Dict]] = []
    thresholds_history = []
    time_from_start = 0
    start = time.time()
    for frame_idx, frame in enumerate(video.frames(), 1):
        inference_start = time.time()

        # dynamically do this by model instead - adds latency for apis tho!
        # frame = descale_image(frame)
        # print('frame.shape: ', frame.shape)

        if args.api or args.tensorflow_serving:
            # classify via api
            results, num_boxes, thresholds = detect_api(frame, args=args)
        # redis server needs to be updated to match api changes before using
        # elif args.redis_server:
        #     r.publish('det', frame)
        #     msg = p.get_message()
        #     while not msg:
        #         msg = p.get_message()
        #         if msg:
        #             results, num_boxes = msg['data']
        else:
            # classify locally
            results, num_boxes, thresholds = \
                run_classifier(frame=frame, clf=clf)

        if frame_idx % args.server_modulus == 0 and args.celery:
            _ = detect_and_compare.delay(
                frame,
                args,
                results,
                thresholds,
                frame_idx
            )

        thresholds_history.append(thresholds)
        inference_end = time.time()

        # this is not done as part of quandrant checking
        # results = rescale_results(frame, results)

        # 2 dimensional by design
        all_results.append(results)

        if args.display:
            frame = draw_boxes(frame, results)
            # cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # below is faster
            cv2.imshow('Frame', frame[:, :, ::-1])

        total_detections += num_boxes

        time_from_start = time.time() - start

        if not args.quiet:
            print('inference time: ', inference_end - inference_start)
            log_metrics(
                time_from_start,
                frame_idx,
                num_boxes,
                total_detections,
                results
            )
            print('console output time: ', time.time() - inference_end)

    try:
        # frame_idx might not exist
        idx = frame_idx
    except Exception:
        idx = 0
    pk = str(uuid.uuid4())
    filename_thresholds = f'./out/{pk}.txt'
    with open('./out/results.txt', 'a+') as f:
        f.write(
            f'Dataset: {args.v}, '
            f'Time: {int(time_from_start)}, '
            f'Frames: {idx}, '
            f'FPS: {int(idx/time_from_start)}, '
            f'Total Det: {total_detections}\n'
        )
        f.write(
            f'Args: {args}\n'
            f'Thresholds: {filename_thresholds}\n'
        )
    with open(filename_thresholds, 'w+') as f:
        for x in thresholds_history:
            f.write(f'{x}\n')

    dump_results(all_results, args.out)
