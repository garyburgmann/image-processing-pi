""" module to use ObjectDetection class with pre-captured video files """
import time
import os
import argparse
import json
import copy

# import requests
import cv2
import numpy as np

from object_detection import ObjectDetection

od = ObjectDetection()


def run(img: np.ndarray):
    img = kwargs['img']

    res = od.exec(img)

    bbox_list = []
    conf_list = []

    CAMERA_HEIGHT, CAMERA_WIDTH = img.shape[:2]

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
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (125, 255, 51), 2)
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

    if True:
        cv2.imshow('img', img[:, :, ::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run object detection')
    parser.add_argument(
        '-v',
        type=str,
        help='path to video file'
    )

    args = parser.parse_args()

    videofile = args.v
    cap = cv2.VideoCapture(videofile)

    start = time.time()
    num_frames = 1
    t = 0
    while(cap.isOpened()):
        ret, cv2_im = cap.read()
        if ret is True:
            frame = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            run(img=frame, frame=num_frames, endpoint=args.e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            t = time.time() - start
            print(
                f'Time: {int(t)}, '
                f'Frames: {num_frames}, '
                f'FPS: {int(num_frames/t)}'
            )
            num_frames += 1

    cap.release()
    cv2.destroyAllWindows()
