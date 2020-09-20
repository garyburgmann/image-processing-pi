""" get some test images """
import os

DIR = '/Users/garyburgmann/mot_challenge/MOT17Det/train/MOT17-05/img1'

images = sorted(os.listdir(DIR))

print(images[:10])

"""
use tensorflow hub models

- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html
"""
import os
import time

import tensorflow as tf

MODEL_DIR= "../models"
MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"

PATH_TO_SAVED_MODEL = os.path.join(MODEL_DIR, MODEL_NAME, "saved_model")

print('Loading model...')
start_time = time.time()

# Load saved model and build the detection function
clf = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


import time
import io
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf


def format_results(detections: Dict, num_detections: int):
    # Get all output details
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']

    results = [
        {
            'bounding_box': boxes[i],
            'class_id': int(classes[i]),
            # 'class': self._labels[classes[i]].lower(),
            'score': scores[i]
        }
        for i in range(num_detections)
        if int(classes[i]) == 1
        # and scores[i] >= 0.51
    ]
    return results


for x in images[:5]:
    img = cv2.imread(f'{DIR}/{x}')
    converted_img  = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
    try:
        start = time.time()
        results = clf(converted_img)
        print('inference time: ', time.time() - start)
        # print('results: ', results)

        num_detections = int(results.pop('num_detections'))
        print('num_detections: ', num_detections)

        detections = {
            key: value[0, :num_detections].numpy()
            for key, value in results.items()
        }

        results = format_results(detections, num_detections)
        # print('\n\nresults: ', results)
    except Exception as exc:
        print('exc: ', exc)

#     cv2.imshow('frame', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
print("DONE!")