"""Example using TF Lite to detect objects with the Raspberry Pi camera."""
import argparse
import io
import re
import time

import numpy as np
# import picamera

from PIL import Image
import tensorflow as tf

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


class ObjectDetection:
    _input_details: None
    _output_details: None

    def __init__(self, interpreter_path: str, threshold: float = 0.6):
        """ ObjectDetection constructor

        interpreter_path: path to .tflite file
        """
        self._interpreter = tf.lite.Interpreter(interpreter_path)
        self._prepare_interpreter()

    def _prepare_interpreter(self):
        # https://www.tensorflow.org/lite/guide/inference
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def _set_input_tensor(self, image):
        """ Sets the input tensor. """
        tensor_index = self._input_details[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, index):
        """ Returns the output tensor at the given index. """
        output_index = self._output_details[index]['index']
        tensor = np.squeeze(self._interpreter.get_tensor(output_index))
        return tensor

    def _detect_objects(image):
        """
        Returns a list of detection results, each a dictionary of object
        info
        """
        self._set_input_tensor(image)
        self._interpreter.invoke()

        # Get all output details
        boxes = self._get_output_tensor(0)
        classes = self._get_output_tensor(1)
        scores = self._get_output_tensor(2)
        count = int(self._get_output_tensor(3))

        results = []
        for i in range(count):
            if scores[i] >= self._threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)
        return results

    def run(self, img_input):
        try:
            _, input_height, input_width, _ = self._input_details[0]['shape']
            image = (
                Image.fromarray(img_input)
                .resize(
                    (input_width, input_height),
                    Image.ANTIALIAS
                )
            )
            start_time = time.monotonic()
            results = self._detect_objects(image)
            elapsed_ms = (time.monotonic() - start_time) * 1000
        except Exception as exc:
            print(str(exc))
            results = []
        finally:
            return results


# def load_labels(path):
#     """Loads the labels file. Supports files with or without index numbers."""
#     with open(path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         labels = {}
#         for row_number, content in enumerate(lines):
#             pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
#             if len(pair) == 2 and pair[0].strip().isdigit():
#                 labels[int(pair[0])] = pair[1].strip()
#             else:
#                 labels[row_number] = pair[0].strip()
#     return labels

# def set_input_tensor(interpreter, image):
#     """Sets the input tensor."""
#     tensor_index = interpreter.get_input_details()[0]['index']
#     input_tensor = interpreter.tensor(tensor_index)()[0]
#     input_tensor[:, :] = image


# def get_output_tensor(interpreter, index):
#     """Returns the output tensor at the given index."""
#     output_details = interpreter.get_output_details()[index]
#     tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
#     return tensor


# def detect_objects(interpreter, image, threshold):
#     """Returns a list of detection results, each a dictionary of object info."""
#     set_input_tensor(interpreter, image)
#     interpreter.invoke()

#     # Get all output details
#     boxes = get_output_tensor(interpreter, 0)
#     classes = get_output_tensor(interpreter, 1)
#     scores = get_output_tensor(interpreter, 2)
#     count = int(get_output_tensor(interpreter, 3))

#     results = []
#     for i in range(count):
#         if scores[i] >= threshold:
#             result = {
#                 'bounding_box': boxes[i],
#                 'class_id': classes[i],
#                 'score': scores[i]
#             }
#             results.append(result)
#     return results


# def run(img_array):
#     # interpreter = tf.lite.Interpreter('/tmp/detect.tflite')
#     # threshold = 0.6
#     # interpreter.allocate_tensors()
#     # _, input_height, input_width, _ = \
#     #     interpreter.get_input_details()[0]['shape']

# #   with picamera.PiCamera(
# #       resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
# #     camera.start_preview()
#     try:
#     #   stream = io.BytesIO()
#     #   annotator = Annotator(camera)
#     #   for _ in camera.capture_continuous(
#     #       stream, format='jpeg', use_video_port=True):
#     #     stream.seek(0)
#         image = (
#             Image.fromarray(img_array)
#             .resize(
#                 (input_width, input_height),
#                 Image.ANTIALIAS
#             )
#         )
#         # image = Image.open(stream).convert('RGB').resize(
#         #     (input_width, input_height), Image.ANTIALIAS)
#         start_time = time.monotonic()
#         results = detect_objects(interpreter, image, threshold)
#         elapsed_ms = (time.monotonic() - start_time) * 1000
#         return results
#         # annotator.clear()
#         # annotate_objects(annotator, results, labels)
#         # annotator.text([5, 0], '%.1fms' % (elapsed_ms))
#         # annotator.update()

#         # stream.seek(0)
#         # stream.truncate()

#     finally:
#         pass
#     #   camera.stop_preview()

# # if __name__ == '__main__':
# #   main()
