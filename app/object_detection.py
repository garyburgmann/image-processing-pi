""" Use TF Lite to detect objects """
import argparse
import io
import re
import time
import subprocess
import os
from typing import List, Dict

import numpy as np
# import picamera
from PIL import Image
try:
    import tensorflow as tf
except Exception as e:
    print(e)
try:
    import tflite_runtime.interpreter as tflite
except Exception as e:
    print(e)


class ObjectDetection:
    _input_details: None
    _output_details: None

    def __init__(
        self,
        model_path: str = '/tmp/detect.tflite',
        labels_path: str = '/tmp/labels_corrected.txt',
        target_label: str = 'person',
        threshold: float = 0.51,
        tflite_runtime: bool = False,
        num_threads: int = 4
    ):
        """ ObjectDetection constructor

        model_path: path to .tflite file
        """
        self._model_path = model_path
        self._labels_path = labels_path
        self._target_label = target_label
        self._threshold = threshold
        self._tflite_runtime = tflite_runtime
        self._num_threads = num_threads
        self._validate_interpreter_and_labels_paths()
        self._interpreter = self._bootstrap_interpreter()
        self._prepare_interpreter()
        self._labels = self._load_labels()

    def _validate_interpreter_and_labels_paths(self) -> None:
        if not (
            os.path.exists(self._model_path)
            and os.path.exists(self._labels_path)
        ):
            raise Exception('missing interpeter and/or labels')

    def _bootstrap_interpreter(self):
        """ choose tf.lite submodule or flite_runtime """
        if self._tflite_runtime:
            return tflite.Interpreter(
                self._model_path,
                num_threads=self._num_threads
            )
        else:
            return tf.lite.Interpreter(
                self._model_path,
                num_threads=self._num_threads
            )

    def _prepare_interpreter(self) -> None:
        # https://www.tensorflow.org/lite/guide/inference
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def _load_labels(self) -> Dict:
        """
        Loads the labels file.
        Supports files with or without index numbers.
        """
        with open(self._labels_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row_number] = pair[0].strip()
        return labels

    def _set_input_tensor(self, image: Image) -> None:
        """ Sets the input tensor. """
        tensor_index = self._input_details[0]['index']
        # A function that can return a new numpy array pointing to the internal
        # TFLite tensor state at any point.
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, index: int) -> np.ndarray:
        """ Returns the output tensor at the given index. """
        output_index = self._output_details[index]['index']
        tensor = np.squeeze(self._interpreter.get_tensor(output_index))
        return tensor

    def _detect_objects(self, image: Image) -> List[Dict]:
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

        results = [
            {
                'bounding_box': boxes[i].tolist(),
                'class': self._labels[classes[i]].lower(),
                'score': float(scores[i])
            }
            for i in range(count)
            if self._target_label.lower()
            in [self._labels[classes[i]].lower(), '__all__']
            and scores[i] >= self._threshold
        ]
        return results

    def _resize_image(self, img: np.ndarray) -> Image.Image:
        image = Image.fromarray(img)
        _, input_height, input_width, _ = self._input_details[0]['shape']
        return image.resize((input_width, input_height), Image.ANTIALIAS)

    def exec(self, img: np.ndarray) -> List[Dict]:
        try:
            image = self._resize_image(img)
            # start_time = time.monotonic()
            results = self._detect_objects(image)
            # elapsed_ms = (time.monotonic() - start_time) * 1000
        except Exception as exc:
            print(str(exc))
            results = []
        finally:
            return results
