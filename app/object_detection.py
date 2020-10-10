""" Use TF Lite to detect objects """
import argparse
import io
import time
import subprocess
import os
from typing import List, Dict

import numpy as np
from PIL import Image
import cv2

from app.redis import get_redis
from app.config import DEFAULT_ARGS

r = get_redis()


class ObjectDetection:
    _input_details: None
    _output_details: None

    def __init__(
        self,
        model_path: str = '/tmp/detect.tflite',
        labels_path: str = '/tmp/labels_corrected.txt',
        target_label: str = 'person',
        threshold: float = 0.21,
        tflite_runtime: bool = False,
        num_threads: int = 4,
        class_id_offset: int = DEFAULT_ARGS.class_id_offset
    ):
        """ ObjectDetection constructor

        model_path: path to .tflite file
        """
        self._model_path = model_path
        self._labels_path = labels_path
        self.target_label = target_label
        self.threshold = threshold
        self._tflite_runtime = tflite_runtime
        self._num_threads = num_threads
        self.class_id_offset = class_id_offset
        self._validate_interpreter_and_labels_paths()
        self._interpreter = self._bootstrap_interpreter()
        self._prepare_interpreter()
        self.labels = self._load_labels()

    def _validate_interpreter_and_labels_paths(self) -> None:
        if not (
            os.path.exists(self._model_path)
            and os.path.exists(self._labels_path)
        ):
            raise Exception('missing interpeter and/or labels')

    def _bootstrap_interpreter(self):
        """ choose tf.lite submodule or flite_runtime """
        print(
            __name__, '| _bootstrap_interpreter | '
            'self._model_path:', self._model_path
        )
        if self._tflite_runtime:
            import tflite_runtime.interpreter as tflite
            return tflite.Interpreter(
                self._model_path,
                num_threads=self._num_threads  # TF 2.3 only - new feature
            )
        else:
            import tensorflow as tf
            # return tf.saved_model.load(self._model_path)    
            return tf.lite.Interpreter(
                self._model_path,
                num_threads=self._num_threads  # TF 2.3 only - new feature
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
        from app.utils import load_labels
        return load_labels(self._labels_path)

    def _set_input_tensor(self, image: np.ndarray) -> None:
        """ Sets the input tensor. """
        tensor_index = self._input_details[0]['index']
        # A function that can return a new numpy array pointing to the internal
        # TFLite tensor state at any point.
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image
        # self._interpreter.set_tensor(tensor_index, image)

    def _get_output_tensor(self, index: int) -> np.ndarray:
        """ Returns the output tensor at the given index. """
        output_index = self._output_details[index]['index']
        tensor = np.squeeze(self._interpreter.get_tensor(output_index))
        return tensor
        # return self._interpreter.get_tensor(output_index)
        # return self._interpreter.tensor(output_index)()

    def _detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Returns a list of detection results, each a dictionary of object
        info
        """
        # from app.utils import format_results

        # TODO:  this is a test, update me!
        # t = r.get('threshold')
        # print("r.get('threshold'): ", t)
        # self.threshold = float(t)

        self._set_input_tensor(image)
        self._interpreter.invoke()

        # Get all output details - use same format as tensorflow serving
        results = {
            'predictions': [
                {
                    'detection_boxes': self._get_output_tensor(0),
                    'detection_classes': self._get_output_tensor(1),
                    'detection_scores': self._get_output_tensor(2),
                    'num_detections': int(self._get_output_tensor(3)),
                }
            ]
        }

        # print(
        #     "results['predictions'][0]['num_detections']: ",
        #     results['predictions'][0]['num_detections']
        # )

        # return format_results(
        #     results['predictions'][0]['num_detections'],
        #     results['predictions'][0]['detection_boxes'],
        #     results['predictions'][0]['detection_scores'],
        #     results['predictions'][0]['detection_classes'],
        #     self.labels,
        #     self.target_label,
        #     self.threshold,
        #     self.class_id_offset
        # )

        return results

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        # print(
        #     "self._input_details[0]['shape']: ",
        #     self._input_details[0]['shape']
        # )
        _, input_height, input_width, _ = self._input_details[0]['shape']
        # image = Image.fromarray(img)
        # return image.resize((input_width, input_height), Image.ANTIALIAS)
        # the below is faster than PIL.Image.Image
        return cv2.resize(img, (input_width, input_height))

    def exec(self, img: np.ndarray) -> List[Dict]:
        try:
            # start_time = time.time()
            img = self._resize_image(img)
            # print('_resize_image time: ', time.time() - start_time)
            results = self._detect_objects(img)
        except Exception as exc:
            print(str(exc))
            results = []
        finally:
            return results
