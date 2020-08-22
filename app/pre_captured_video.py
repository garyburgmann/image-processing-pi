""" Used to load pre-captured videos with opencv """
from typing import Generator

import cv2
import numpy as np
# from PIL import Image


class PreCapturedVideo:
    def __init__(self, video_path: str):
        self._video_path = video_path
        self._capture = cv2.VideoCapture(self._video_path)

    def frames(self) -> Generator[np.ndarray, None, None]:
        while(self._capture.isOpened()):
            ret, cv2_im = self._capture.read()
            if ret is True and cv2_im.any():
                yield cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                # yield Image.fromarray(img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self._capture.release()
        cv2.destroyAllWindows()
