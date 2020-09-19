""" Used to load pre-captured videos with opencv """
from typing import Generator, Union

import cv2
import numpy as np


class OpenCVVideoCapture:
    def __init__(
        self,
        video_path: Union[str, None] = None,
        live: bool = False,
        camera: int = 0
    ):
        self._video_path = video_path
        self._live = live
        self._camera = camera
        self._capture = self.bootstrap_capture()
        # self._capture.set(3, 640)
        # self._capture.set(4, 480)

    def bootstrap_capture(self) -> cv2.VideoCapture:
        if self._live:
            return cv2.VideoCapture(self._camera)
        else:
            return cv2.VideoCapture(self._video_path)

    def frames(self) -> Generator[np.ndarray, None, None]:
        while(self._capture.isOpened()):
	
            ret, cv2_im = self._capture.read()
            if ret is True and cv2_im.any():
                yield cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self._capture.release()
        cv2.destroyAllWindows()
