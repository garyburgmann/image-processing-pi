""" Used to load MOT datasets with OpenCV """
from typing import Generator, List
import os
# import time

import cv2
import numpy as np


class OpenCVMOTLoop:
    _images_subdir = 'img1'  # appears in all datasets

    def __init__(self, dataset_dir: str):
        self._dataset_dir = dataset_dir
        self._images_dir = os.path.join(self._dataset_dir, self._images_subdir)
        self._images = self.get_image_list()

    def get_image_list(self) -> List[str]:
        """ returns sorted list of image names in dir """
        return sorted(os.listdir(self._images_dir))

    def frames(self) -> Generator[np.ndarray, None, None]:
        """ open images from dataset path and yield """
        for x in self._images:
            # start = time.time()
            im = cv2.imread(os.path.join(self._images_dir, x))
            # print('cv2.imread: ', time.time() - start)
            yield cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
