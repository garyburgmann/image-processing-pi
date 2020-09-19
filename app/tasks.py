from celery import Celery
import numpy as np

from app.utils import detect_api
from app.config import CELERY_CONFIG


app = Celery(__name__)
app.config_from_object(CELERY_CONFIG)


@app.task(bind=True)
def detect(self, frame: np.ndarray, idx: int):
    print(f'{__name__} | idx: {idx}')
    results, num_boxes = detect_api(frame)
    print(f'{__name__} | num_boxes: {num_boxes}')
    return 0
