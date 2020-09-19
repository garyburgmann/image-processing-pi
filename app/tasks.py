from celery import Celery
import numpy as np

from app.utils import detect_api


app = Celery(
    'tasks',
    broker='redis://localhost:6379/1',
    accept_content=['pickle'],
    task_serializer='pickle'
)


@app.task(bind=True)
def detect(frame: np.ndarray, idx: int):
    print(f'{__name__} | idx: {idx}')
    results, num_boxes = detect_api(frame)
    print(f'{__name__} | num_boxes: {num_boxes}')
    return 0
