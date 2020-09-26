import argparse
from typing import List, Dict, Tuple
import copy

from celery import Celery
import numpy as np

from app.utils import (
    detect_api,
    get_quadrant_results,
    compare_quadrants_from_results
)
from app.config import CELERY_CONFIG, CELERY_API_SERVER


app = Celery(__name__)
app.config_from_object(CELERY_CONFIG)


@app.task(bind=True)
def detect_and_compare(
    self,
    frame: np.ndarray,
    args: argparse.Namespace,
    onboard: List[Dict],
    original_thresholds: Dict,  # invalidate results if too far diverged
    idx: int
):
    print(f'{__name__} | idx: {idx}')

    args_copy = copy.deepcopy(args)
    args_copy.__dict__[CELERY_API_SERVER] = True
    # override as using full model may require diff offset to onboard
    args_copy.class_id_offset = args_copy.class_id_offset_celery
    results, num_boxes, thresholds = detect_api(frame, args=args_copy)

    onboard_res, offboard_res = get_quadrant_results(frame, onboard, results)

    print(f'{__name__} | idx: {idx} | onboard_res: {onboard_res.__dict__}')
    print(f'{__name__} | idx: {idx} | offboard_res: {offboard_res.__dict__}')

    updated_quadrants = compare_quadrants_from_results(
        onboard_res,
        offboard_res,
        original_thresholds
    )
    print(
        f'{__name__} | idx: {idx} | updated_quadrants: '
        f'{updated_quadrants.__dict__}'
    )

    # print(f'{__name__} | num_boxes: {num_boxes}')
    return 0
