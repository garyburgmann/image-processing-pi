from types import SimpleNamespace
import copy

DEFAULT_ARGS = SimpleNamespace(
    model_path='./models/ssd_mobilenet_v3_large_coco_2020_01_14/model.tflite',
    labels_path='./labels/coco_labels.txt',
    target='person',
    threshold=0.3,
    server_threshold=0.5,
    num_threads=4,
    camera=0,
    class_id_offset=0,
    server_modulus=30,
    class_id_offset_celery=-1  # for tensorflow_serving
)
APP_SERVER_OPTIONS = ['flask', 'falcon']
APP_SERVER = SimpleNamespace(
    type=APP_SERVER_OPTIONS[1],
    port=8000,
    host='0.0.0.0',
    debug=False,
    endpoint='predict',
    default_args=copy.deepcopy(DEFAULT_ARGS),
    remote_base_url='http://localhost'  # 'http://192.168.42.174'
)
assert APP_SERVER.type in APP_SERVER_OPTIONS
APP_SERVER.default_args.model_path = (
    './models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
    '/tflite/saved_model/detect.tflite'
)
TENSORFLOW_SERVING = SimpleNamespace(
    port=8501,
    remote_base_url='http://localhost',
    model_name='default'
)
CLASSIFIER_INPUT_SHAPE = SimpleNamespace(
    width=300,
    height=300
)
REDIS = SimpleNamespace(
    celery_broker_url='redis://localhost:6379/1',
    host='localhost',
    port=6379,
    db=0
)
CELERY_CONFIG = SimpleNamespace(
    broker_url=REDIS.celery_broker_url,
    task_serializer='pickle',
    accept_content=['pickle']
)
CELERY_API_SERVER_OPTIONS = ['api', 'tensorflow_serving']
CELERY_API_SERVER = CELERY_API_SERVER_OPTIONS[1]
assert CELERY_API_SERVER in CELERY_API_SERVER_OPTIONS
# THRESHOLD_INCREMENT = 0.05  # used to tune quadrants
THRESHOLD_CONFIG = SimpleNamespace(
    increment=0.05,
    tolerance=0.05
) 
