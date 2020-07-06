import time
import os
import argparse
import json
import copy

import requests
import cv2

import pedestrian_detection_pi as pdp


def run(**kwargs):
    frame = kwargs['frame']
    filename = kwargs['filename']
    img = kwargs['img']

    # p.o.c. running DO droplet with 2 GB memory, 1 vCPU
    if kwargs['endpoint'] == 'api':
        # using api hosted with digital ocean
        res = requests.post(
            'https://pd.garyburgmann.com/classify',
            json=img.tolist()
        )
        # print(res.json())
        res.raise_for_status()
        res = res.json()['result']
    else:
        # tf lite model from local module
        res = pdp.run(img)

    bbox_list = []
    conf_list = []
    if res:
        bbox_list = [x['bounding_box'] for x in res]
        print('boxe(s) found: {}'.format(bbox_list))
        conf_list = [x['score'] for x in res]

    CAMERA_HEIGHT, CAMERA_WIDTH = img.shape[:2]
    for idx, (ymin, xmin, ymax, xmax) in enumerate(bbox_list):
        # ymin, xmin, ymax, xmax = box
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)
        confidence = conf_list[idx]

        # draw bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (125, 255, 51), 2)
        # draw label above bounding box
        cv2.putText(
            img,
            f'Person | {confidence:.2f}',
            (xmin, ymin - 20),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # with open(f'mot17_06/results_{filename}.txt', 'a+') as f:
        #     f.write(
        #         f'{frame},-1,{xmin},{ymin},{xmax - xmin},{ymax - ymin},'
        #         f'{confidence}\n'
        #     )

    if True:
        cv2.imshow('img', img[:, :, ::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run people detection')
    parser.add_argument(
        '-v',
        type=str,
        help='path to video file'
    )
    parser.add_argument(
        '-e',
        type=str,
        help='api or local',
        default='local'
    )

    args = parser.parse_args()

    videofile = args.v
    cap = cv2.VideoCapture(videofile)

    start = time.time()
    num_frames = 1
    t = 0
    while(cap.isOpened()):
        ret, cv2_im = cap.read()
        if ret is True:
            frame = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            run(img=frame, frame=num_frames, filename=start, endpoint=args.e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            t = time.time() - start
            print(
                f'Time: {int(t)}, '
                f'Frames: {num_frames}, '
                f'FPS: {int(num_frames/t)}'
            )
            num_frames += 1

    cap.release()
    cv2.destroyAllWindows()
