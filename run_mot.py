import subprocess
import argparse
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run object for a whole MOT')
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        help='path to model',
        default='./models/ssd_mobilenet_v3_small_coco_2020_01_14/model.tflite'
    )
    parser.add_argument(
        '--mot',
        type=str,
        help='MOT to run'
    )
    parser.add_argument(
        '--lite',
        action='store_true',
        help='flag to use tflite model',
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='flag to use less console output',
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        help='confidence threshold',
        default=0.5
    )
    parser.add_argument(
        '--attempt',
        type=str,
        help='store in sub dir for multiple attempts per model',
        default='1'
    )
    parser.add_argument(
        '--tensorflow_serving',
        action='store_true',
        help='flag to use tensorflow serving api for detection',
    )
    parser.add_argument(
        '--class_id_offset',
        type=int,
        help='some models start class ids at 0, some at 1',
        default=0
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    mot_year = args.mot.split('/')[-1]

    out_dir = os.path.join(
        'out',
        args.model_path.split('/')[-2],
        mot_year,
        args.attempt
    )
    print('out_dir: ', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for subdir in ['train', 'test']:  # test doesn't have ground truth
        context_dir = os.path.join(args.mot, subdir)
        mots = sorted(os.listdir(context_dir))
        for mot in mots:
            mot_to_run = os.path.join(context_dir, mot)
            file_to_write = os.path.join(out_dir, f'{mot}.txt')

            print(f'now running {mot_to_run}')
            print(f'args: {args}')
            cmd = f'./main.py '
            cmd += f'-v {mot_to_run} '
            cmd += f'-o {file_to_write} '
            cmd += f'--threshold {args.threshold} '
            cmd += f'--model_path {args.model_path}'

            if args.quiet:
                cmd += ' --quiet'
            if args.lite:
                cmd += ' --lite'
            if args.tensorflow_serving:
                cmd += ' --tensorflow_serving'  # model will be ignored
            if args.class_id_offset:
                cmd += f' --class_id_offset {args.class_id_offset}'

            _ = subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
