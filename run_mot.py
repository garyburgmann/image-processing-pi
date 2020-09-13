import subprocess
import argparse
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run object for a whole MOT')
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
        default=0.51
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    mot_year = args.m.split('/')[-1]

    for subdir in ['train']:  # test doesn't have ground truth
        context_dir = os.path.join(args.m, subdir)
        mots = sorted(os.listdir(context_dir))

        for mot in mots:
            out_dir = os.path.join('out', mot_year)
            os.makedirs(out_dir, exist_ok=True)

            mot_to_run = os.path.join(context_dir, mot)
            file_to_write = os.path.join(out_dir, f'{mot}.txt')

            print(f'now running {mot_to_run}')
            cmd = f'./main.py -v {mot_to_run} -o {file_to_write} --threshold {args.threshold}'

            if args.quiet:
                cmd += ' --quiet'
            if args.lite:
                cmd += ' --lite'

            _ = subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
