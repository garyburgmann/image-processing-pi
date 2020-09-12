import subprocess
import argparse
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run object for a whole MOT')
    parser.add_argument(
        '-m',
        type=str,
        help='MOT to run'
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    mot_year = args.m.split('/')[-1]

    for each in ['test', 'train']:
        context_dir = os.path.join(args.m, each)
        mots = sorted(os.listdir(context_dir))
        for mot in mots:
            out_dir = os.path.join('out', mot_year, each)
            os.makedirs(out_dir, exist_ok=True)

            mot_to_run = os.path.join(context_dir, mot)
            file_to_write = os.path.join(out_dir, f'{mot}.txt')

            print(f'now running {mot_to_run}')
            _ = subprocess.call(
                f'./main.py -v {mot_to_run} -o {file_to_write} --quiet',
                shell=True
            )


if __name__ == '__main__':
    main()
