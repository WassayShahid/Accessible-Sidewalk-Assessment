"""Crop the raw dataset to 640x640 around each label. The zips hold uncropped images, about 3 MB each.

Expects each Validated/<Category>.zip unzipped so that <raw-dir>/<Category>/crops and <raw-dir>/<Category>/csv exist.
Writes <output-dir>/<Category>/{train,test}/ and <output-dir>/<Category>/{train,test}.csv.
"""
import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image

from dataset import CATEGORIES

CROP_RADIUS = 320


def crop_image(source, destination, normalized_x, normalized_y):
    """Same crop as crop.py in sidewalk-tagger-ai. Labels near an edge give a smaller crop."""
    with Image.open(source) as image:
        width, height = image.size
        x = int(normalized_x * width)
        y = int(normalized_y * height)
        box = (max(0, x - CROP_RADIUS), max(0, y - CROP_RADIUS), min(width, x + CROP_RADIUS), min(height, y + CROP_RADIUS))
        image.crop(box).save(destination)


def prepare_split(raw_dir, output_dir, category, split, workers):
    csv_path = os.path.join(raw_dir, category, 'csv', f'{split}.csv')
    image_dir = os.path.join(raw_dir, category, 'crops')
    if not os.path.exists(csv_path) or not os.path.isdir(image_dir):
        raise FileNotFoundError(f'Expected {csv_path} and {image_dir}. Unzip Validated/{category}.zip into {raw_dir}.')

    frame = pd.read_csv(csv_path)
    split_dir = os.path.join(output_dir, category, split)
    os.makedirs(split_dir, exist_ok=True)
    shutil.copy(csv_path, os.path.join(output_dir, category, f'{split}.csv'))

    jobs, missing = [], 0
    for row in frame.itertuples():
        source = os.path.join(image_dir, row.filename)
        destination = os.path.join(split_dir, row.filename)
        if not os.path.exists(source):
            missing += 1
        elif not os.path.exists(destination):
            jobs.append((source, destination, row.normalized_x, row.normalized_y))

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda job: crop_image(*job), jobs))
    print(f'{category}/{split}: cropped {len(jobs)}, already done {len(frame) - len(jobs) - missing}, missing {missing}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)

    for category in CATEGORIES:
        for split in ('train', 'test'):
            prepare_split(args.raw_dir, args.output_dir, category, split, args.workers)


if __name__ == '__main__':
    main()
