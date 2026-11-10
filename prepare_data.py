"""Crop the raw dataset to 640x640 around each label. The zips hold uncropped 1440x960 PNGs, about 3 MB each.

Reads either unzipped folders (<raw-dir>/<Category>/crops and <raw-dir>/<Category>/csv) or, with --zip-dir,
the Validated/<Category>.zip files directly, so nothing has to be unzipped.
Writes <output-dir>/<Category>/{train,test}/ and <output-dir>/<Category>/{train,test}.csv.
"""
import argparse
import io
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image

from dataset import CATEGORIES

CROP_RADIUS = 320
JPEG_QUALITY = 95


def crop_image(source, destination, normalized_x, normalized_y):
    """Same crop as crop.py in sidewalk-tagger-ai. Labels near an edge give a smaller crop.
    source is a path or a file object. A .jpg destination is saved as RGB JPEG."""
    with Image.open(source) as image:
        width, height = image.size
        x = int(normalized_x * width)
        y = int(normalized_y * height)
        box = (max(0, x - CROP_RADIUS), max(0, y - CROP_RADIUS), min(width, x + CROP_RADIUS), min(height, y + CROP_RADIUS))
        crop = image.crop(box)
        if destination.lower().endswith('.jpg'):
            crop.convert('RGB').save(destination, quality=JPEG_QUALITY)
        else:
            crop.save(destination)


class FolderSource:
    """Images and CSVs from an unzipped Validated/<Category>.zip."""

    def __init__(self, raw_dir, category):
        self.root = os.path.join(raw_dir, category)
        if not os.path.isdir(os.path.join(self.root, 'crops')):
            raise FileNotFoundError(f'Expected {self.root}/crops and {self.root}/csv. Unzip Validated/{category}.zip into {raw_dir}.')

    def read_csv(self, split):
        return pd.read_csv(os.path.join(self.root, 'csv', f'{split}.csv'))

    def has_image(self, filename):
        return os.path.exists(os.path.join(self.root, 'crops', filename))

    def open_image(self, filename):
        return os.path.join(self.root, 'crops', filename)

    def close(self):
        pass


class ZipSource:
    """Images and CSVs read straight out of Validated/<Category>.zip. Each worker thread gets its own handle."""

    def __init__(self, zip_dir, category):
        self.path = os.path.join(zip_dir, f'{category}.zip')
        if not os.path.exists(self.path):
            raise FileNotFoundError(f'Expected {self.path}')
        self.category = category
        self.local = threading.local()
        self.handles = []
        self.names = set(self.zip().namelist())

    def zip(self):
        if not hasattr(self.local, 'zip'):
            self.local.zip = zipfile.ZipFile(self.path)
            self.handles.append(self.local.zip)
        return self.local.zip

    def read_csv(self, split):
        return pd.read_csv(io.BytesIO(self.zip().read(f'{self.category}/csv/{split}.csv')))

    def has_image(self, filename):
        return f'{self.category}/crops/{filename}' in self.names

    def open_image(self, filename):
        return io.BytesIO(self.zip().read(f'{self.category}/crops/{filename}'))

    def close(self):
        for handle in self.handles:
            handle.close()


def output_name(filename, jpeg):
    return os.path.splitext(filename)[0] + '.jpg' if jpeg else filename


def prepare_split(source, output_dir, category, split, workers, jpeg=False):
    frame = source.read_csv(split)
    split_dir = os.path.join(output_dir, category, split)
    os.makedirs(split_dir, exist_ok=True)

    jobs, missing = [], 0
    for row in frame.itertuples():
        destination = os.path.join(split_dir, output_name(row.filename, jpeg))
        if not source.has_image(row.filename):
            missing += 1
        elif not os.path.exists(destination):
            jobs.append((row.filename, destination, row.normalized_x, row.normalized_y))

    def run(job):
        filename, destination, normalized_x, normalized_y = job
        crop_image(source.open_image(filename), destination, normalized_x, normalized_y)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, jobs))

    # Written last, so a CSV in output_dir means the split finished.
    frame['filename'] = [output_name(name, jpeg) for name in frame['filename']]
    frame.to_csv(os.path.join(output_dir, category, f'{split}.csv'), index=False)
    print(f'{category}/{split}: cropped {len(jobs)}, already done {len(frame) - len(jobs) - missing}, missing {missing}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--zip-dir', help='read Validated/<Category>.zip files from here instead of --raw-dir')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument('--jpeg', action='store_true', help='save crops as JPEG (about 4x smaller than PNG)')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)

    for category in args.categories:
        source = ZipSource(args.zip_dir, category) if args.zip_dir else FolderSource(args.raw_dir, category)
        try:
            for split in ('train', 'test'):
                prepare_split(source, args.output_dir, category, split, args.workers, args.jpeg)
        finally:
            source.close()


if __name__ == '__main__':
    main()
