"""Zip the code (no data, checkpoints or venv) into colab/sidewalk-code.zip, for uploading to Google Drive.

Run from anywhere: python colab/package_code.py
"""
import glob
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, 'colab', 'sidewalk-code.zip')
PATTERNS = ['*.py', 'requirements.txt', 'README.md', 'tests/*.py']


def main():
    files = sorted({path for pattern in PATTERNS for path in glob.glob(os.path.join(ROOT, pattern))})
    with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, os.path.relpath(path, ROOT).replace(os.sep, '/'))
    print(f'Wrote {OUTPUT} ({len(files)} files, {os.path.getsize(OUTPUT) / 1e3:.0f} KB)')


if __name__ == '__main__':
    main()
