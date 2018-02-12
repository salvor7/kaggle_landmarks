#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
from urllib.request import urlopen
from PIL import Image
from io import BytesIO


def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def DownloadImage(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    out_dir = os.path.join(out_dir, key[:2])

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    filename = os.path.join(out_dir, f'{key}.jpg')

    if os.path.exists(filename):
        print(f'Skipping file {key}')
        return

    try:
        response = urlopen(url)
        image_data = response.read()
    except Exception as err:
        print(f'Warning: Could not download image {key} from {url} because {err}')
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except Exception as err:
        print(f'Warning: Failed to parse image {key} because {err}')
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except Exception as err:
        print(f'Warning: Failed to convert image {key} to RGB because {err}')
        return

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except Exception as err:
        print(f'Warning: Failed to save image {filename} because {err}')
        return


def Run():
    if len(sys.argv) != 3:
        print(f'Syntax: {sys.argv[0]} <data_file.csv> <output_dir/>')
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = ParseData(data_file)
    pool = multiprocessing.Pool(processes=16)
    pool.map(DownloadImage, key_url_list)


if __name__ == '__main__':
    Run()