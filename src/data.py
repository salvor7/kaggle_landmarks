import csv
import os
import pathlib
from collections import defaultdict
from PIL import Image

import functools


def path_string(*args):
    """Data accessor for the project

    path_string('test_images', '34983459873.jpg')
    'c:\\AllProjects\\kagglelandmarks\\data\\test_images\\34983459873.jpg'
    """
    return os.path.join(str(pathlib.Path(__file__).parents[1]), 'data', *args)


@functools.lru_cache(maxsize=1)
def landmark_images():
    """Returns dictionaries linking training image path strings to landmarks

    :param lm: str
    :return: [str, ...]
    """
    landmark2image = defaultdict(list)
    image2landmark = {}

    with open(path_string('recognition', 'train.csv')) as training_set:
        reader = csv.reader(training_set)
        for image, url, landmark in reader:
            image2landmark[image] = landmark
            landmark2image[landmark].append(image)

    return landmark2image, image2landmark


def resize_image(filename, src_folder, dest_folder, target_size):
    """
    Opens the image file, resizes it to the target_size via the BICUBIC filter, and saves it in the dest_folder.
        The file is saved in JPEG format.

    :param  filename: string containing the full name, including file extension, of the image
            src_folder: string containing the path of the image - this must end in the folder character '/' or '\'
            dest_folder: string containing the path of the saving destination - this must end in the folder character '/' or '\',
                        and the folder must exist
            target_size: int containing the size of the shorter side of the image

    :return nothing. File is saved and resized if completes properly
    """

    # open the image
    try:
        image_orig = Image.open(src_folder + filename)
    except Exception as err:
        print(f'Warning: Failed to open image {filename} because {err}')
        return

    # determine the new size of the image
    img_size = image_orig.size
    if img_size[0] < img_size[1]:
        ratio = target_size / img_size[0]
    else:
        ratio = target_size / img_size[1]
    img_resize = (int(img_size[0] * ratio), int(img_size[1] * ratio))

    # resize the image
    try:
        image_resized = image_orig.resize(img_resize, Image.BICUBIC)
    except Exception as err:
        print(f'Warning: Failed to resize image {filename} because {err}')
        return

    # save the file to the destination folder
    try:
        image_resized.save(dest_folder + filename, format='JPEG', quality=90)
    except Exception as err:
        print(f'Warning: Failed to save image {filename} because {err}')
        return

    return
