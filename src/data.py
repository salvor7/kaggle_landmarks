import csv
import os
import pathlib
import random
from collections import defaultdict
from PIL import Image

import functools
from shutil import copyfile


def path_string(*args):
    """Data accessor for the project

    path_string('test_images', '34983459873.jpg')
    '....\\kagglelandmarks\\data\\test_images\\34983459873.jpg'
    """
    return os.path.join(str(pathlib.Path(__file__).parents[1]), *args)


@functools.lru_cache()
def image_path(image_hex):
    """Return the path of a competition image file given it's hex name

    :param image_hex: str
    :param data_set: str
    :param extension: str
    :return: str
    """
    for dataset in ['train', 'test', 'index']:
        image_location = path_string('data', dataset+'_images', image_hex[:2], image_hex+'.jpg')
        if os.path.isfile(image_location):
            return image_location

    raise FileNotFoundError(f'Key {image_hex} does not refer to a landmark image.')


@functools.lru_cache(maxsize=1)
def training_images():
    """Returns dictionaries linking training image path strings to landmarks

    :param maxsize: int
    :return: ({str: [str, ...], ...}, {str: str})
    """
    landmark2image = defaultdict(list)
    image2landmark = {}

    with open(path_string('data', 'recognition', 'train.csv')) as training_set:
        reader = csv.reader(training_set)
        for image_hex, url, landmark in reader:
            image2landmark[image_hex] = landmark
            landmark2image[landmark].append(image_hex)

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
        return False

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
        return False

    # save the file to the destination folder
    try:
        image_resized.save(dest_folder + filename, format='JPEG', quality=90)
    except Exception as err:
        print(f'Warning: Failed to save image {filename} because {err}')
        return False

    return True


def make_subsample(train=1000, valid=1000, classes=10):
    """Make a subsample of landmark training images

    Creates both training set, and a validation set from training data, and organizes them
    in to landmark folders.

    :param train: int
    :param valid: int
    :return: None
    """
    while True:
        random_classes = random.sample(training_images()[0].keys(), k=classes)
        available_images = sum([training_images()[0][class_] for class_ in random_classes], [])
        try:
            subsample = random.sample(available_images, k=train+valid)
        except ValueError:
            pass
        else:
            break

    for idx, image_name in enumerate(subsample):
        if idx < train:
            subfolder = 'train_images'
        else:
            subfolder = 'valid_images'

        landmark_name = training_images()[1][image_name]
        image_new_folder = path_string('data', 'sample', subfolder, landmark_name)
        os.makedirs(image_new_folder, exist_ok=True)

        image_old_path = path_string('data', 'train_images', image_name[:2], image_name+'.jpg')
        image_new_path = os.path.join(image_new_folder, image_name+'.jpg')
        try:    # copy if file was downloaded
            copyfile(image_old_path, image_new_path)
        except FileNotFoundError:
            try:    # remove folder if no files are in it already
                os.rmdir(image_new_folder)
            except OSError:
                pass


if __name__ == '__main__':
    make_subsample()
