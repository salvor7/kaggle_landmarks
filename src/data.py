import csv
import os
import pathlib
import random
from collections import defaultdict

import functools
from shutil import copyfile


def path_string(*args):
    """Data accessor for the project

    path_string('test_images', '34983459873.jpg')
    '....\\kagglelandmarks\\data\\test_images\\34983459873.jpg'
    """
    return os.path.join(str(pathlib.Path(__file__).parents[1]), *args)


@functools.lru_cache(maxsize=1)
def landmark_images():
    """Returns dictionaries linking training image path strings to landmarks

    :param lm: str
    :return: [str, ...]
    """
    landmark2image = defaultdict(list)
    image2landmark = {}

    with open(path_string('data', 'recognition', 'train.csv')) as training_set:
        reader = csv.reader(training_set)
        for image, url, landmark in reader:
            image2landmark[image] = landmark
            landmark2image[landmark].append(image)

    return landmark2image, image2landmark


def make_subsample(train=1000, valid=1000, classes=10):
    """Make a subsample of landmark training images

    Creates both training set, and a validation set from training data, and organizes them
    in to landmark folders.

    :param train: int
    :param valid: int
    :return: None
    """
    while True:
        random_classes = random.sample(landmark_images()[0].keys(), k=classes)
        available_images = sum([landmark_images()[0][class_] for class_ in random_classes], [])
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

        landmark_name = landmark_images()[1][image_name]
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
