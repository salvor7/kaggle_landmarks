import csv
import os
import pathlib
from collections import defaultdict

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
