import os
import pathlib


def path_string(*args):
    """Data accessor for the project

    path_string('test_images', '34983459873.jpg')
    'c:\\AllProjects\\kagglelandmarks\\data\\test_images\\34983459873.jpg'
    """
    return os.path.join(str(pathlib.Path(__file__).parents[1]), 'data', *args)
