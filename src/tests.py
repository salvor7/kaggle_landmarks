import logging
logging.getLogger('tensorflow').disabled = True     # turn down verbosity of TF logs

import data
from resnet50 import Resnet50


def test_instantiation():
    # simple test to see if the object can be instantiated
    Resnet50()


def test_prediction():
    """Test we can get a prediction from the Resnet50 object using a training image"""
    batches = Resnet50.get_batches(path=data.path_string('data', 'sample', 'train_images'))
    for images, classes in batches:
        Resnet50().predict(images)
        break


def test_landmark_images():
    landmark2images, image2landmark = data.landmark_images()

    assert len(image2landmark) > 1225000
    assert len(landmark2images) > 14951
