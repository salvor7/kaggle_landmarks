"""
Adds source code folder to PYTHONPATH, so imports can be made from notebooks.
"""

import os
import sys
import pathlib


# make the src module available for import
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / 'src'))

import data
from resnet50 import Resnet50
