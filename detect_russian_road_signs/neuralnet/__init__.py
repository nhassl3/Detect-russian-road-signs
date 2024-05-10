# -*- coding: utf-8 -*-
from pathlib import Path

__all__ = [
    "ROOT_DIR", "PROJECT_NAME", "VERSION", "DATASET_DIR",
    "WORKING_DIR", "TRAIN_DIR", "VAL_DIR", "TEST_DIR",
    "CONFIGURATION", "RESULT_DIR", "RUNS_DIR", "BEST_DIR",
    "LAST_DIR",
]

ROOT_DIR = Path(__file__).parent.parent.parent
PROJECT_NAME = 'roud-signs-rus'
VERSION = 5

DATASET_DIR = ROOT_DIR / f'{PROJECT_NAME.title()}-{VERSION}'
WORKING_DIR = ROOT_DIR / 'working'
RUNS_DIR = ROOT_DIR / 'runs'
CONFIGURATION = WORKING_DIR / f"{PROJECT_NAME.replace('-', '_')}.yaml"

TRAIN_DIR = WORKING_DIR / 'train'
VAL_DIR = WORKING_DIR / 'val'
TEST_DIR = WORKING_DIR / 'test'

BEST_DIR = RUNS_DIR / 'detect/train/weights/best.pt'
LAST_DIR = RUNS_DIR / 'detect/train/weights/last.pt'

RESULT_DIR = WORKING_DIR / 'test_random_images'

from . import dataset, draw, network
