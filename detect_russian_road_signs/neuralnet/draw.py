# -*- coding: utf-8 -*-
import random
import typing
import os
import shutil

import matplotlib.pyplot as plt
from ultralytics import YOLO

from detect_russian_road_signs.neuralnet import *

__all__ = ['predict', 'draw_predicted',]


def predict(
        model: typing.Optional[YOLO] = None,
        path_source: str = TEST_DIR / 'images',
        random_images: bool = False,
        count: int = 12
) -> typing.Optional[YOLO.predict]:
    if model is None:
        if not os.path.exists(RUNS_DIR):
            return
        model = YOLO(LAST_DIR)
    if random_images:
        if not os.path.exists(RESULT_DIR):
            os.mkdir(RESULT_DIR)
        images = os.listdir(TEST_DIR / 'images')
        for _ in range(count):
            random_image = random.choice(images)
            shutil.copyfile(TEST_DIR / 'images' / random_image, RESULT_DIR / random_image)
        result = model.predict(source=RESULT_DIR)
        shutil.rmtree(RESULT_DIR)
        return result
    return model.predict(source=path_source)


def draw_predicted(predicted, rows: int = 3, cols: int = 5):
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(rows*(cols-1), rows*(cols-1)))
    for i, res in enumerate(predicted):
        ax.ravel()[i].imshow(res.plot())
    plt.tight_layout()
    plt.axis('off')
    plt.show()
