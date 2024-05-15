# -*- coding: utf-8 -*-
import typing
import random
import os
import shutil

from ultralytics import YOLO

from detect_russian_road_signs.neuralnet import *


def create_neural_network(
        data: str,  # dataset
        pretrained_weights: str,  # ready weights
        epochs: int = 10,  # count of the epochs by default
        device='cpu'  # if not CUDA, you automatically use the central processing unit
) -> YOLO:
    model = YOLO(pretrained_weights)
    model.train(data=data, epochs=epochs, device=device)
    return model


def predict(
        model: typing.Optional[YOLO] = None,  # neural network
        path_source: str = TEST_DIR / 'images',  # path to source with images
        random_images: bool = False,  # use random pictures from dataset `test` dir
        count: int = 12  # count of the pictures for random choice
) -> typing.Optional[YOLO.predict]:
    if model is None:
        if not os.path.exists(RUNS_DIR):
            return
        model = YOLO(BEST_DIR)
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
