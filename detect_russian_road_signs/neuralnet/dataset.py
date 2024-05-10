# -*- coding: utf-8 -*-
import requests
import yaml
import os
import shutil
from roboflow import Roboflow

from detect_russian_road_signs import API_KEY
from detect_russian_road_signs.neuralnet import *

__all__ = ['download_dataset']

CLASSES = ("Дети, Жилая зона, Въезд запрещён, Уступи дорогу, Автобус, Главная дорога, Заправка,"
           " Дорожные работы, Движение запрещено, Грузовики, Остановка запрещена, Пешеходный переход,"
           " Пересечение с велодорожкой, Ограничение скорости 20, Ограничение скорости 40,"
           " Велосипедная дорожка, Искусственная неровность, Неровная дорога").split(', ')


def download_dataset() -> None:
    # Download dataset
    # If dataset already exists ignore this step
    if not os.path.exists(WORKING_DIR):
        print(True)
        rf = Roboflow(api_key=API_KEY)  # API KEY in __init__ detect_russian_road_signs module
        project = rf.workspace("ksenia-komlach").project(PROJECT_NAME)
        project.version(VERSION).download("yolov8")
        try:
            shutil.copytree(DATASET_DIR / 'train', TRAIN_DIR)
            shutil.copytree(DATASET_DIR / 'valid', VAL_DIR)
            shutil.copytree(DATASET_DIR / 'test', TEST_DIR)
        except FileExistsError:
            pass
        finally:
            shutil.rmtree(DATASET_DIR, ignore_errors=True)

    # Download pretrained weights
    if not os.path.exists(WORKING_DIR / 'yolo8s.pt'):
        print(True)
        with requests.get('https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt') as response:
            with open(WORKING_DIR / "yolo8s.pt", "wb") as file:
                file.write(response.content)

    if not os.path.exists(CONFIGURATION):
        print(True)
        data = {
            "names": CLASSES,
            "nc": len(CLASSES),
            "roboflow": {
                "license": "Public",
                "project": PROJECT_NAME,
                "url": "https://universe.roboflow.com/ksenia-komlach/roud-signs-rus/dataset/5",
                "version": VERSION,
                "workspace": "ksenia-komlach"
            },
            "test": str(TEST_DIR),
            "train": str(TRAIN_DIR),
            "val": str(VAL_DIR),
        }
        with open(CONFIGURATION, "w+", encoding='utf-8') as file:
            yaml.dump(data, file, allow_unicode=True)
