# -*- coding: utf-8 -*-
import random
import time
import typing
import os
import shutil

import cv2
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


def draw_predicted(predicted, rows: int = 3, cols: int = 3) -> None:
    if not predicted:
        print("Can't draw predicted road signs because predicted road signs is empty")
        return
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))
    for i, res in enumerate(predicted):
        ax.ravel()[i].imshow(res.plot())
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.title("Predicted road signs")
    plt.axis('off')
    plt.show()


def show_video(
        video: str,
        model: typing.Optional[YOLO] = None,
        save_video=False,
        show=True,
        image_size: int = 608
) -> None:
    if model is None:
        if not os.path.exists(RUNS_DIR):
            raise FileExistsError("WEIGHTS IS NOT A VALID MODEL")
        model = YOLO(BEST_DIR)
        model.fuse()
    cap = cv2.VideoCapture(video)
    if save_video:
        ...
        # TODO: write video writer

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(
            frame,
            iou=0.4,
            conf=0.5,
            persist=True,
            verbose=False,
            imgsz=image_size,
        )
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, clss in zip(boxes, classes):
                random.seed(int(clss)+8)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(
                    frame,
                    model.model.names[clss],  # type: ignore
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                    )

        if show:
            frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
