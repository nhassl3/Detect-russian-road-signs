# -*- coding: utf-8 -*-
import random
import time
import typing
import os

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from detect_russian_road_signs.neuralnet import *

__all__ = ['draw_predicted', 'show_video']


def draw_predicted(predicted, rows: int = 3, cols: int = 3) -> None:
    """
    Draws predicted road signs using predicted road signs dataset.
    :param predicted: predicted road signs dataset.
    :param rows: rows of predicted road signs dataset.
    :param cols: cols of predicted road signs dataset.
    :return: No return value.
    """
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
) -> None:
    """
    Show video with predicted road signs.
    :param video: filepath to video file.
    :param model: ready model to predict a frame of the image.
    :return: No return value.
    """
    # If model is None, create variable model with "BEST" weights
    # If weights not found, raise exception "FileExistsError"
    if model is None:
        if not os.path.exists(RUNS_DIR):
            raise FileExistsError("WEIGHTS IS NOT A VALID MODEL")
        model = YOLO(BEST_DIR)
        model.fuse()

    # Open the video and read him
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    while cap.isOpened():
        # Delay between frames
        time.sleep(0.0013)

        ret, frame = cap.read()
        if not ret:
            break

        # Tracking the frame
        results = model.track(
            frame,  # frame of the video
            iou=0.5,  # min/max suppression
            conf=0.3,  # minimal confidence
            persist=True,  # persist
            verbose=False,  # do not output unnecessary logs
            imgsz=640,  # image size,
            tracker=r'.\trackers\bytetrack.yaml'  # configuration of the tracker
        )

        if results[0].boxes.id is not None:
            # Get boxes and classes of a frame
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            # Run through boxing and class
            for box, clss in zip(boxes, classes):
                # Generation random color for the box rectangle
                random.seed(int(clss)+8)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # Draw the rectangle with text class name
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)  # draw the rectangle around sign
                cv2.putText(
                    frame,  # frame
                    model.model.names[clss],  # type: ignore  # text
                    (box[0], box[1]),  # above box
                    cv2.FONT_HERSHEY_COMPLEX,  # font (supports Cyrillic alphabet)
                    0.6,  # font scale
                    (0, 0, 0),  # color of the text
                    2,  # thickness
                    cv2.LINE_AA  # line type
                    )

        # Show frame
        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        cv2.imshow("Video", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
