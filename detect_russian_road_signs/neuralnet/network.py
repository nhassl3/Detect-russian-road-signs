# -*- coding: utf-8 -*-
from ultralytics import YOLO


def create_neural_network(data: str, pretrained_weights: str, epochs: int = 10, device='cpu') -> YOLO:
    model = YOLO(pretrained_weights)
    model.train(data=data, epochs=epochs, device=device)
    return model
