# -*- coding: utf-8 -*-
import sys
import logging

import torch

import neuralnet

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logging.info('Starting')
    logging.info(f"Python: {sys.version}")
    logging.info(f"Path: {sys.argv[0]}")
    logging.info(f"TORCH VERSION: {torch.__version__}")
    logging.info(f"Count of CUDA Devices: {torch.cuda.device_count()}")
    logging.info(f"CUDA IS AVAILABLE: {torch.cuda.is_available()}")

    # Download dataset, weights and set configuration file
    neuralnet.dataset.download_dataset()

    # Train the neural network
    # **( ATTENTION! To train the model, more than 80% of RAM memory is required )**
    model = neuralnet.network.create_neural_network(
        data=neuralnet.CONFIGURATION,  # configuration file with main information about dataset
        pretrained_weights=neuralnet.WORKING_DIR / "yolo8s.pt",  # weights from GitHub
        epochs=110,
        device=torch.device('cuda'),  # device: [0] - GPU (required ~4.1GB for 6GB GPU)
    )  # IF YOU ALREADY HAVE THE TRAINED MODEL, DON'T UNCOMMENT THIS LINES [24-29]

    result = neuralnet.draw.predict(model=model, random_images=True, count=14)
    neuralnet.draw.draw_predicted(predicted=result)
    logging.info('Finished')
