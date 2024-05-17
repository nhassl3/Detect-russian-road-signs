# Detect-Russian-Road-Signs
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Fantaz1er/detect-russian-road-signs)![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django)![GitHub License](https://img.shields.io/github/license/Fantaz1er/detect-russian-road-signs)

![Logo](https://cdn-icons-png.flaticon.com/512/6707/6707424.png)

## Version  

**0.1.0**
##  Description 

Coursework on Python programming language version 3.x was completed by a first-year student of the Information Systems and Technologies Faculty at Nizhny Novgorod State University of Architecture and Civil Engineering (**IS-35 group**).

As part of the project, a convolutional neural network was developed based on pre-trained weights and biases from the well-known **YOLOv8 model**. The neural network's purpose was to recognize traffic signs in Russia. The network was trained on a dataset previously created in-house using the Roboflow service, which provides a REST API for workspace management. To obtain a training sample, the student registered with the service and obtained an API key with a unique code. The API key was assigned to the student environment variable to prevent unauthorized access by other users.

## Scheme of using the program

**Poetry** is used in this project. Before starting work, make sure that you have [poetry](https://python-poetry.org/docs)
After cloning the project, run the following commands:
>poetry shell

>poetry update or poetry lock

With these commands, you will download all the necessary libraries with the necessary versions and their dependencies into your virtual environment.

Also, don't forget to get the **API key** after registering on [Roboflow](https://universe.robotflow.com/ksenia-komlach/roud-signs-rus/dataset/5/download/yolov8 )

Go to the directory with the project `detect_russian_road_signs/` and run `main.py `

## Dependencies

##### python = ">=3.10,<=3.12"
#### matplotlib = "^3.8.4"
#### pyyaml = "^6.0.1"
#### roboflow = "^1.1.28"
#### ultralytics = "^8.2.11"
#### ipywidgets = "^8.1.2"

## License

This project uses the [MIT](https://github.com/Fantaz1er/detect-russian-road-signs/blob/main/LICENSE )

## Contact

To contact the author of the project, write to the following email: **matvey.kvasov.05@mail.ru**
