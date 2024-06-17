# Hand Gesture Recognition Model

This repository contains a hand gesture recognition model built using Convolutional Neural Networks (CNN) with TensorFlow and Keras. The model is trained to classify different hand gestures from images, enabling intuitive human-computer interaction and gesture-based control systems.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/2200031989/PRODIGY_ML_04
    cd hand-gesture-recognition
    ```

2. Install the required dependencies:

    ```bash
    pip install tensorflow keras numpy opencv-python matplotlib
    ```

## Dataset

The dataset should be organized into subdirectories, with each subdirectory representing a different hand gesture class. Each subdirectory should contain images of the corresponding gesture.


## Model Architecture

The CNN model consists of the following layers:
- Convolutional Layers
- MaxPooling Layers
- Flatten Layer
- Dense Layers
- Dropout Layer

## Training

To train the model, run the `train.py` script:

```bash
python train.py

