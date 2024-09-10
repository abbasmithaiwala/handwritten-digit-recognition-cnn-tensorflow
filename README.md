# Handwritten Digit Recognition

This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits using the MNIST dataset. The model was trained on the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits (0-9).

## Project Overview

The goal of this project is to create an application that recognizes handwritten digits. The model is built using **TensorFlow** and **Keras**, and a graphical interface is created using **pygame** and **OpenCV** to allow users to draw digits, which the model will then predict.

### Key Features:
- Trained on the MNIST dataset.
- **CNN-based architecture** for image recognition.
- Provides an interactive interface where users can draw digits using the mouse, and the model predicts the drawn digit in real time.
  
## Requirements

To run the project, you need the following dependencies:

```bash
numpy
pandas
matplotlib
tensorflow
keras
pygame
opencv-python
```
You can install these libraries using the following command:
```
pip install numpy pandas matplotlib tensorflow keras pygame opencv-python
```

## Dataset

The project uses the MNIST dataset, which contains 28x28 grayscale images of handwritten digits. The dataset is available directly from the tensorflow.keras.datasets module.

## Project Structure

- handwritten_digit_recognition.keras: Pre-trained CNN model saved using Keras.
- Interactive Drawing Canvas: Implemented using pygame and OpenCV, allowing users to draw digits for real-time prediction.
- Training and Testing: The model is trained on 60,000 training samples and tested on 10,000 samples from the MNIST dataset.

## How to Run

1. Clone this repository:

    ```bash
    git clone https://github.com/abbasmithaiwala/handwritten-digit-recognition-cnn-tensorflow
    ```

2. Navigate to the project directory:

    ```bash
    cd handwritten-digit-recognition
    ```

3. Run the model:

    ```bash
    python Digit Recognition Model Execution.ipynb
    ```

4. A drawing canvas will open. Use your mouse to draw a digit, and the model will predict the drawn digit.

## Model Architecture

The CNN architecture consists of the following layers:

1. **Convolutional Layers**: Extract features from the input images.
2. **MaxPooling Layers**: Reduce the dimensionality of the feature maps.
3. **Dense (Fully Connected) Layers**: Perform the final classification of the digits.

## Results

The model achieves an accuracy of **over 99%** on the MNIST test set.

## Future Improvements

- Improve the drawing canvas for better user experience.
- Fine-tune the model for faster predictions.
- Add the ability to save and export drawings as image files.

## Acknowledgments

- The MNIST dataset: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- TensorFlow and Keras libraries for model building.
