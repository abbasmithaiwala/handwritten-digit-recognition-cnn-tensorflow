import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('handwritten_digit_recognition.keras')

# Set up the canvas
st.title("Handwritten Digit Recognition")
st.write("Draw a digit in the canvas below and click 'Predict'.")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=25,
    stroke_color="white",
    background_color="black",
    width=500,
    height=500,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Extract the image data from the canvas
        img1 = canvas_result.image_data

        # Convert the canvas image to grayscale
        gray1 = cv2.cvtColor(img1.astype('uint8'), cv2.COLOR_BGR2GRAY)

        # Resize the image to 28x28 pixels as required by the model
        resized1 = cv2.resize(gray1, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize the image to scale the pixel values between 0 and 1
        newing1 = resized1 / 255.0

        # Reshape the image to match the input shape expected by the model
        newing1 = np.array(newing1).reshape(-1, 28, 28, 1)

        # Make a prediction
        predictions1 = model.predict(newing1)

        # Display the predicted digit
        st.write(f"Predicted Digit: {np.argmax(predictions1)}")
    else:
        st.write("Canvas is empty. Please draw a digit and try again.")
