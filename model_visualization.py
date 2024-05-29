import os
import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.backend import ctc_batch_cost
import matplotlib.pyplot as plt

# Define the character set to include Lithuanian letters
alphabet = "abcdefghijklmnopqrstuvwxyząčęėįšųūž"
ctc_blank_index = len(alphabet)  # Index for the CTC blank token

# Define CTC loss function
def ctc_loss_function(y_true, y_pred):
    input_length = np.ones((y_pred.shape[0], 1)) * y_pred.shape[1]
    label_length = np.ones((y_true.shape[0], 1)) * y_true.shape[1]
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Function to load letter images
def load_letter_images(directories):
    images = []
    labels = []
    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith(".png") or file.endswith(".jpg"):
                letter = file.split('.')[0]
                image_path = os.path.join(directory, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                labels.append(letter)
    return images, labels

# Load the trained model
model_path = 'Models/model.keras'
model = load_model(model_path, custom_objects={'ctc_loss_function': ctc_loss_function},safe_mode=False, compile=False)

# Load some sample data
directories = ["input_synthetic_data", "augmented_images", "output_letters"]
images, labels = load_letter_images(directories)

# Function to decode predictions
def decode_prediction(pred, alphabet):
    pred_text = ''
    for p in pred:
        index = np.argmax(p)
        if index != ctc_blank_index:
            pred_text += alphabet[index]
    return pred_text

# Visualize predictions
def visualize_predictions(model, images, labels, alphabet, num_samples=5):
    plt.figure(figsize=(15, 5))
    
    for i in range(num_samples):
        img = cv2.resize(images[i], (128, 32))  # Resize to width 128 and height 32
        img = img.astype(np.float32) / 255.0
        img = np.stack((img,)*3, axis=-1)  # Convert grayscale to RGB by duplicating the single channel
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        pred = model.predict(img)
        pred_text = decode_prediction(pred[0], alphabet)
        true_text = labels[i]
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(cv2.resize(images[i], (128, 32)), cmap='gray')
        plt.title(f'True: {true_text}\nPred: {pred_text}')
        plt.axis('off')
    
    plt.show()

# Visualize some predictions
visualize_predictions(model, images, labels, alphabet)