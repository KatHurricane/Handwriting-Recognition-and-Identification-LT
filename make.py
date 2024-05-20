import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Set up the ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Path to the single image
image_path = './dataset/ą.png'

# Load the image and convert it to an array
img = load_img(image_path, target_size=(128, 128))  # Resize as necessary
x = img_to_array(img)
x = np.expand_dims(x, axis=0)  # Expand dimensions to create a batch

# Flow method to generate augmented images
i = 0
save_dir = './augmented_images'
os.makedirs(save_dir, exist_ok=True)
for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='png'):
    i += 1
    if i > 20:  # Change this to the number of augmented images you want
        break

print(f'Generated {i} augmented images and saved to {save_dir}')
