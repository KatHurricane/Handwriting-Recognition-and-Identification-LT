import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Set up the ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,  # Ensure that images are not flipped horizontally
    vertical_flip=False,    # Ensure that images are not flipped vertically
    fill_mode='nearest'
)

# Path to the directory containing images
input_dir = './input_synthetic_data'
save_dir = './augmented_images'
os.makedirs(save_dir, exist_ok=True)

# Process each image in the directory
for image_name in os.listdir(input_dir):
    if image_name.endswith('.png'):  # You can add other image formats if needed
        image_path = os.path.join(input_dir, image_name)
        
        # Load the image and convert it to an array
        img = load_img(image_path, target_size=(128, 128))  # Resize as necessary
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Expand dimensions to create a batch

        # Flow method to generate augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='png'):
            i += 1
            if i > 20:  # Change this to the number of augmented images you want per input image
                break

        print(f'Generated {i} augmented images for {image_name} and saved to {save_dir}')
