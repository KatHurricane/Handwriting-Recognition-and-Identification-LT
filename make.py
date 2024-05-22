import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image

# Function to add padding to the image
def add_padding(image, target_size=(128, 128)):
    width, height = image.size
    target_width, target_height = target_size

    # Calculate padding
    delta_width = target_width - width
    delta_height = target_height - height
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

    # Add padding with white background
    new_image = Image.new('RGB', target_size, (255, 255, 255))
    new_image.paste(image, padding[:2])
    
    return new_image

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
        
        # Load the image
        img = Image.open(image_path)
        img = add_padding(img, target_size=(128, 128))
        
        # Convert the image to an array
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Expand dimensions to create a batch

        # Flow method to generate augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='png'):
            i += 1
            if i > 20:  # Change this to the number of augmented images you want per input image
                break

        print(f'Generated {i} augmented images for {image_name} and saved to {save_dir}')
