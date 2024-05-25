import os
import numpy as np
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
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # Ensure that images are not flipped horizontally
    vertical_flip=False,    # Ensure that images are not flipped vertically
    fill_mode='nearest'
)

# Path to the directory containing images
input_dir = './input_synthetic_data'
save_dir = './augmented_images'
os.makedirs(save_dir, exist_ok=True)

# Function to generate a unique identifier
def generate_unique_id():
    return np.random.randint(10000, 99999)

# List to keep track of images that were not generated
not_generated_images = []

# Process each image in the directory
for image_name in os.listdir(input_dir):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for PNG and JPG images
        image_path = os.path.join(input_dir, image_name)
        
        try:
            # Load the image
            img = Image.open(image_path)
            img = add_padding(img, target_size=(128, 128))
            
            # Convert the image to an array
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)  # Expand dimensions to create a batch

            # Extract base name of the image (without extension)
            base_name = os.path.splitext(image_name)[0]

            # Flow method to generate augmented images
            i = 0
            for batch in datagen.flow(x, batch_size=1):
                i += 1
                if i > 49:  # Change this to the number of augmented images you want per input image
                    break

                # Generate a unique filename for each augmented image
                unique_id = generate_unique_id()
                new_image_name = f"{base_name}_{unique_id}.png"
                new_image_path = os.path.join(save_dir, new_image_name)

                # Save the augmented image
                augmented_image = batch[0] * 255
                augmented_image = augmented_image.astype(np.uint8)
                img_to_save = Image.fromarray(augmented_image)
                img_to_save.save(new_image_path)

            print(f'Generated {i} augmented images for {image_name} and saved to {save_dir}')

        except Exception as e:
            print(f'Error processing {image_name}: {e}')
            not_generated_images.append(image_name)

# Print the list of images that were not generated
if not_generated_images:
    print("The following images were not generated due to errors:")
    for image_name in not_generated_images:
        print(image_name)
else:
    print("All images were processed successfully.")
