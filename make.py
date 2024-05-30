import os
import numpy as np
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image

def add_padding(image, target_size=(128, 128)):
    width, height = image.size
    target_width, target_height = target_size

    delta_width = target_width - width
    delta_height = target_height - height
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))

    new_image = Image.new('RGB', target_size, (255, 255, 255))
    new_image.paste(image, padding[:2])
    
    return new_image

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  
    vertical_flip=False,    
    fill_mode='nearest'
)

input_dir = './input_synthetic_data'
save_dir = './augmented_images'
os.makedirs(save_dir, exist_ok=True)

def generate_unique_id():
    return np.random.randint(10000, 99999)

not_generated_images = []

for image_name in os.listdir(input_dir):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  
        image_path = os.path.join(input_dir, image_name)
        
        try:
            img = Image.open(image_path)
            img = add_padding(img, target_size=(128, 128))
            
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)  
            base_name = os.path.splitext(image_name)[0]

            i = 0
            for batch in datagen.flow(x, batch_size=1):
                i += 1
                if i > 49:  
                    break

                unique_id = generate_unique_id()
                new_image_name = f"{base_name}_{unique_id}.png"
                new_image_path = os.path.join(save_dir, new_image_name)

                augmented_image = batch[0] * 255
                augmented_image = augmented_image.astype(np.uint8)
                img_to_save = Image.fromarray(augmented_image)
                img_to_save.save(new_image_path)

            print(f'Generated {i} augmented images for {image_name} and saved to {save_dir}')

        except Exception as e:
            print(f'Error processing {image_name}: {e}')
            not_generated_images.append(image_name)

if not_generated_images:
    print("The following images were not generated due to errors:")
    for image_name in not_generated_images:
        print(image_name)
else:
    print("All images were processed successfully.")
