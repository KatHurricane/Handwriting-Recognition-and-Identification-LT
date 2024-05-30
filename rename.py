import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

image_directory = 'temp-output_letters'

pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  

def preprocess_image(image_path, padding=50):
    img = Image.open(image_path)
    img = img.convert('L')  
    img = ImageOps.expand(img, border=padding, fill='white')  
    img = img.filter(ImageFilter.MedianFilter())  
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  
    return img

def get_unique_filename(directory, base_name, extension):
    """
    Generate a unique filename by appending a number if the filename already exists.
    """
    new_filename = f"{base_name}.{extension}"
    counter = 1
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base_name}_{counter}.{extension}"
        counter += 1
    return new_filename

for filename in os.listdir(image_directory):
    file_path = os.path.join(image_directory, filename)
    
    try:
        img = preprocess_image(file_path)
        
        text = pytesseract.image_to_string(img, lang='lit', config='--psm 10').strip()
        
        letter = next((char for char in text if char.isalpha()), None)
        
        if letter:
            extension = filename.split('.')[-1]  
            new_filename = get_unique_filename(image_directory, letter, extension)
            new_file_path = os.path.join(image_directory, new_filename)
            
            original_img = Image.open(file_path)
            original_img.save(new_file_path)
            
            os.remove(file_path)
            
            print(f"Renamed {filename} to {new_filename} and deleted the original file.")
        else:
            os.remove(file_path)
            print(f"Could not extract a valid letter from {filename}. File has been deleted.")
    
    except Exception as e:
        print(f"Error processing {filename}: {e}. Leaving the file unchanged.")

print("Renaming and cleanup process completed.")
