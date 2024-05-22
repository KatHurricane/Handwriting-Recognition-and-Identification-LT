import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Define the path to the directory containing images
image_directory = 'input_synthetic_data'

# Set the path to the tesseract executable (update this if necessary)
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # Adjust this path if needed

def preprocess_image(image_path, padding=50):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.expand(img, border=padding, fill='white')  # Add padding
    img = img.filter(ImageFilter.MedianFilter())  # Apply a median filter
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Increase contrast
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

# Loop through each file in the directory
for filename in os.listdir(image_directory):
    # Construct the full file path
    file_path = os.path.join(image_directory, filename)
    
    # Preprocess the image file
    try:
        img = preprocess_image(file_path)
        
        # Use pytesseract to do OCR on the image, specifying the Lithuanian language
        text = pytesseract.image_to_string(img, lang='lit', config='--psm 10').strip()
        
        # Check if the text contains only alphabetic characters
        if text.isalpha():
            letter = text[0]
            extension = filename.split('.')[-1]  # Keep the original file extension
            new_filename = get_unique_filename(image_directory, letter, extension)
            new_file_path = os.path.join(image_directory, new_filename)
            
            # Open the original image again to save without padding
            original_img = Image.open(file_path)
            original_img.save(new_file_path)
            
            # Remove the original file after renaming
            os.remove(file_path)
            
            print(f"Renamed {filename} to {new_filename} and deleted the original file.")
        else:
            # Delete the image if it contains symbols or numbers
            os.remove(file_path)
            print(f"Deleted {filename} because it contains symbols or numbers.")
    
    except Exception as e:
        print(f"Error processing {filename}: {e}. Leaving the file unchanged.")

print("Renaming process completed.")
