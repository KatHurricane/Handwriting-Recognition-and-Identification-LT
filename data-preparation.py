import cv2
import pytesseract
import numpy as np
import os

# Set Tesseract path and language configuration
pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract_executable'  # Update this path
tessdata_dir_config = '--tessdata-dir "tessdata" -l lit'

# Ensure the directory to save letters exists
output_dir = 'output_letters'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Specify the input directory
input_dir = 'input_images'

# Function to pad and resize the letter image to 25x25
def pad_and_resize(letter_img, size=25):
    h, w = letter_img.shape
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_letter = cv2.resize(letter_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a new image of the desired size with white background
    padded_letter = np.ones((size, size), dtype=np.uint8) * 255
    pad_h, pad_w = (size - new_h) // 2, (size - new_w) // 2
    
    padded_letter[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_letter
    return padded_letter

# Iterate over each image in the input directory
for image_filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_filename)
    
    # Check if the file is an image
    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Load the image
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours and save each letter
        letter_image_region = []
        for i, contour in enumerate(contours):
            # Filter out small contours to avoid noise
            if cv2.contourArea(contour) < 50:
                continue

            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Extract the letter using the bounding box coordinates
            letter = gray[y:y+h, x:x+w]

            # Pad and resize the letter to 25x25 pixels
            padded_resized_letter = pad_and_resize(letter, size=25)

            # Save the padded and resized letter as an image
            letter_filename = os.path.join(output_dir, f'{os.path.splitext(image_filename)[0]}_letter_{i}.png')
            cv2.imwrite(letter_filename, padded_resized_letter)

            letter_image_region.append((x, y, w, h))

        # Draw bounding boxes around detected letters
        for (x, y, w, h) in letter_image_region:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the image with bounding boxes (for debugging)
        cv2.imshow('Detected Letters', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()