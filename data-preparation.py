import cv2
import pytesseract
import numpy as np
import os

# Ensure the directory to save letters exists
output_dir = 'letters'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the image
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and save each letter
letter_image_region = []
for i, contour in enumerate(contours):
    # Get the bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)
    # Extract the letter using the bounding box coordinates
    letter = gray[y:y+h, x:x+w]
    # Save the letter as an image
    letter_filename = os.path.join(output_dir, f'letter_{i}.png')
    cv2.imwrite(letter_filename, letter)
    letter_image_region.append((x, y, w, h))

# Optionally, you can draw bounding boxes around detected letters
for (x, y, w, h) in letter_image_region:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with bounding boxes (for debugging)
cv2.imshow('Detected Letters', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
