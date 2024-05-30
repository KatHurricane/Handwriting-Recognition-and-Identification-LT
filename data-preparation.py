import cv2
import pytesseract
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
tessdata_dir_config = '--tessdata-dir "tessdata" -l lit'

output_dir = 'temp-output_letters'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_dir = 'temp-input_images'

for image_filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_filename)
    
    if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        letter_image_region = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 50:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            
            if w < 30 or h < 30:
                continue

            letter = gray[y:y+h, x:x+w]

            letter_filename = os.path.join(output_dir, f'{os.path.splitext(image_filename)[0]}_letter_{i}.png')
            cv2.imwrite(letter_filename, letter)

            letter_image_region.append((x, y, w, h))

        for (x, y, w, h) in letter_image_region:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Detected Letters', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
