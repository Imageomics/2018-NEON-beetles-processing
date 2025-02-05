# Beetle Image Processing  

The code in ```Beetle_Palooza_image_separation.ipynb``` contains a Python script for processing beetle images collected from the 2018 NEON dataset. The script loads images, applies preprocessing, detects beetles using contour detection, and extracts individual beetle images while filtering out images with excessive blue color.

## Features  
- Loads images from a specified directory  
- Converts images to grayscale and applies thresholding  
- Detects contours to identify individual beetles  
- Extracts and saves detected beetles as separate images  
- Filters out images based on size and blue pixel percentage  

## Dependencies  
Ensure you have the following Python libraries installed:  
```bash
pip install opencv-python numpy
```

## Usage  

### 1. Set Directory Paths  
Modify the script to set the correct **input** and **output** directories:  
```python
# Define input and output directories
input_directory = "D:/University/RA/Beetle_Palooza/Data/Full_Data/2018-NEON-beetles/group_images/"
output_directory = "D:/University/RA/Beetle_Palooza/Data/separate_images/"
```

### 2. Run the Script  
Execute the script to process images:  
```bash
python process_beetles.py
```

## Code Overview  

### Load Image Paths  
The script scans a directory for image files and stores their paths.  
```python
import os
import cv2
import numpy as np

# Get all image file paths
image_paths = [os.path.join(input_directory, filename) for filename in os.listdir(input_directory)]
```

### Image Processing  
- Converts images to grayscale  
- Applies thresholding  
- Detects contours for object segmentation  
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### Filtering & Saving Images  
- Extracts bounding boxes for detected objects  
- Saves images while filtering by size and blue pixel content  
```python
blue_threshold = 200  
blue_pixel_percentage_threshold = 5  
min_file_size = 50 * 1024  
max_file_size = 700 * 1024  
```

## Output  
The processed beetle images are stored in the specified **output directory**, with unwanted images automatically removed.
