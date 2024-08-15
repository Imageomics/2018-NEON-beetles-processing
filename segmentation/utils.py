import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch, torchvision
import argparse
import wget
import ast
import glob, os

# saved local SAM checkpoint (same as URL) (replace with your own paths to your ckpts)
SAM_CHECKPOINT = "/home/ramirez.528/BeetlePalooza/sam_vit_l_0b3195.pth"
YOLO_CHECKPOINT = "/home/ramirez.528/2018-NEON-beetles-processing/segmentation/yolo_beetles_best.pt"

def load_dataset_images(dataset_path, color_option=0):
    '''Load in actual images from filepaths from all subfolders in the provided dataset_path'''

    file_extensions = ["jpg", "JPG", "jpeg", "png"]

    #Get training images and mask paths then sort
    image_filepaths = []
    for directory_path in glob.glob(dataset_path):
        if os.path.isfile(directory_path):
            image_filepaths.append(directory_path)
        elif os.path.isdir(directory_path):
            for ext in file_extensions:
                for img_path in glob.glob(os.path.join(directory_path, f"*.{ext}")):
                    image_filepaths.append(img_path)


    #sort image and mask fps to ensure we have the same order to index
    image_filepaths.sort()

    #get actual masks and images
    dataset_images = []

    for img_path in image_filepaths:
        if color_option == 0:
            #read image in grayscale
            img = cv2.imread(img_path, color_option)
        elif color_option == 1:
            #read in color and reverse order to RGB since opencv reads in BGR
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dataset_images.append(img)

    #Enable dataset loading for inhomogeneous data
    try:
        # For improved performance and memory usage, try to convert the list of images to a numpy array
        dataset_images = np.array(dataset_images)
    except ValueError as e:
        # But data may be inhomogeneous depending on preprocessing
        print(f"Error converting images to numpy array: {e}")
        print("Returning list of images instead.")
        return dataset_images, image_filepaths
    
    return dataset_images, image_filepaths


def read_image_paths(dataset_path):
    '''Read in all filepaths to images'''

    file_extensions = ["jpg", "JPG", "jpeg", "png"]

    #Get training images and mask paths then sort
    image_filepaths = []
    for directory_path in glob.glob(dataset_path):
        if os.path.isfile(directory_path):
            image_filepaths.append(directory_path)
        elif os.path.isdir(directory_path):
            for ext in file_extensions:
                for img_path in glob.glob(os.path.join(directory_path, f"*.{ext}")):
                    image_filepaths.append(img_path)


    #sort image and mask fps to ensure we have the same order to index
    image_filepaths.sort()
    return image_filepaths


def load_batch_images(image_filepaths, color_option=0):
    '''Read images from batch of image_filepaths provided.
    Input:
        image_filepaths:  list of filepaths to images we want to read in
    
    Returns:
        dataset_images:   list of cv2 image objects
    '''

    #get actual masks and images
    dataset_images = []

    for img_path in image_filepaths:
        if color_option == 0:
            #read image in grayscale
            img = cv2.imread(img_path, color_option)
        elif color_option == 1:
            #read in color and reverse order to RGB since opencv reads in BGR
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dataset_images.append(img)

    #Enable dataset loading for inhomogeneous data
    try:
        # For improved performance and memory usage, try to convert the list of images to a numpy array
        dataset_images = np.array(dataset_images)
    except ValueError as e:
        # But data may be inhomogeneous depending on preprocessing
        print(f"Error converting images to numpy array: {e}")
        print("Returning list of images instead.")
        return dataset_images
    
    return dataset_images

def get_sam_model(device):
    '''Get the SAM VIT l Model'''
    model_type = "vit_l"
    sam_checkpoint = SAM_CHECKPOINT
    if not os.path.exists(sam_checkpoint):
        model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        sam_checkpoint = wget.download(model_url)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam


def get_yolo_model():
    '''Download trained yolo v8 model from huggingface and load in weights'''
    ## check if we already have the trained yolo checkpoint file
    yolo_checkpoint = YOLO_CHECKPOINT
    if not os.path.exists(yolo_checkpoint):
        # download from huggingface
        print("Downloading yolo beetle detection model from HF...")
        model_url = "https://huggingface.co/imageomics/yolo_beetle_detection/resolve/main/yolo_beetles_best.pt"
        yolo_checkpoint = wget.download(model_url)

    model = YOLO(yolo_checkpoint)
    return model

def rescale_coordinates(old_x, old_y, old_width, old_height, new_width, new_height):
    # Calculate the scaling factors
    x_scale = new_width / old_width
    y_scale = new_height / old_height

    # Calculate the new coordinates
    new_x = old_x * x_scale
    new_y = old_y * y_scale

    return new_x, new_y

def get_scaled_coords(coords, resized_image_shape, image_shape):

    old_width, old_height = resized_image_shape[0], resized_image_shape[1]  # Original image size
    new_width, new_height = image_shape[0], image_shape[1]  # New image size

    x1_new, y1_new = rescale_coordinates(coords['x1'], coords['y1'], old_width, old_height, new_width, new_height)
    x2_new, y2_new = rescale_coordinates(coords['x2'], coords['y2'], old_width, old_height, new_width, new_height)

    new_coords = {'x1': int(x1_new), 'y1': int(y1_new), 'x2': int(x2_new), 'y2': int(y2_new)}

    return new_coords

def get_bounding_boxes(masks):
    bounding_boxes = []
    for mask in masks:
        # Find the indices where the mask is 1
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():  # Check if there are any true values
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            bounding_boxes.append((x_min, y_min, x_max, y_max))
        else:
            bounding_boxes.append(None)  # No bounding box if the mask is empty
    return bounding_boxes


def batch_indices_np(total, batch_size):
    indices = np.arange(total)
    return np.array_split(indices, np.ceil(total / batch_size))
