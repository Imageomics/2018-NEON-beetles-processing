import os
import glob
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch, torchvision
import argparse
import wget
import ast
from utils import get_sam_model

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


def crop_image_with_opencv(image, bbox, padding=0):
    """
    Crop an image using OpenCV given a bounding box.
    
    Parameters:
        image (numpy.ndarray): The input image.
        bbox (tuple): The bounding box (x_min, y_min, x_max, y_max).
    
    Returns:
        cropped_image (numpy.ndarray): The cropped image.
    """
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[y_min-padding:y_max+padding, x_min-padding:x_max+padding]
    return cropped_image



def batch_indices_np(total, batch_size):
    indices = np.arange(total)
    return np.array_split(indices, np.ceil(total / batch_size))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory containing images we want to predict masks for. ex: /User/micheller/BeetlePalooza/group_images")
    parser.add_argument("--csv", required=True, help="Path to CSV containing Elytra measurements for beetles. ex: /User/micheller/BeetlePalooza/individual_metadata_full.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    classes = {0: 'background', 1: 'beetle'}
    
    # Leverage GPU if available
    use_cuda = torch.cuda.is_available()
    DEVICE   = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: ", DEVICE)

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    # read beetle msmt csv in
    individual_measurements = pd.read_csv(args.csv)
    sam = get_sam_model(DEVICE)
    predictor = SamPredictor(sam)
    errors = []
    for i, row in individual_measurements.iterrows():
        picture_id = row['pictureID']

        # Calculate the midpoint of the bounding box
        resized_image_shape = ast.literal_eval(row['resized_image_dim'])
        image_shape = ast.literal_eval(row['image_dim'])

        # get the scaled elytra coords
        length_coords  = get_scaled_coords(ast.literal_eval(row['coords_pix_length']), resized_image_shape, image_shape)
        width_coords  = get_scaled_coords(ast.literal_eval(row['coords_pix_width']), resized_image_shape, image_shape)

        input_point = np.array([[length_coords['x1'], length_coords['y1']],
                                [width_coords['x1'], width_coords['y1']],
                                [width_coords['x2'], width_coords['y2']],
                                [length_coords['x2'], length_coords['y2']]])
        input_label = np.array([1,1,1,1])

        # # configurate SAM model to use current image
        img_path= args.images + '/' + picture_id
        image = cv2.imread(img_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        # get the mask and bounding box
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False)
        
        print(masks.shape)
        
        bounding_box = get_bounding_boxes(masks)[0]
        print(bounding_box)

        # crop down the image to a bounding box
        cropped_img = crop_image_with_opencv(image, bounding_box)
        print(picture_id, cropped_img.shape)

        # save the image
        individual_folder = 'individual_images'
        os.makedirs(individual_folder, exist_ok=True)

        #save the resized cropped wings to their path
        try:
            cv2.imwrite(row['file_name'], cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        except FileNotFoundError:
            errors.append(row['file_name'])
    
    print('The following images could encountered errors during cropping:', errors)
    return errors

if __name__ == "__main__":
    main()
