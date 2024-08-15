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
from utils import get_yolo_model, read_image_paths


def crop_image_with_opencv(image, bbox, padding=0):
    """
    Crop an image using OpenCV given a bounding box.
    
    Parameters:
        image (numpy.ndarray): The input image.
        bbox (list): The bounding box (x_min, y_min, x_max, y_max).
    
    Returns:
        cropped_image (numpy.ndarray): The cropped image.
    """
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cropped_image = image[y_min-padding:y_max+padding, x_min-padding:x_max+padding]
    return cropped_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory containing images we want to predict masks for. ex: /home/ramirez.528/2018-NEON-beetles/group_images")
    parser.add_argument("--output", required=True, help="Path to folder where you want to save the folder of individual cropped beetles. ex: /home/ramirez.528/2018-NEON-beetles")
    return parser.parse_args()


def main():
    args = parse_args()

    # read in a list of filenames in our input dataset
    group_image_filepaths = read_image_paths(args.images + "/*")

    # create output folder
    individual_folder = args.output + '/individual_images'
    os.makedirs(individual_folder, exist_ok=True)

    classes = {0: 'beetle', 1: 'scale_bar'}
    
    # Leverage GPU if available
    use_cuda = torch.cuda.is_available()
    DEVICE   = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: ", DEVICE)

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    # load yolo model trained to detect beetles and scale bars
    model = get_yolo_model()
    errors = []

    # begin cropping beetles and scale bars
    for img_path in group_image_filepaths:
        # read in image
        picture_id = img_path.split('/')[-1].split(".jpg")[0]
        image = cv2.imread(img_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # use yolo to get bbox for our beetles
        print(f"Predicting bboxes for {img_path}")
        results = model.predict(img_path, 
                                verbose=False)
        r = results[0]
        bboxes_labels = r.boxes.cls
        bboxes_xyxy = r.boxes.xyxy

        print("PRED LABELS:", bboxes_labels)
        # use each bounding box to crop out beetles
        for idx, bbox_label in enumerate(bboxes_labels):
            bounding_box = np.array(bboxes_xyxy[idx].tolist()) #[x1,y1,x2,y2]
            bbox_label = bbox_label.item()

            # crop down the image to a bounding box
            cropped_img = crop_image_with_opencv(image, bounding_box)
            print(picture_id, cropped_img.shape)

            # save the image of the cropped individual
            try:
                subfolder = individual_folder + f"/{picture_id}"
                os.makedirs(subfolder, exist_ok=True)
                cv2.imwrite(subfolder + f"/{classes[bbox_label]}_{idx}.jpg", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            except FileNotFoundError:
                errors.append(picture_id)
    
    print('The following images could encountered errors during cropping:', errors)
    return errors

if __name__ == "__main__":
    main()
