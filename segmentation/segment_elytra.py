import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import torch

from utils import load_dataset_images, read_image_paths, get_sam_model
from segment_anything import SamAutomaticMaskGenerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory containing individual beetle images, ex: /User/micheller/images")
    parser.add_argument("--result", required=True, help="Directory containing result of the segmentation. ex: /User/micheller/result")
    return parser.parse_args()


def crop_to_bb(img):
    labels = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 10
    where = np.nonzero(labels)
    y1, x1 = np.amin(where, axis=1)
    y2, x2 = np.amax(where, axis=1)
    img_cropped = img[y1:y2,x1:x2, :]
    return img_cropped


def is_mask_closer_to_white(image, mask):
    """Check if the masked region of the image is closer to white than to black."""
    masked_pixels = image[mask > 0]
    average_color = np.mean(masked_pixels, axis=0)
    
    white = np.array([255, 255, 255])
    black = np.array([0, 0, 0])
    
    distance_to_white = np.linalg.norm(average_color - white)
    distance_to_black = np.linalg.norm(average_color - black)
    
    return distance_to_white < distance_to_black

def save_anns(anns, image, path):
    max_area = 0
    best_res = None    

    for i, ann in enumerate(anns):
        mask = ann['segmentation']
        res = np.asarray(image) * np.repeat(mask[:, :, None], 3, axis=2)

        # Check if the masked region is closer to white than to black, we filter the background regions
        if is_mask_closer_to_white(image, mask):
            continue

        # Crop to bounding box
        res_cropped = crop_to_bb(res)

        # Calculate area of segmentation
        area = np.sum(mask)

        # Check if this is the largest segmentation found so far
        if area > max_area:
            max_area = area
            best_res = res_cropped

    # Save the largest non-background segmentation
    if best_res is not None:
        best_res_bgr = cv2.cvtColor(best_res, cv2.COLOR_RGB2BGR)

        # Now save the image with the correct color channels
        cv2.imwrite(path, best_res_bgr)
        plt.figure(figsize=(20,20))
        plt.imshow(best_res)
        plt.axis('off')
        plt.show() 
    else:
        print("No valid segmentation found to save.")



def main():
    args = parse_args()

    
    # Leverage GPU if available
    use_cuda = torch.cuda.is_available()
    DEVICE   = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: ", DEVICE)

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    
    # read images in
    dataset_folder = args.images + '/*'
    output_folder = args.result
    image_filepaths = read_image_paths(dataset_folder)
    print(f'Number of images in dataset: {len(image_filepaths)}')


    # load SAM model
    segmentation_model = get_sam_model(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(model=segmentation_model, box_nms_thresh=0.5)

    # create masks using SAM (segment-anything-model)
    print("Segmenting beetle images...")
    for fp in image_filepaths:
        
        image = cv2.imread(fp)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        
        #create the path to which the mask will be saved
        output_path = os.path.join(output_folder, fp.split('/')[-1]) 
        
        os.makedirs(output_folder, exist_ok=True)
        
        #save mask with cv2 to preserve pixel categories
        print(f"Mask path:{output_path}")
        save_anns(masks, image, output_path)

    return


if __name__ == "__main__":
    main()
