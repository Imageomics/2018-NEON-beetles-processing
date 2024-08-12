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

from utils import load_dataset_images, read_image_paths

def get_mask(image_path, predictor, beetle_df, annotator='IsaFluck'):
    
    # get elytra length (even) and width line (odd) coords
    picture_id = image_path.split('/')[-1]
    elytra_coords = beetle_df[(beetle_df.pictureID == picture_id) & 
                            (beetle_df.user_name == annotator)].coords_pix_scaled_up.tolist()
    
    # read in image
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # configurate SAM model to use current image
    predictor.set_image(image)

    # counter for number of beetles in the image
    num_beetles = 0

    # segment beetles
    img_mask = np.zeros((image.shape[0], image.shape[1]))
    for i in range(0, len(elytra_coords), 2):
        # convert to actual dicts
        length_coords = ast.literal_eval(elytra_coords[i])
        width_coords = ast.literal_eval(elytra_coords[i+1])

        # use all available xy points to segment the beetle with SAM
        input_point = np.array([[length_coords['x1'], length_coords['y1']],
                        [width_coords['x1'], width_coords['y1']],
                        [width_coords['x2'], width_coords['y2']],
                        [length_coords['x2'], length_coords['y2']]])
        
        input_label = np.array([1,1,1,1])

        #get mask using SAM
        mask, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        mask = mask.squeeze()
        mask = mask * 1 #convert to 0 and 1's
        img_mask += mask

        # resolve potential overlapping mask values so that we only have 1s and 0s
        overlapping_pixels = (mask != 0) & (img_mask != 0)
        img_mask[overlapping_pixels] = mask[overlapping_pixels]

        num_beetles += 1

    #convert mask to np.uint8    
    img_mask = img_mask.astype(np.uint8)
    return img_mask, num_beetles


def get_sam_model(device):
    '''Get the SAM VIT l Model'''
    model_type = "vit_l"
    sam_checkpoint = "/home/ramirez.528/BeetlePalooza/sam_vit_l_0b3195.pth"
    if not os.path.exists(sam_checkpoint):
        model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        sam_checkpoint = wget.download(model_url)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def batch_indices_np(total, batch_size):
    indices = np.arange(total)
    return np.array_split(indices, np.ceil(total / batch_size))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory containing images we want to predict masks for. ex: /User/micheller/BeetlePalooza/group_images")
    parser.add_argument("--csv", required=True, help="Path to CSV containing Elytra measurements for beetles. ex: /User/micheller/BeetlePalooza/BeetleMeasurements.csv")
    parser.add_argument("--results", required=False, default = 'segmentation_results.csv', help="Path to the csv created containing \
                        how many beetles were segmented in each image.")
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
    beetle_df = pd.read_csv(args.csv)
    
    # read images in
    dataset_folder = args.images + '/*'
    image_filepaths = read_image_paths(dataset_folder)
    print(f'Number of images in dataset: {len(image_filepaths)}')

    # establish df to store segmentation info
    dataset_segmented = pd.DataFrame(columns = ['image', 'num_beetles'])

    # load SAM model
    segmentation_model = get_sam_model(DEVICE)

    # create masks using SAM (segment-anything-model)
    i = 0 #df indexer
    print("Segmenting beetle images...")
    for fp in image_filepaths:
        print(fp)

        #create a mask where beetle pixels are 1s and everything else is 0s
        mask, num_beetles = get_mask(fp, segmentation_model, beetle_df) 
        
        #create the path to which the mask will be saved
        mask_path = fp.replace(args.images, f'{args.images}_masks')
        mask_path = mask_path.replace(f".{fp.split('.')[-1]}", "_mask.png") #replace extension and save mask as a png
        
        #create the folder in which the mask will be saved in if it doesn't exist already
        mask_filename = "/" + mask_path.split('/')[-1] #
        mask_folder = mask_path.replace(mask_filename, "")
        os.makedirs(mask_folder, exist_ok=True)
        
        #save mask with cv2 to preserve pixel categories
        print(f"Mask path:{mask_path}")
        cv2.imwrite(mask_path, mask)

        #enter relevant segmentation data for the image in our dataframe
        dataset_segmented.loc[i, 'image'] = fp.split('/')[-1]
        dataset_segmented.loc[i, 'num_beetles'] = num_beetles

        i += 1

    # Save csv containing information about segmentation masks per each image
    dataset_segmented.to_csv(args.results, index=False)
    return

if __name__ == "__main__":
    main()