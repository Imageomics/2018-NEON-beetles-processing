# BeetlePalooza Dataset Code

This repository hosts the code and notebooks used to explore and process the [BeetlePalooza](https://github.com/Imageomics/BeetlePalooza-2024/wiki) dataset: [2018 NEON Ethanol-preserved Ground Beetles](https://huggingface.co/datasets/imageomics/2018-NEON-beetles).


## Data Exploration and Analysis

### Getting Started

In a fresh `python` environment, run:
`pip install -r requirements.txt`.

CSVs explored in the notebook are pulled directly from Huggingface through their URL (these are pointing to the particular commit for the version). Adjusted CSVs are saved to a `data/` folder which is ignored by `git` since they are too large (versioning requires `git lfs`, so they are stored on Hugging Face).

### Notebooks

 - EDA-0-1 gives an initial exploration of the data. It adds and renames some columns in the metadata file for the dataset.
 - EDA-0-2 explores the variation in the measurements of individuals (provides graphs). It also checks the potential outliers and creates a measurement ID, providing a unique ID for the beetle measurement CSV.
 - EDA-0-3 fixes the outliers that were mislabeled, then generates individual-based CSVs for segmentation and connection to the individual images to be created from the segmentation process. 
 - EDA-0-4 adds "scientificName", "genus", "species", "NEON_sampleID", and "siteID" columns to the resized beetle metadata file to display alongside the resized images in the dataset viewer on HF.

### Metadata

 - all_measurements is a CSV with all the measurements done by each annotator (each row is a pair of measurements for a single beetle).
 - individual_metadata_full is a CSV with all the measurements done by Isadora Fluck (each row represents an individual beetle with its pair of elytra measurements). This was created for the segmentation process.
 - multi_annotator_count is a CSV with counts of annotations per image, the expected number (based on the number of rows and annotators associated with that image), and the maximum `individual` number provided for that image (if `max_individual` is less than 99, that is the number of individuals in that image; if it's 99 or greater, then there may be more individuals based on the individual count and numeric export from Zooniverse).

Note that `all_measurments.csv` and `individual_metadata_full.csv` are supersets of the `individual_metadata.csv` in [2018 NEON Ethanol-preserved Ground Beetles](https://huggingface.co/datasets/imageomics/2018-NEON-beetles) (they contributed to its creation from [`BeetleMeasurements.csv`](https://huggingface.co/datasets/imageomics/2018-NEON-beetles/blob/main/BeetleMeasurements.csv)), and are thus reproduced here under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license and should be [cited appropriately](https://huggingface.co/datasets/imageomics/2018-NEON-beetles#citation) if re-used.


## Segmentation

The segmentation folder contains scripts to leverage the elytra length and width coordinates and Meta's Segment-Anything model to segment beetles out. 

To configure your environment using conda run:
```
cd segmentation
conda env create --file environment.yaml
conda activate beetles
```

To predict segmentation masks for beetles imaged, run:
`python3 predict_masks.py --images <path to images> --csv <path to image metadata csv> --results <optional; name for csv of segmentation results>`

To remove the background of beetle images using their segmentation masks run:
```
python3 remove_background.py --images <path to images> --masks <path to segmentation masks>
```

To crop out individual beetles from images run:
```
python3 individual_beetles.py --images <path to group_images> --csv <path to metadata/individual_metadata_full.csv>
```

**FYI**: The script to crop out individual beetles works well for the images that have coords_pix_length and coords_pix_width information correctly align to beetles. However, there are a couple images where this is not the case, and thus the segmentation of beetles will not result in a nice crop of the individual beetles.


### Beetle detection

For a more robust method with minimal input, we can use the (YOLO detector fine-tuned on beetle images)[https://huggingface.co/imageomics/yolo_beetle_detection/blob/main/yolo_beetles_best.pt] to detect and crop individuals.

```
python3 yolo_crop_beetles.py --images <path to 2018-NEON-beetles/group_images> --output </home/ramirez.528/2018-NEON-beetles>
```

This method, unlike the individual_beetles.py script above, does not require any coordinates as input to crop down to the individual beetles. Images are saved in an overall individual_images folder, with subfolders named after the corresponding group picture ID containing the individual beetle and scale bar images. The structure is as follows:

individual_images
  |
  |
  |------- <pictureID 1>
  |             |
  |             |-------- <beetle_1.jpg>
  |             |-------- <beetle_2.jpg>
  |             |-------- <...>
  |             |-------- <scale_bar_i.jpg>
  |
  |------- <pictureID 2>
  |             |
  |             |-------- <beetle_i+1.jpg>
  |             |-------- <beetle_i+2.jpg>
  |             |-------- <...>
  |             |-------- <scale_bar_n.jpg>