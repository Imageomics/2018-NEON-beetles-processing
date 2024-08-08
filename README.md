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
`python3 remove_background.py --images <path to images> --masks <path to segmentation masks>`


