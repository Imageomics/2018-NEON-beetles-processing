# BeetlePalooza Dataset Code

This repository hosts the code and notebooks used to explore and process the [BeetlePalooza](https://github.com/Imageomics/BeetlePalooza-2024/wiki) dataset: [2018 NEON Ethanol-preserved Ground Beetles](https://huggingface.co/datasets/imageomics/2018-NEON-beetles).


## Getting Started

In a fresh `python` environment, run:
`pip install -r requirements.txt`.

CSVs explored in the notebook are pulled directly from Huggingface through their URL (these are pointing to the particular commit for the version). Adjusted CSVs are saved to a `data/` folder which is ignored by `git` since they are too large (versioning requires `git lfs`, so they are stored on Hugging Face).
