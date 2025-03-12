# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: bp-eda
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %% [markdown]
# # Update Beetle Image Metadata files for viewer
#
# The viewer likely needs unique rows per image for the full-sized images (`group_images`, assuming it's not truly limited by their size) and we want to add more info to the file for the resized images so they can be viewed by `genus`, `species`, `NEON_sampleID`, and `siteID`

# %%
# get metadata from HF
df = pd.read_csv("https://huggingface.co/datasets/imageomics/2018-NEON-beetles/resolve/0420eb8c5d582b83220f16aa2f11f36e2e832674/BeetleMeasurements_resized.csv", low_memory = False)
df.head()

# %%
# get metadata from HF
df_detail = pd.read_csv("https://huggingface.co/datasets/imageomics/2018-NEON-beetles/resolve/54c160e18d3032e4f13003691bb514db4eef4ece/BeetleMeasurements.csv", low_memory = False)
df_detail.head(2)

# %%
print(df["pictureID"].nunique(), df.shape)

# %%
cols_to_add = ["scientificName", "genus", "species", "NEON_sampleID", "siteID"]

# %%
for pic_id in list(df["pictureID"]):
    temp = df_detail.loc[df_detail["pictureID"] == pic_id].copy()
    for col in cols_to_add:
        df.loc[df["pictureID"] == pic_id, col] = temp[col].values[0]

df.head()

# %%
df.to_csv("../data/BeetleMeasurements_resized.csv", index = False)

# %% [markdown]
# ## Update `group_images` and `group_images_masks` Metadata
#
# Run [sum-buddy](https://github.com/Imageomics/sum-buddy) for folder image contents information (run at root of repo, relative path to local copy of [HF repo](https://huggingface.co/datasets/imageomics/2018-NEON-beetles)).
# ```console
# sum-buddy --output-file metadata/group_images_sb.csv ../2018-NEON-beetles/group_images
# sum-buddy --output-file metadata/group_images_masks_sb.csv ../2018-NEON-beetles/group_images_masks
# ```

# %%
# Get metadata for images
meta_df = pd.read_csv("https://huggingface.co/datasets/imageomics/2018-NEON-beetles/resolve/10f6ed40764864e1edc0c0022f66642367161606/BeetleMeasurements.csv", low_memory=False)
meta_df.head(2)

# %%
gp_df = pd.read_csv("../metadata/group_images_sb.csv", low_memory=False)
gp_df.head()

# %%
print(meta_df.shape, gp_df.shape)

# %%
gp_meta = pd.merge(meta_df, gp_df[["filename", "md5"]],
                   left_on = "pictureID",
                   right_on = "filename",
                   how = "right")
gp_meta.shape

# %%
gp_meta["dupes"] = gp_meta.duplicated(subset = ["pictureID", "filename", "md5"], keep = "first")
gp_meta["dupes"].value_counts()

# %%
gp_meta_cleaned = gp_meta.loc[~gp_meta["dupes"]].copy()
gp_meta_cleaned.shape

# %%
gp_meta[["pictureID", "filename", "md5"]].nunique()

# %% [markdown]
# We have one `pictureID` duplicated, though the `filename` and `md5` are unique. It should be a unique identifier, so why is that happening?

# %%
gp_meta_cleaned["double-picID"] = gp_meta_cleaned.duplicated("pictureID", keep = False)
gp_meta_cleaned.loc[gp_meta_cleaned["double-picID"]]

# %% [markdown]
# Or it's missing...

# %%
gp_meta_cleaned[["pictureID", "filename", "md5"]].info()

# %%
gp_meta_cleaned.loc[gp_meta_cleaned["pictureID"].isna()]

# %% [markdown]
# It looks like this has an extra `0` added.

# %%
meta_df.loc[meta_df["pictureID"] == "A0000006924.jpg"]

# %% [markdown]
# Actually, found the image in the folder. Opened the image, its ID should be `A00000069245.jpg`, as the ID comes from the tube the beetles were in and `A00000069245` is the code on the tube label. We'll rename this image and update the file.

# %%
gp_df.loc[gp_df["filename"] == "A00000006924.jpg"] = "A00000069245.jpg"

gp_meta = pd.merge(meta_df, gp_df[["filename", "md5"]],
                   left_on = "pictureID",
                   right_on = "filename",
                   how = "inner")
print(gp_meta.shape)

gp_meta["dupes"] = gp_meta.duplicated(subset = ["pictureID", "filename", "md5"], keep = "first")
print(gp_meta["dupes"].value_counts())

gp_meta_cleaned = gp_meta.loc[~gp_meta["dupes"]].copy()
gp_meta_cleaned.shape

# %% [markdown]
# Now we just adjust the `filename` column to be `file_name` for HF dataset viewer and drop the `dupes` column.

# %%
gp_meta_cleaned.rename(columns={"filename": "file_name"}, inplace=True)
gp_meta_cleaned.drop(columns="dupes", inplace=True)
gp_meta_cleaned.head(2)

# %%
# Add subset Column
gp_meta_cleaned["subset"] = "group_images"

# %% [markdown]
# ### Save `group_images` metadata
#
# Use relative path to local copy of [HF repo](https://huggingface.co/datasets/imageomics/2018-NEON-beetles).

# %%
gp_meta_cleaned.to_csv("../../2018-NEON-beetles/group_images/metadata.csv", index = False)

# %% [markdown]
# ### Update Masks Subset
#
# Check for image that was mislabeled.

# %%
gp_m_df = pd.read_csv("../metadata/group_images_masks_sb.csv", low_memory=False)

gp_m_df.loc[gp_m_df["filename"] == "A00000006924.jpg"]

# %%
gp_m_df.head()

# %%
gp_m_df.loc[gp_m_df["filename"] == "A00000006924_mask.png"]

# %% [markdown]
# Okay, let's rename this and then we'll create a `pictureID` column to merge with larger dataframe.

# %%
gp_m_df.loc[gp_m_df["filename"] == "A00000006924_mask.png", "filename"] = "A00000069245_mask.png"

# %%
for pic in list(gp_m_df["filename"]):
    gp_m_df.loc[gp_m_df["filename"] == pic, "pictureID"] = pic.split("_mask")[0] + ".jpg"

gp_m_df.head()

# %%
print(gp_m_df.shape)

gp_m_meta = pd.merge(meta_df, gp_m_df[["filename", "md5", "pictureID"]],
                   on = "pictureID",
                   how = "inner")
print(gp_m_meta.shape)

gp_m_meta["dupes"] = gp_m_meta.duplicated(subset = ["pictureID", "filename", "md5"], keep = "first")
print(gp_m_meta["dupes"].value_counts())

gp_m_meta_cleaned = gp_m_meta.loc[~gp_m_meta["dupes"]].copy()
gp_m_meta_cleaned.shape

# %%
gp_m_meta_cleaned.head(2)

# %% [markdown]
# Now we just adjust the `filename` column to be `file_name` for HF dataset viewer and drop the `dupes` column.

# %%
gp_m_meta_cleaned.rename(columns={"filename": "file_name"}, inplace=True)
gp_m_meta_cleaned.drop(columns="dupes", inplace=True)
gp_m_meta_cleaned.head(2)

# %%
# Add subset Column
gp_m_meta_cleaned["subset"] = "group_images_masks"

# %% [markdown]
# #### Save `group_images_masks` metadata file
#
# Use relative path to local copy of [HF repo](https://huggingface.co/datasets/imageomics/2018-NEON-beetles).

# %%
gp_m_meta_cleaned.to_csv("../../2018-NEON-beetles/group_images_masks/metadata.csv", index = False)

# %%
