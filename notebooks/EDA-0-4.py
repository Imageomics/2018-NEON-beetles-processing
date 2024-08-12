# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
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
# at commit 0420eb8c5d582b83220f16aa2f11f36e2e832674
df = pd.read_csv("https://huggingface.co/datasets/imageomics/2018-NEON-beetles/resolve/main/BeetleMeasurements_resized.csv", low_memory = False)
df.head()

# %%
# at commit 54c160e18d3032e4f13003691bb514db4eef4ece
df_detail = pd.read_csv("https://huggingface.co/datasets/imageomics/2018-NEON-beetles/resolve/main/BeetleMeasurements.csv", low_memory = False)
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

# %%
