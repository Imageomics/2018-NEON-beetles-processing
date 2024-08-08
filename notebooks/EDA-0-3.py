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
#     display_name: std
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import seaborn as sns

# %% [markdown]
# # Beetlepalooza Beetle Measurement Data

# %%
df = pd.read_csv("https://huggingface.co/datasets/imageomics/BeetlePalooza/resolve/15a82c862588b2e7b709b1aa982161d8c3a7c75f/BeetleMeasurements.csv", low_memory = False)
#df = pd.read_csv("../data/BeetleMeasurements.csv", low_memory = False)
df.head()

# %%
df.info()

# %%
df.nunique()

# %% [markdown]
# ### Fix Outliers for BeetleMeasurements.csv
#
# We saw in EDA-0-2 that we have 2 outliers: Looking at picture, the first one (`A00000046078_10`, annotated by `IsaFluck`) is missing half the elytra (length-wise cut). `A00000046104_10`, annotated by `rileywolcheski`, is just at an angle, length is definitely more than the width. We'll have to adjust both of these to be labeled correctly.
#
# We'll get the `measureID` for each so we can switch the length/width labels, then save the updated `BeetleMeasurements.csv` (this will be the final version). From there we'll update `all_measurement.csv` and create `individual_metadata.csv` with just Isadora's annotations so that we should have one row/pair of measurements per beetle.

# %%
df.loc[(df["combinedID"] == "A00000046078_10") & (df["user_name"] == "IsaFluck")]

# %% [markdown]
# Thankfully there is just one individual with this `combinedID`, so we can save the length and width `measureID`s and re-asign those labels in the `structure` column.

# %%
len_meas_id = df.loc[(df["combinedID"] == "A00000046078_10") & (df["user_name"] == "IsaFluck") & (df["structure"] == "ElytraLength"), "measureID"].values[0]
w_meas_id = df.loc[(df["combinedID"] == "A00000046078_10") & (df["user_name"] == "IsaFluck") & (df["structure"] == "ElytraWidth"), "measureID"].values[0]

print(f"We will reasign the measure {len_meas_id} to be structure 'ElytraWidth', and measure {w_meas_id} to be structure 'ElytraLength")

# %%
df.loc[df["measureID"] == len_meas_id, "structure"] = "ElytraWidth"
df.loc[df["measureID"] == w_meas_id, "structure"] = "ElytraLength"

df.loc[(df["combinedID"] == "A00000046078_10") & (df["user_name"] == "IsaFluck")]

# %% [markdown]
# Perfect!
#
# Now let's fix the other measurements from `A00000046104_10`. Should also be just the two measurements (lenght & width), but we'll double check before proceeding.

# %%
df.loc[(df["combinedID"] == "A00000046104_10") & (df["user_name"] == "rileywolcheski")]

# %% [markdown]
# Yep, just the two!

# %%
len_meas_id = df.loc[(df["combinedID"] == "A00000046104_10") & (df["user_name"] == "rileywolcheski") & (df["structure"] == "ElytraLength"), "measureID"].values[0]
w_meas_id = df.loc[(df["combinedID"] == "A00000046104_10") & (df["user_name"] == "rileywolcheski") & (df["structure"] == "ElytraWidth"), "measureID"].values[0]

print(f"We will reasign the measure {len_meas_id} to be structure 'ElytraWidth', and measure {w_meas_id} to be structure 'ElytraLength")

# %%
df.loc[df["measureID"] == len_meas_id, "structure"] = "ElytraWidth"
df.loc[df["measureID"] == w_meas_id, "structure"] = "ElytraLength"

df.loc[(df["combinedID"] == "A00000046104_10") & (df["user_name"] == "rileywolcheski")]

# %% [markdown]
# ### Save Updated Beetle Measurement CSV

# %%
df.to_csv("../data/BeetleMeasurements.csv", index = False)

# %% [markdown]
# ## Make Individual Measurement Dataset
#
# First, we will update the `all_measurements` CSV, creating an analyzable CSV with measurements by each annotator for each individual (each row will be one pair of measurements).
#
# Then we will reduce to just unique individuals (by reducing to just `user_name == "IsaFluck"`).

# %%
df_meas = pd.DataFrame({"combinedID": list(df.loc[df["structure"] == "ElytraWidth", "combinedID"]),
                        "lying_flat": list(df.loc[df["structure"] == "ElytraWidth", "lying_flat"]),    # This is most important/relevant for width measurement anyway (in case of inconsistency, which shouldn't occur)
                        "coords_pix_length": list(df.loc[df["structure"] == "ElytraLength", "coords_pix"]),
                        "coords_pix_width": list(df.loc[df["structure"] == "ElytraWidth", "coords_pix"]),
                        "elytraLength_pix": list(df.loc[df["structure"] == "ElytraLength", "dist_pix"]),
                        "elytraWidth_pix": list(df.loc[df["structure"] == "ElytraWidth", "dist_pix"]),
                        "elytraLength_cm": list(df.loc[df["structure"] == "ElytraLength", "dist_cm"]),
                        "elytraWidth_cm": list(df.loc[df["structure"] == "ElytraWidth", "dist_cm"]),
                        "measureID_length": list(df.loc[df["structure"] == "ElytraLength", "measureID"]),
                        "measureID_width": list(df.loc[df["structure"] == "ElytraWidth", "measureID"]),
                        "genus": list(df.loc[df["structure"] == "ElytraWidth", "genus"]),
                        "species": list(df.loc[df["structure"] == "ElytraWidth", "species"]),
                        "NEON_sampleID": list(df.loc[df["structure"] == "ElytraWidth", "NEON_sampleID"]),
                        "user_name": list(df.loc[df["structure"] == "ElytraWidth", "user_name"]),
                        }) # should match up
df_meas.head()

# %% [markdown]
# ### A couple plots before we save this

# %%
# color by genus

sns.scatterplot(df_meas, x = "elytraLength_cm", y = "elytraWidth_cm", hue = "genus", legend = False)

# %%
# color by species

sns.scatterplot(df_meas, x = "elytraLength_cm", y = "elytraWidth_cm", hue = "species", legend = False)

# %% [markdown]
# ### Save Updated Copy of All Measurements
# We've now added measurment IDs and fixed the two measurements that were reversed. We'll keep the All Measurements CSV available for comparison across annotators in case there's an interest.

# %%
df_meas.to_csv("../metadata/all_measurements.csv", index = False)

# %% [markdown]
# ## Make Individual CSV
#
# Will have one pair of measurements for each individual with an individual ID based on the measurement IDs. These will be just the measurements done by `user_name == "IsaFluck"`, since she annotated each image and this is the only way to ensure uniqueness based on the Zooniverse individual labeling export.

# %%
df_individual = df_meas.loc[df_meas["user_name"] == "IsaFluck"].copy()
df_individual.info()

# %%
df_individual.nunique()

# %% [markdown]
# This seems like we've lost some individuals considering there were 11,104 unique `combinedID`s in the full dataset, but this is what we have to work with.

# %% [markdown]
# ### Plots

# %%
# color by genus

sns.scatterplot(df_individual, x = "elytraLength_cm", y = "elytraWidth_cm", hue = "genus", legend = False)

# %%
# color by species

sns.scatterplot(df_individual, x = "elytraLength_cm", y = "elytraWidth_cm", hue = "species", legend = False)

# %% [markdown]
# ### Add Individual ID and Save
#
# We'll add the `individualID`: `<measureID_length>_<measureID_width>` and `file_name`: `individual_images/<individualID>.jpg`, then save this as `individual_metadata_full.csv` for segmentation. Then we'll make the paired-down `individual_metadata.csv` as described in the HF README:
#
#   - `individualID`: ID of beetle in the individual image (`<measureID_length>_<measureID_width>`). This is a unique identifier for this CSV.
#   - `combinedID`: Generated from `PictureID` (minus the `.jpg`) plus `_<individual>`. (Matches `combinedID` in `BeetleMeasurements.csv`.)
#   - `elytraLength`: Length of the elytra in centimeters. Indicated by the green line in the image below.
#   - `elytraWidth`: Width of the elytra in centimeters. Indicated by the purple line in the image below.
#   - `measureID_length`: `measureID` from `BeetleMeasurements.csv` for the `elytraLength` of this individual. Can be used to fetch measure-specific information from `BeetleMeasurements.csv`.
#   - `measureID_width`: `measureID` from `BeetleMeasurements.csv` for the `elytraWidth` of this individual. Can be used to fetch measure-specific information from `BeetleMeasurements.csv`.
#   - `genus`: Genus of the individual (generated by taking the first word in the `scientificName` from `BeetleMeasurements.csv`). There are 36 unique genera labeled.
#   - `species`: Species of the individual (generated from the word(s) following the `genus` in the `scientificName` from `BeetleMeasurements.csv). There are 78 unique species labeled.
#   - `NEON_sampleID`: NEON identifier for the sample (576 unique IDs), prefixed by the `plotID`. (Matches `NEON_sampleID` in `BeetleMeasurements.csv`.)
#   - `file_name`: Relative path to image from the root of the directory (`individual_images/<individualID>.jpg`); allows for image to be displayed in the dataset viewer alongside its associated metadata.

# %%
cols = list(df_individual.columns)
cols.insert(0, "individualID")
cols

# %%
df_individual["individualID"] = df_individual["measureID_length"] + "_" + df_individual["measureID_width"]
df_individual = df_individual[cols].copy()
df_individual.head()

# %%
df_individual["file_name"] = "individual_images/" + df_individual["individualID"] + ".jpg"
df_individual.head()

# %%
df_individual.to_csv("../metadata/individual_metadata_full.csv", index = False)

# %%
cols_hf = [col for col in list(df_individual.columns) if "pix" not in col]

df_ind_hf = df_individual[cols_hf].copy()
df_ind_hf.head()

# %%
df_ind_hf.rename(columns = {"elytraLength_cm": "elytraLength",
                            "elytraWidth_cm": "elytraWidth"},
                 inplace = True)
df_ind_hf.head(2)

# %%
# remove user_name column

cols_to_keep = [col for col in list(df_ind_hf.columns) if col != "user_name"]

# %%
df_ind_hf[cols_to_keep].to_csv("../data/individual_metadata.csv", index = False)

# %%
