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
#df = pd.read_csv("https://huggingface.co/datasets/imageomics/BeetlePalooza/resolve/15a82c862588b2e7b709b1aa982161d8c3a7c75f/BeetleMeasurements.csv", low_memory = False)
df = pd.read_csv("../data/BeetleMeasurements.csv", low_memory = False)
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

# %%
