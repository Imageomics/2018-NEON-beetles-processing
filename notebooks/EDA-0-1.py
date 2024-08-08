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
#
# This dataset has images of multiple beetles from each site (the individual beetles will be segmented out). Each image has multiple rows in the measurements CSV (2 per beetle, one representing each of the two measurements performed on the beetles).

# %%
df = pd.read_csv("https://huggingface.co/datasets/imageomics/BeetlePalooza/resolve/bbefad05d50ed55da82e99bd330afe12a5fd1d97/BeetleMeasurements.csv", low_memory = False)
df.head()

# %% [markdown]
# We'll want to remove the `Unnamed: 0` column or set it as index (would be good to have some type of UUID for this CSV). `NEON_sampleID_` should be changed to either `NEON_sample_id` or have the trailing underscore removed; need consistency across the "ID" columns.
#
# Remaining column info:
#   - `PictureID`: Name of the image: `<sample-barcode>.jpg`. Unique identifier for _group_ images, not for dataset. --- Probably should lowercase this.
#   - `scalebar`: Pixel coordinates of the ruler/scalebar in the image.
#   - `scale_dist_pix`: Integer. The length in pixels of the scalebar.
#   - `individual`: Integer. The beetle in the image to whom the measurements refer. Awaiting confirmation of how they are counted.
#   - `structure`: Whether the measurement applies to the length or width of the elytra (`ElytraLength` or `ElytraWidth`, respectively).
#   - `lyingstraight`: Whether or not the beetle is "straight" in the image (`Yes` or `No`). Further guidance on this term's meaning would be helpful.
#   - `coords_pix`: Pixel coordinates of the line marking the length or width of the elytra (green or purple --confirm which is which). Note that the lines are more than one pixel wide, which is why these coordinates form a rectangle.
#   - `dist_pix`: Float. The length or width of the elytra (indicated by `structure`) as measured in pixels.
#   - `dist_cm`: Float. The length or width of the elytra (indicated by `structure`) as measured in centimeters using the scalebar (the red line in the reference image denotes the pixel count for 1cm).
#   - `scientificName`: Scientific name of the specimen (`<Genus> <species>`).
#   - `siteID`: String. Identifier for the site from which the specimens were collected.
#   - `field_site_name`: Name of site from which the specimens were collected.
#   - `plotID`: Identifier for the plot from which the specimens were collected (`<siteID>_<plot number>`).
#   - `user_name`: Name of person inputting the information? (`<first><Last>`) or just their username in the system?
#   - `workflow_id`: Integer identifier for the workflow used...??
#

# %%
df.info()

# %% [markdown]
# We have 34 entries (presumably 17 individuals) without `scientificName`. Everything else is completely filled in. Should check how many images that is and overall info. Everything else is completely filled.

# %%
df.nunique()


# %% [markdown]
# We will need to count uniqueness as pair of `PictureID` + `individual` to get count of individuals.

# %%
def get_genus(sci_name):
    if type(sci_name) == float:
        return sci_name
    return sci_name.split(" ")[0]

df["genus"] = df["scientificName"].apply(get_genus)
df["genus"].nunique()

# %% [markdown]
# There are 85 different species among 36 genera.
#
# Now let's check on the missing `scientificName`.

# %%
missing_sci = df.loc[df.scientificName.isna()].copy()
missing_sci.nunique()

# %% [markdown]
# It is just one image with 17 individuals.

# %%
missing_sci.sample()


# %% [markdown]
# Sometimes the `PictureID` is the `NEON_sampleID_`.

# %%
def generate_indivdiual_id(pictureID, individual):
    return pictureID.split(".jpg")[0] + "_" + str(individual)

df["individualID"] = df.apply(lambda x: generate_indivdiual_id(x["PictureID"], x["individual"]), axis = 1)
df["individualID"].nunique()

# %% [markdown]
# But there should be 19,532 since there are 39,064 entries and should have two per individual...well `coords_pix` is not entirely unique...

# %%
df.sample()

# %% [markdown]
# Let's remove that index column and check for duplication.

# %%
df = df[list(df.columns)[1:]].copy()
df["duplicate"] = df.duplicated(keep = "first")
df["duplicate"].value_counts()

# %% [markdown]
# Let's try ignoring the `user_name` to check.

# %%
df["duplicate"] = df.duplicated(subset = [col for col in list(df.columns) if col != "user_name"], keep = "first")
df["duplicate"].value_counts()

# %% [markdown]
# Let's check the `coords_pix` that are duplicated.

# %%
df["coords_dupe"] = df.duplicated(subset = ["coords_pix"], keep = False)
df["coords_dupe"].value_counts()

# %% [markdown]
# Still not enough to account for the difference, but let's take a sample of them.

# %%
df.loc[df["coords_dupe"]].head(7)

# %% [markdown]
# Based on comparing a few of these, the duplication seems to be caused by variations in the person inputting the data. They match the line coordinates, but the scalebar coordinates are slightly different (in one case just which is 1 vs 2) and the cm calculation differs (despite matching pixel distance). We also see here a disagreement on `lyingstraight` here.

# %%
for username in list(df.user_name.unique()):
    temp = df.loc[df["user_name"] == username].copy()
    print(f"{username} annotated {temp.PictureID.nunique()} images in {temp.shape[0]} entries, with {temp.individualID.nunique()} unique individuals (picID + individual)")

# %% [markdown]
# Looks like `IsaFluck` annotated all images, but not all individuals. The other two people each annotated about half. Based on the numbers per annotator there are at least 2 images that were looked at by 3 people (if only 2, then all other images were reviewed by 2 people).
#
# They all seem to have more than two rows per individual (as in a handful of individuals got 3+ entries for each annotator). Here I'm counting individuals using the `PictureID` (minus the `.jpg`) plus the `individual` number from the image.
#
# Some inconsistency could also come from different numbering schemes, though that would suggest many more issues.

# %%
df["dupeID"] = df.duplicated(subset = ["individualID"], keep = "first")

individual_count_df = df.loc[~df["dupeID"]].copy()

individual_count_df.info()

# %%
sns.histplot(individual_count_df, y = "genus", hue = "scientificName", legend = False)

# %% [markdown]
# ### Add `file_name` Column for HF
#
# Add the `file_name` column with relative path from root of directory to the images for the dataset viewer. It should be `group_images/<PictureID>`.

# %%
df["file_name"] = "group_images/" + df["PictureID"]
df["file_name"].nunique()

# %% [markdown]
# ### Quick Look at Size Comparisons
#
# Let's just take the first measurement for length and width of the elytra (in cm) to add for a column on the individual subset (curious about range). This we won't worry about saving since the individual CSV is going to be made after we address the annotation inconsistencies.
#  - Next step will be to compare the measurements for each individual and see what the variation is across the estimated 2-3 annotations per individual.

# %%
for id in list(individual_count_df["individualID"]):
    meas_temp = df.loc[df["individualID"] == id].copy()
    
    elytra_length = meas_temp.loc[meas_temp["structure"] == "ElytraLength", "dist_cm"].values[0]
    elytra_width = meas_temp.loc[meas_temp["structure"] == "ElytraWidth", "dist_cm"].values[0]
    
    individual_count_df.loc[individual_count_df["individualID"] == id, "elytraLength"] = elytra_length
    individual_count_df.loc[individual_count_df["individualID"] == id, "elytraWidth"] = elytra_width

# %%
sns.scatterplot(individual_count_df, x = "elytraLength", y = "elytraWidth", hue = "genus", legend = False)

# %% [markdown]
# Comparing annotations for individuals will be particularly interesting (there's one big outlier here)...who is that guy?

# %%
individual_count_df.loc[(individual_count_df["elytraWidth"] > .8) & (individual_count_df["elytraLength"] < .5)]

# %%
df.loc[df["individualID"] == "A00000046078_10"]

# %% [markdown]
# Ahh, `A00000046078_10` was annotated 3 times and had length and width transposed for one of the measurements! (likely just the one and the other two are correct, since they're generally longer than wide---the width isn't at the widest point)
#
#
# Big note to address the annotations!!
#
#
# ### Save updated CSV with 3 new columns

# %%
df.columns

# %% [markdown]
# We'll save the column rename until after a conversation with the team, but let's save this updated copy (without the duplicate markers).

# %%
cols_to_keep = [col for col in list(df.columns) if col not in ['duplicate', 'coords_dupe', 'dupeID']]
df[cols_to_keep].to_csv("../data/BeetleMeasurements.csv", index = False)

# %%

# %% [markdown]
# ## Re-open to Add Species and Update Column Names
#
# We'll add a species column to more easily identify those labeled only to the genus level. Additionally, we will standardize the column names: ID values will be camelCase and we'll set the remainder with underscores (there's a bit of variety, but NEON seems to use camelCase for the IDs (location & type of measurement): https://www.neonscience.org/data-samples/data-management/data-formats-conventions).

# %%
df = pd.read_csv("https://huggingface.co/datasets/imageomics/BeetlePalooza/resolve/190cfc9c71b813b8faa338d251d3d02ee8dc14a9/BeetleMeasurements.csv", low_memory = False) 
df.head()

# %%
import numpy as np

# %%
longer_sp_names = []

def get_species(sciName):
    if (type(sciName) != float) and ("sp." not in sciName):
        names = sciName.split(" ")
        if len(names) == 2:
            return(names[1])
        elif len(names) > 2:
            longer_sp_names.append(sciName)
            genus_ = names[0] + " "
            return sciName.split(sep = genus_)[1]
    return np.nan


# %%
df["species"] = df["scientificName"].apply(get_species)
print(f"Among our 36 unique genera we have {df['species'].nunique()} unique species")
print(f"There are {len(set(longer_sp_names))} species labels longer than one word; their scientific names are: {set(longer_sp_names)}")
print(f"We have {df.loc[df['species'].isna(), 'individualID'].nunique()} individuals not labeled to the species level; {df.loc[(df['species'].isna()) & (df['scientificName'].isna()), 'individualID'].nunique()} of them don't even have a scientific name label.")

# %%
df.loc[(df["species"].isna()),"scientificName"].value_counts()

# %%
"Abacidus" in list(df["genus"].unique())

# %% [markdown]
# `Abacidus` seems to be a subgenus of `Pterostichus` ([ref](https://bugguide.net/node/view/1740293)). Those counts seem a bit low for the number of individuals in images...

# %%
"sp" in list(df["species"].unique())

# %%
"(" in list(df["species"].unique())

# %%
df["species"].isna().sum()

# %% [markdown]
# Total of 62 entries missing species, but it's only 27 individuals (it would seem), and we got one subgenus.

# %%
df.sample(7)

# %%
df.columns

# %%
df.rename(columns = {'PictureID': 'pictureID',
                     'scale_dist_pix': 'cm_pix',        #this is number of pixels in a cm on the scalebar
                     'lyingstraight': 'lying_flat',     # this is actually about how flat they are on the lattice (vs rolled on the side; it can distort width measure)
                     'NEON_sampleID_': 'NEON_sampleID',
                     'workflow_id': 'workflowID',
                     'field_site_name': 'site_name'
                     },
          inplace = True)
df.head(2)

# %% [markdown]
# Generate true `measureID` to give us a unique identifier for the dataset; this way we can merge with the inidividual CSV we'll create from Isadora's measurements. We'll also save the current `individualID` column as `combinedID`.

# %%
import uuid

df.rename(columns = {'individualID': 'combinedID'}, inplace = True)

df["measureID"] = [uuid.uuid4() for i in range(df.shape[0])]

df.head()

# %% [markdown]
# Now let's just move `species` to follow `genus` and `measureID` to follow `combinedID`, then we'll save this updated CSV.

# %%
col_list = list(df.columns)[:-2]
col_list.insert(-2, "species")
col_list.insert(-1, "measureID")
col_list

# %%
df[col_list].head(2)

# %%
df[col_list].to_csv("../data/BeetleMeasurements.csv", index = False)

# %%
df["measureID"].nunique()

# %% [markdown]
# Quick check, what's the scientific name of our sample image (use the image on [GitHub](https://github.com/Imageomics/BeetlePalooza-2024/blob/main/BeetleImage-A00000046094.jpg))?

# %%
df.loc[df["pictureID"] == "A00000046094.jpg", "scientificName"].values[0]

# %%
