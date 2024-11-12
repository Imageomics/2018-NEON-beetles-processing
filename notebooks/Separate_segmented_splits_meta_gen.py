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
#     display_name: std
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %% [markdown]
# # Get Metadata for [Separate_segmented_train_test_splits_80_20](https://huggingface.co/datasets/imageomics/2018-NEON-beetles/tree/main/Separate_segmented_train_test_splits_80_20)
#
# Using [`sum-buddy`](https://github.com/Imageomics/sum-buddy) package within the [2018-NEON-beetles repository](https://huggingface.co/datasets/imageomics/2018-NEON-beetles):
# ```
# sum-buddy -o Separate_segmented_train_test_splits_80_20/checksum_metadata.csv Separate_segmented_train_test_splits_80_20
# ```

# %%
df = pd.read_csv("../../2018-NEON-beetles/Separate_segmented_train_test_splits_80_20/checksum_metadata.csv")
df.head()

# %% [markdown]
# ## Make `split`, `species`, and `file_name` Columns
#
# `file_name` will be path within the `Separate_segmented_train_test_splits_80_20` for the dataset viewer on HF (and we'll name the file `metadata.csv` at that point).

# %%
df["filepath"].values[:10]


# %%
def get_split(filepath):
    return filepath.split("/")[1]

def get_species(filepath):
    return filepath.split("/")[2]

def get_file_name(filepath):
    return filepath.split("splits_80_20/")[1]


# %%
df["species"] = df["filepath"].apply(get_species)
df["split"] = df["filepath"].apply(get_split)
df["file_name"] = df["filepath"].apply(get_file_name)

df["subset"] = "separate segmented splits"

# %%
df.head()

# %%
df.species.value_counts()

# %% [markdown]
# ## Display Distribution

# %%
import seaborn as sns

# %%
sns.histplot(df.sort_values("species"), y = "species", hue = "split")

# %% [markdown]
# ## Save CSV
#
# We don't need the `filepath` column since we're including the `file_name` for the purpose of the dataset viewer.

# %%
df[list(df.columns)[1:]].head()

# %%
df[list(df.columns)[1:]].to_csv("../../2018-NEON-beetles/Separate_segmented_train_test_splits_80_20/metadata.csv", index = False)

# %%
