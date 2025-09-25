from pathlib import Path
import pandas as pd
import ray

# INFO: Ray data is a distributed data processing library that provides a
# Python API for paralle data processing.

# INFO: Ray is typically used in the following manner:
# 1. Create a Ray Dataset from external storage or in-memory data.
# 2. Apply transformations to the data
# 3. Write the outputs to external storage or feed the outputs to training workers

data_path = "data/yellow_tripdata_2021-03.parquet"
ds = ray.data.read_parquet(data_path)

# WARN: Ray Data by default adopts lazy evaluation. This means that the data is not loaded
# into memory until it is needed. Even in the above read_parquet call, only a small
# portion of the data is read to infer schema and other metadata.
#
# INFO: A Ray Dataset specifies a sequences of transformations that are applied to the data.
# The data itself will be organized into blocks, where each block is a collection of rows.
# Each block is an object reference to a PyArrow Table with the given schema.
# And since a Dataset is just a list of Ray object references, it can be freely passed between
# Ray actors, tasks and libraries.

print(ds.take_batch(batch_format="pandas"))


# NOTE: Data Tranformations can be applied to a Ray Dataset using the `map_batches` method.


def simple_transform(df):
    df["adjusted_total_amount"] = df["total_amount"] - df["tip_amount"]
    return df


# NOTE: The `map_batches` method will batch each block of the dataset and apply the
# function to each batch in parallel.
#
# WARN: The default `batch_format` in Ray Data is `numpy`, which means the data will be
# returned as a Numpy array. For optimal performance, it is recommmended to avoid
# converting the data to Pandas dataframes unless necessary.
print(ds.map_batches(simple_transform).take_batch(batch_format="numpy"))


# Let's add another transformation to the dataset
def compute_tip_percentage(df):
    df["tip_percentage"] = df["tip_amount"] / df["total_amount"]
    return df


df_adjusted = ds.map_batches(simple_transform, batch_format="pandas")
# NOTE: We can also control certain additional options such as the batch size
ds = df_adjusted.map_batches(
    compute_tip_percentage, batch_size=1024, batch_format="pandas"
)

# NOTE: As previously mentioned, Ray Data adopts lazy evaluation.
# Most transformations will not be applied until the data until we either:
# - write a dataset to external storage
# - explicitly materialize the dataset with functions such as `take_batch`
# - iterate over the dataset

print(ds.take_batch(20, batch_format="pandas"))


# INFO: Writing Data
# Let's write the adjusted data. There are Ray Data equivalents
# for common Pandas function such as `write_parquet` for `to_parquet` etc.

# WARN: We are using a local example
local_path = "./cluster_storage/"
# path = "/mnt/cluster_storage" For remote clusters
storage_folder = f"local://{(Path(local_path) / 'data/adjust_data_ray/').resolve()}"


# NOTE: You will see we get multiple files in the folder. This is because
# Ray Data uses Ray tasks to process data in a distributed manner.
# Each task writes its own file, and the number of files is proportional
# to the number of blocks the dataset is partitioned into. We can could
# change this by calling the `repartition()` method on the dataset.
ds.write_parquet(f"{storage_folder}")


# INFO: Shuffling, Grouping and Aggregating


# NOTE: There are different ways to shuffle the data in Ray Data with varying
# degrees of parallelism and randomness.

# INFO: 1. File Based Shuffling
# To randomly shuffle the ordering of inputs files before reading, uses the shuffle="files" parameters.
ds_file_shuffled = ray.data.read_parquet(
    data_path, shuffle="files", shuffle_seed=42
)  # WARN: specifying seed for reproducibility

# INFO: 2. Shuffling block order
# This option randomizes the order of blocks in a dataset.
# Applying this operation alone doesn't involve heavy computation.
# However, it requires Ray Data to materialize all blocks before applying the operation.

ds = ray.data.read_parquet(data_path)
# NOTE: To perform block order shuffling, use the `randomize_block_order`
# This performs a reordering within a node, which is contrasted to the global shuffling
# performed by `random_shuffle`
ds_block_shuffled = ds.randomize_block_order()

# NOTE: 3. Shuffle all rows globally
# To shuffle all rows globally, we can call `random_shuffle`. This is the slowest operation
# and requires transferring data across the network between workers. This option also
# achieves the best randomness among all options
ds_global_shuffled = ds.random_shuffle()


data_path = "data/yellow_tripdata_2021-03.parquet"
ds = ray.data.read_parquet(data_path)
# INFO: Grouping
# In case you want to group the data by a column, you can use the `group_by` method.
# Then we use the `map_groups` function to apply transformations
payment_groups = ds.groupby("payment_type")

# INFO: As for aggregations, we have many function available to the `GroupedData` object
# like `count`, `sum`, `mean`, `min`, `max`, etc
total_trip_distance_by_type = payment_groups.sum("trip_distance")
print(total_trip_distance_by_type.to_pandas())
