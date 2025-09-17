import asyncio
import fastapi
import pandas as pd
import requests

import xgboost
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

import ray
import ray.tune
import ray.train
from ray.train.xgboost import XGBoostTrainer as RayTrainXGBoostTrainer
from ray.train import RunConfig
import ray.data
import ray.serve

from pathlib import Path

# NOTE: We will be using the XGBoost ( a gradient boosted trees framework ) model to the June 2021 New York City Taxi Ride-Sharing Dataset
# The goal is to predict the tip_amount
features = ["passenger_count", "trip_distance", "fare_amount", "tolls_amount"]
label_column = "tip_amount"


def load_data():
    path = "s3://anyscale-public-materials/nyc-taxi-cab/yellow_tripdata_2021-03.parquet"
    # NOTE: only these columns will be read from the file
    df = pd.read_parquet(path, columns=features + [label_column])
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df[label_column],
        test_size=0.2,
        random_state=42,  # used for reproducibility
    )
    # INFO: A DMatrix is a data structure internally used by XGBoost
    # It can be thought of as a 2D matrix with a label column
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)

    return dtrain, dtest


storage_folder = "./cluster_storage"


model_path = f"{storage_folder}/model.ubj"


def my_xgboost_func(params):
    evals_result = {}  # NOTE: This will store the evaluation results
    dtrain, dtest = load_data()
    bst = xgboost.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10,
        evals=[(dtest, "eval")],
        evals_result=evals_result,
    )
    bst.save_model(model_path)
    return {
        "eval-rmse": evals_result["eval"]["rmse"][-1]
    }  # INFO: we get the last value of the rmse metric


params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.1,
}
my_xgboost_func(params)
