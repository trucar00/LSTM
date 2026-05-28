import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
import os
import pickle
import json
from pathlib import Path

files = [
    #"three_months/feats_all_gear2/2023_1_3_feats.parquet",
    #"three_months/feats_all_gear2/2023_4_6_feats.parquet",
    #"three_months/feats_all_gear2/2023_7_9_feats.parquet",
    #"three_months/feats_all_gear2/2023_10_12_feats.parquet",
    "three_months/feats_all_gear2/2024_1_3_feats.parquet",
    "three_months/feats_all_gear2/2024_4_6_feats.parquet",
    #"three_months/feats_all_gear2/2024_7_9_feats.parquet",
    #"three_months/feats_all_gear2/2024_10_12_feats.parquet"
]

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt", "dist_to_shore_km"]

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES

all_mmsis = set()

for f in files:
    print("reading ", f)
    m = pd.read_parquet(f, columns=["mmsi"])["mmsi"].unique()
    all_mmsis.update(m)

mmsis = np.array(list(all_mmsis))
rng = np.random.default_rng(42)
rng.shuffle(mmsis)

# Split into train test and validation set by mmsi so that no vessel appear in both.
n = len(mmsis)
train_mmsi = set(mmsis[:int(0.70*n)])
val_mmsi   = set(mmsis[int(0.70*n):int(0.85*n)])
test_mmsi  = set(mmsis[int(0.85*n):])

mu_sigma_path = Path(f"parameters_2024_1_3_4_6.pkl")
if mu_sigma_path.exists():
    print(f"Loading mu/sigma from {mu_sigma_path}")
    with open(mu_sigma_path, "rb") as f:
        params = pickle.load(f)

    mu = params["mu"]
    sigma = params["sigma"]

else:

    # Fit normalization on TRAIN ONLY
    sum_x = pd.Series(0.0, index=FEATURES)
    sum_x2 = pd.Series(0.0, index=FEATURES)
    count = 0

    needed_cols = ["mmsi", "date_time_utc"] + BASE_FEATURES

    for f in files:
        df = pd.read_parquet(f, columns=needed_cols)
        df = df[df["mmsi"].isin(train_mmsi)].copy()

        df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
        month = df["date_time_utc"].dt.month

        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        x = df[FEATURES]
        sum_x += x.sum()
        sum_x2 += (x ** 2).sum()
        count += len(x)

    mu = sum_x / count
    sigma = np.sqrt((sum_x2 / count) - mu**2).replace(0, 1)
    print("mu and sigma found")

    with open(mu_sigma_path, "wb") as f:
        pickle.dump({"mu": mu, "sigma": sigma}, f)