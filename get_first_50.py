import pandas as pd
import numpy as np

get_cols = ["mmsi", "trajectory_id", "date_time_utc", "lon", "lat", "dt", "y", "y_train", "sample_weight", "cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk","log_dist", "ra_dcog", "log_dt"]
FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk","log_dist", "ra_dcog", "log_dt"]

#df = pd.read_csv("ais_labeled_features.csv", usecols=get_cols)

df = pd.read_parquet("ais_conf_labeled_features.parquet", columns=get_cols)

first_50_mmsis = df["mmsi"].drop_duplicates().head(50)

df_50 = df[df["mmsi"].isin(first_50_mmsis)]

print(df_50[FEATURES].isna().sum())
print(np.isinf(df_50[FEATURES].to_numpy()).sum(axis=0))

print(df_50.head())

label_counts = df_50["y_train"].value_counts(dropna=False)
print(label_counts)

df_50.to_parquet("first_50_feats_conf_labels.parquet", index=False)