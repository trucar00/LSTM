import pandas as pd
import numpy as np

FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk","log_dist", "ra_dcog", "log_dt"]

df = pd.read_csv("ais_labeled_features.csv")

first_50_mmsis = df["mmsi"].drop_duplicates().head(50)

df_50 = df[df["mmsi"].isin(first_50_mmsis)]

print(df_50[FEATURES].isna().sum())
print(np.isinf(df_50[FEATURES].to_numpy()).sum(axis=0))

print(df_50.head())

label_counts = df_50["y"].value_counts(dropna=False)
print(label_counts)

df_50.to_csv("first_50_feats.csv", index=False)