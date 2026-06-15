import pandas as pd
import numpy as np

""" files = [
    "three_months/feats_new_rule_online/2024_1_3_feats.parquet",
    "three_months/feats_new_rule_online/2024_4_6_feats.parquet",
    "three_months/feats_new_rule_online/2024_7_9_feats.parquet",
    "three_months/feats_new_rule_online/2024_10_12_feats.parquet",
] """

files = ["2024_1_3.parquet"]

all_mmsis = set()
for f in files:
    print("reading", f)
    m = pd.read_parquet(f, columns=["mmsi"])["mmsi"].unique()
    all_mmsis.update(m)

mmsis = np.array(list(all_mmsis))
split_rng = np.random.default_rng(5)
split_rng.shuffle(mmsis)
n = len(mmsis)
train_mmsi = set(mmsis[:int(0.70 * n)])
val_mmsi   = set(mmsis[int(0.70 * n):int(0.85 * n)])
test_mmsi  = set(mmsis[int(0.85 * n):])

df_mmsis = pd.DataFrame({
    "mmsi": np.concatenate([
        list(train_mmsi),
        list(val_mmsi),
        list(test_mmsi),
    ]),
    "split": (
        ["train"] * len(train_mmsi)
        + ["validation"] * len(val_mmsi)
        + ["test"] * len(test_mmsi)
    ),
})

print(df_mmsis.head())
#df_mmsis.to_csv("train_val_test_mmsis.csv", index=False)

print(len(train_mmsi))

# Read only the necessary columns
split_df = pd.concat(
    [
        pd.read_parquet(f, columns=["mmsi", "report"])
        for f in files
    ],
    ignore_index=True,
)

# Assign each message to a split based on MMSI
split_df["split"] = np.select(
    [
        split_df["mmsi"].isin(train_mmsi),
        split_df["mmsi"].isin(val_mmsi),
        split_df["mmsi"].isin(test_mmsi),
    ],
    [
        "Train",
        "Validation",
        "Test",
    ],
    default="Unknown",
)

print(split_df.head())

# Number of unique MMSIs per gear type and split
mmsis_per_gear = (
    split_df
    .groupby(["report", "split"])["mmsi"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(columns=["Train", "Validation", "Test"], fill_value=0)
)

mmsis_per_gear["Total"] = mmsis_per_gear[
    ["Train", "Validation", "Test"]
].sum(axis=1)

print(mmsis_per_gear)