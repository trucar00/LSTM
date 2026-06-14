import pandas as pd
import numpy as np

files = [
    "three_months/feats_new_rule_online/2024_1_3_feats.parquet",
    "three_months/feats_new_rule_online/2024_4_6_feats.parquet",
    "three_months/feats_new_rule_online/2024_7_9_feats.parquet",
    "three_months/feats_new_rule_online/2024_10_12_feats.parquet",
]

# ============================================================
# MMSI split — fixed across seeds so train/val/test are identical
# ============================================================

all_mmsis = set()
for f in files:
    print("reading", f)
    m = pd.read_parquet(f, columns=["mmsi"])["mmsi"].unique()
    all_mmsis.update(m)

mmsis = np.array(list(all_mmsis))
split_rng = np.random.default_rng(10)
split_rng.shuffle(mmsis)

n = len(mmsis)
val_mmsi = set(mmsis[: n // 2])
test_mmsi = set(mmsis[n // 2 :])

split_path = "../split_mmsis_val_test.csv"

mmsi_split = pd.DataFrame({
    "mmsi": (
        list(val_mmsi)
        + list(test_mmsi)
    ),
    "split": (
        ["validation"] * len(val_mmsi)
        + ["test"] * len(test_mmsi)
    ),
})

mmsi_split = mmsi_split.sort_values(
    ["split", "mmsi"]
).reset_index(drop=True)

mmsi_split.to_csv(split_path, index=False)

print(f"Saved MMSI split to {split_path}")
print(mmsi_split["split"].value_counts())


# Read only the necessary columns
split_df = pd.concat(
    [
        pd.read_parquet(f, columns=["mmsi", "gear_report"])
        for f in files
    ],
    ignore_index=True,
)

# Assign each message to a split based on MMSI
split_df["split"] = np.select(
    [
        split_df["mmsi"].isin(val_mmsi),
        split_df["mmsi"].isin(test_mmsi),
    ],
    [
        "Validation",
        "Test",
    ],
    default="Unknown",
)

# Number of unique MMSIs per gear type and split
mmsis_per_gear = (
    split_df
    .groupby(["gear_report", "split"])["mmsi"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(columns=["Validation", "Test"], fill_value=0)
)

mmsis_per_gear["Total"] = mmsis_per_gear[
    ["Validation", "Test"]
].sum(axis=1)

print(mmsis_per_gear)