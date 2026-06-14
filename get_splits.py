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

# Read MMSI + gear_report first
split_df = pd.concat(
    [
        pd.read_parquet(f, columns=["mmsi", "gear_report"])
        for f in files
    ],
    ignore_index=True,
)

# One row per MMSI.
# If an MMSI has multiple gear_report values, use the most frequent one.
mmsi_gear = (
    split_df
    .groupby("mmsi")["gear_report"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
)

split_rng = np.random.default_rng(5)

val_mmsi = set()
test_mmsi = set()

for gear, group in mmsi_gear.groupby("gear_report"):
    gear_mmsis = group["mmsi"].to_numpy()
    split_rng.shuffle(gear_mmsis)

    n = len(gear_mmsis)
    n_val = n // 2

    val_mmsi.update(gear_mmsis[:n_val])
    test_mmsi.update(gear_mmsis[n_val:])

print("Validation MMSIs:", len(val_mmsi))
print("Test MMSIs:", len(test_mmsi))
print("Overlap:", len(val_mmsi & test_mmsi))

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