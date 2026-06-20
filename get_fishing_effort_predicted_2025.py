import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from pathlib import Path

#jje
# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
files = [
    "predictions_all_2025/2025_all_vessels_monthpair-1.parquet",
    "predictions_all_2025/2025_all_vessels_monthpair-2.parquet",
    "predictions_all_2025/2025_all_vessels_monthpair-3.parquet",
    "predictions_all_2025/2025_all_vessels_monthpair-4.parquet",
    "predictions_all_2025/2025_all_vessels_monthpair-5.parquet",
    "predictions_all_2025/2025_all_vessels_monthpair-6.parquet",

]

# Region of interest
REGION_LAT_SOUTH = 55
REGION_LAT_NORTH = 82
REGION_LON_WEST = -10
REGION_LON_EAST = 45

# Grid size in degrees
# 0.05 degrees is roughly 5.5 km in latitude.
# Longitude distance varies with latitude.
GRID_SIZE = 0.05

# Maximum allowed time gap between AIS messages.
# Larger gaps are ignored, since we do not know what happened between messages.
MAX_GAP_HOURS = 6

# If True, only count time when the vessel remains in the same grid cell
# between two consecutive AIS messages.
# If False, assign the time interval to the grid cell of the first message.
SAME_CELL_ONLY = False


def only_foreign_vessels(df):
    print("Unique mmsis before dropping norwegian vessels ", df["mmsi"].nunique())
    df["mmsi"] = df["mmsi"].astype(str)
    df["mmsi"] = df["mmsi"].str.strip()

    df["landcode"] = df["mmsi"].str.slice(stop=3)
    df["landcode"] = pd.to_numeric(df["mmsi"].str.slice(stop=3), errors="coerce")

    df_only_foreign = df[~df["landcode"].isin([257, 258, 259])].reset_index(drop=True) # drop all norwegian vessels
    print("Unique mmsis after dropping norwegian vessels ", df_only_foreign["mmsi"].nunique())

    return df_only_foreign

def only_russian_vessels(df):
    print("Unique mmsis before extracting russian vessels ", df["mmsi"].nunique())
    df["mmsi"] = df["mmsi"].astype(str)
    df["mmsi"] = df["mmsi"].str.strip()

    df["landcode"] = df["mmsi"].str.slice(stop=3)
    df["landcode"] = pd.to_numeric(df["mmsi"].str.slice(stop=3), errors="coerce")

    df_only_russian = df[df["landcode"] == 273].reset_index(drop=True) # drop all norwegian vessels
    print("Unique mmsis before extracting russian vessels ", df_only_russian["mmsi"].nunique())
    
    return df_only_russian


FOREIGN = True
# ------------------------------------------------------------
# Read fishing predictions
# ------------------------------------------------------------
dfs = []

for f in files:
    table = pq.read_table(
        f,
        columns=["mmsi", "date_time_utc", "lon", "lat", "pred_fishing"],
        filters=[("pred_fishing", "==", 1)]
    )
    df_part = table.to_pandas()

    if FOREIGN:
        df_part = only_foreign_vessels(df_part)
    
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"], utc=True)

# Keep only valid positions inside region
df = df[
    (df["lat"] >= REGION_LAT_SOUTH) &
    (df["lat"] < REGION_LAT_NORTH) &
    (df["lon"] >= REGION_LON_WEST) &
    (df["lon"] < REGION_LON_EAST)
].copy()

df = df.dropna(subset=["mmsi", "date_time_utc", "lon", "lat"])

print(f"Fishing AIS messages inside region: {len(df):,}")
print(f"Unique vessels: {df['mmsi'].nunique():,}")


# ------------------------------------------------------------
# Assign each message to a grid cell
# ------------------------------------------------------------
df["lon_bin"] = np.floor((df["lon"] - REGION_LON_WEST) / GRID_SIZE).astype(int)
df["lat_bin"] = np.floor((df["lat"] - REGION_LAT_SOUTH) / GRID_SIZE).astype(int)

# Grid-cell lower-left corner
df["lon_min"] = REGION_LON_WEST + df["lon_bin"] * GRID_SIZE
df["lat_min"] = REGION_LAT_SOUTH + df["lat_bin"] * GRID_SIZE

# Grid-cell center, useful for plotting with scatter/pcolormesh
df["lon_center"] = df["lon_min"] + GRID_SIZE / 2
df["lat_center"] = df["lat_min"] + GRID_SIZE / 2


# ------------------------------------------------------------
# Compute fishing time per vessel per grid cell
# ------------------------------------------------------------
df = df.sort_values(["mmsi", "date_time_utc"]).copy()

# Time until next message from same vessel
df["next_time"] = df.groupby("mmsi")["date_time_utc"].shift(-1)
df["next_lon_bin"] = df.groupby("mmsi")["lon_bin"].shift(-1)
df["next_lat_bin"] = df.groupby("mmsi")["lat_bin"].shift(-1)

df["dt_hours"] = (
    df["next_time"] - df["date_time_utc"]
).dt.total_seconds() / 3600

# Remove invalid or too-large gaps
df = df[
    (df["dt_hours"] > 0) &
    (df["dt_hours"] <= MAX_GAP_HOURS)
].copy()

# Optional: only count intervals where the vessel stays in the same grid cell
if SAME_CELL_ONLY:
    df = df[
        (df["lon_bin"] == df["next_lon_bin"]) &
        (df["lat_bin"] == df["next_lat_bin"])
    ].copy()

print(f"Valid fishing intervals: {len(df):,}")
print(f"Total estimated fishing hours: {df['dt_hours'].sum():,.1f}")


# ------------------------------------------------------------
# Aggregate total fishing hours per grid cell
# ------------------------------------------------------------
effort = (
    df.groupby(["lat_bin", "lon_bin", "lat_center", "lon_center"], as_index=False)
      .agg(
          fishing_hours=("dt_hours", "sum"),
          n_messages=("dt_hours", "size"),
          n_vessels=("mmsi", "nunique")
      )
)

print(effort.sort_values("fishing_hours", ascending=False).head())


# ------------------------------------------------------------
# Optional: save aggregated effort
# ------------------------------------------------------------
output_path = Path("predictions_all_2025/fishing_effort_grid_hours_pred_foreign_2025.parquet")
effort.to_parquet(output_path, index=False)

print(f"Saved grid effort to: {output_path}")