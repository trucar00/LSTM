import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gc
import rasterio

# -- HELPER FUNCTIONS --
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of the earth in meters

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1


    # apply formulae
    a = (pow(np.sin(dlat / 2), 2) +  
             np.cos(lat1) * np.cos(lat2) * pow(np.sin(dlon / 2), 2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c

    return dist

def angle_wrap(a):
    return (a + 180) % 360 - 180
# ---------------------------

FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "accel", "ra_accel", "jerk", "ra_jerk", "dcog", "ra_dcog", "log_dist", "log_dt"]
GEAR = ["Trål", "Not", "Krokredskap", "Snurrevad", "Garn", "Traps"]

KEEP_COLS = ["mmsi", "trajectory_id", "date_time_utc", "lon", "lat"] + FEATURES


def column_fixing(df):
    df["gear_report"] = df["label"]
    #df.loc[df["conf_no_fishing"], "label"] = "conf_no_fishing"
    #df.loc[df["unknown_no_fishing"], "label"] = "unknown"

    df = df.drop(columns=["row_id", "high_speed", "no_fish_cl", "close_to_shore", "passed_any_rule", "conf_no_fishing", "unknown_no_fishing"], errors="ignore")

    counts = df["label"].value_counts()
    print(counts)

    # Include all fishing as FISHING
   
    for gear in GEAR:
        df.loc[df["label"] == gear, "label"] = "fishing"

    print(df["label"].unique())
    return df

def add_features(df, online):
    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"]).copy()

    g = df.groupby("trajectory_id", sort=False)

    # Previous values within each trajectory
    df["prev_time"] = g["date_time_utc"].shift(1)
    df["prev_lat"] = g["lat"].shift(1)
    df["prev_lon"] = g["lon"].shift(1)
    df["prev_cog"] = g["cog"].shift(1)

    # Time delta
    df["dt"] = (df["date_time_utc"] - df["prev_time"]).dt.total_seconds()

    # Distance to previous point
    df["dist_to_prev"] = haversine(
        df["prev_lat"].to_numpy(),
        df["prev_lon"].to_numpy(),
        df["lat"].to_numpy(),
        df["lon"].to_numpy()
    )

    # Log-transform heavy-tailed features
    df["log_dt"]       = np.log1p(df["dt"].clip(lower=0))
    df["log_dist"]     = np.log1p(df["dist_to_prev"].clip(lower=0))

    # Encode COG as sin/cos so the 0/360 discontinuity doesn't confuse the model
    df["cog_sin"] = np.sin(np.radians(df["cog"]))
    df["cog_cos"] = np.cos(np.radians(df["cog"]))

    # Calculated speed in m/s
    df["speed_calc_ms"] = df["dist_to_prev"] / df["dt"]

    # Acceleration
    df["prev_speed_calc_ms"] = g["speed_calc_ms"].shift(1)
    df["accel"] = (df["speed_calc_ms"] - df["prev_speed_calc_ms"]) / df["dt"]

    # Jerk
    df["prev_accel"] = g["accel"].shift(1)
    df["jerk"] = (df["accel"] - df["prev_accel"]) / df["dt"]

    # Angular rate of course change (deg/s)
    df["dcog"] = angle_wrap(df["cog"] - df["prev_cog"]) / df["dt"]

    # Remove invalid rows
    feature_cols = ["dt", "dist_to_prev", "speed_calc_ms", "accel", "jerk", "dcog"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols).copy()

    # Optional cleanup of helper columns
    df = df.drop(columns=[
        "prev_time", "prev_lat", "prev_lon", "prev_cog",
        "prev_speed_calc_ms", "prev_accel"
    ])

    # Smooth noisy derivative features
    SMOOTH_COLS = ["accel", "jerk", "dcog"]
    WINDOW = 5
    if online:
        for col in SMOOTH_COLS:
            df[f"ra_{col}"] = (
                df.groupby("trajectory_id")[col]
                .transform(lambda x: x.rolling(window=WINDOW, center=False, min_periods=1).mean())
            )
    else:

        for col in SMOOTH_COLS:
            df[f"ra_{col}"] = (
                df.groupby("trajectory_id")[col]
                .transform(lambda x: x.rolling(window=WINDOW, center=True, min_periods=1).mean())
            )

    return df

#df = add_features(df)

def check_feats(df):
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "nr_messages"]
    print(counts)

    print(df[FEATURES].isna().sum())
    print(np.isinf(df[FEATURES]).sum())

    print(df[FEATURES].describe().T[["mean", "std", "min", "max"]])
    print(df[FEATURES].abs().max().sort_values(ascending=False))
    return


def check_speed(df):
    row = df.loc[df["speed_calc_ms"].idxmax()]
    print(row)
    traj_id = row["trajectory_id"]
    time = row["date_time_utc"]

    prev_row = df[
        (df["trajectory_id"] == traj_id) &
        (df["date_time_utc"] < time)
    ].sort_values("date_time_utc").iloc[-1]

    print("CURRENT:\n", row[["trajectory_id", "lat","lon","date_time_utc","dt","dist_to_prev","speed_calc_ms"]])
    print("\nPREVIOUS:\n", prev_row[["lat","lon","date_time_utc"]])

    dist = haversine(
        prev_row["lat"], prev_row["lon"],
        row["lat"], row["lon"]
    )
    print("Distance (m):", dist)
    print("dt (s):", row["dt"])
    print("speed (m/s):", dist / row["dt"])


def concat2():
    for i in range(1, 12+1, 2):
        dfs = []
        for j in range(i, i+2):
            df = pd.read_parquet(f"../../Test/IDUN/Processed_AIS_2025/Cleaned_pq_new/{j:02d}.parquet", engine="pyarrow")
            dfs.append(df)

        all_vessels_three_months = pd.concat(dfs, ignore_index=True)
        all_vessels_three_months.to_parquet(f"three_months/all_vessels_2025/all_vessels_2025_{i}_{i+1}_no_label.parquet", index=False)

def main(online):
    for year in range(2025, 2025+1):
        for i in range(1, 12+1, 3):
            df = pd.read_parquet(f"three_months/all_vessels_2025/all_vessels_{year}_{i}_{i+2}_no_label.parquet", engine="pyarrow")
            print("Nr of unique trajs: ", df["trajectory_id"].nunique())
            print(df.shape)
            #df = column_fixing(df)
            df = add_features(df, online)
            print(df.shape)
            df = df[KEEP_COLS]
            df.to_parquet(f"three_months/all_vessels_2025/all_vessels_{year}_{i}_{i+2}_feats.parquet", index=False)
            del df
            gc.collect()

def russian(online):
    df = pd.read_parquet("Data/russian_svalbard_trawler_cleaned.parquet", engine="pyarrow")
    df = add_features(df, online)
    df.to_parquet("russian_svalbard_trawler_feats_online.parquet", index=False)
    print(df.shape)

if __name__ == "__main__":
    concat2()
    main(online=True)
    #russian(online=True)
    #concat()
    #concat2()
