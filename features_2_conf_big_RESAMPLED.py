import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gc

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

FEATURES = ["cog_interp_sin", "cog_interp_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "ra_dcog", "log_dist", "log_dt"]
GEAR = ["Trål", "Not", "Krokredskap", "Snurrevad", "Garn", "Traps", "Bur og ruser"]
CONCAT_GEAR = ["Trål", "Not", "Krokredskap", "Snurrevad", "Garn", "Traps"]


# Build features

def add_features(df, online=False):
    df = df.copy()
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"]).copy()

    g = df.groupby("trajectory_id", sort=False)

    # Previous values within each trajectory
    df["prev_time"] = g["date_time_utc"].shift(1)
    df["prev_lat"] = g["lat"].shift(1)
    df["prev_lon"] = g["lon"].shift(1)
    df["prev_cog_interp"] = g["cog_interp"].shift(1)

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
    df["cog_interp_sin"] = np.sin(np.radians(df["cog_interp"]))
    df["cog_interp_cos"] = np.cos(np.radians(df["cog_interp"]))

    # Calculated speed in m/s
    df["speed_calc_ms"] = df["dist_to_prev"] / df["dt"]

    # Acceleration
    df["prev_speed_calc_ms"] = g["speed_calc_ms"].shift(1)
    df["accel"] = (df["speed_calc_ms"] - df["prev_speed_calc_ms"]) / df["dt"]

    # Jerk
    df["prev_accel"] = g["accel"].shift(1)
    df["jerk"] = (df["accel"] - df["prev_accel"]) / df["dt"]

    # Angular rate of course change (deg/s)
    df["dcog"] = angle_wrap(df["cog_interp"] - df["prev_cog_interp"]) / df["dt"]

    # Remove invalid rows
    feature_cols = ["dt", "dist_to_prev", "speed_calc_ms", "accel", "jerk", "dcog"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols).copy()

    # Optional cleanup of helper columns
    df = df.drop(columns=[
        "prev_time", "prev_lat", "prev_lon", "prev_cog_interp",
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


def check_feats(df):
    counts = df["report"].value_counts().reset_index()
    counts.columns = ["report", "nr_messages"]
    print(counts)

    print(df[FEATURES].isna().sum())
    print(np.isinf(df[FEATURES]).sum())
    #print(df["y"].value_counts(dropna=False))

    print(df[FEATURES].describe().T[["mean", "std", "min", "max"]])
    print(df[FEATURES].abs().max().sort_values(ascending=False))
    return

def get_fishing_segments(df, seg_id_end):
    df = df.sort_values(["trajectory_id", "date_time_utc"]).reset_index(drop=True)
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    new_traj  = df["trajectory_id"].ne(df["trajectory_id"].shift())
    gear_flip = df["report"].ne(df["report"].shift())

    df["segment_id"] = ((new_traj | gear_flip ).astype("int64").cumsum()).astype(str) + "-" + seg_id_end
    return df[df["report"].isin(GEAR)].copy()

def main(online):
    for year in range(2023, 2023+1):
        for i in range(1, 12+1, 3):
            df = pd.read_parquet(f"three_months/resampled/{year}_{i}_{i+2}.parquet", engine="pyarrow")
            print("Fixing columns")
            df = add_features(df, online=online)
            print(df.head())
            check_feats(df)
            seg_id_end = str(year) + "-" + str(i) + "-" + str(i+2)
            df = get_fishing_segments(df, seg_id_end=seg_id_end)
            print(df["report"].unique())
            df.to_parquet(f"three_months/resampled/{year}_{i}_{i+2}_feats.parquet", index=False)
            del df
            gc.collect()
            # CHANFE CHANFE ACCORDING TO WHAT TYPE

if __name__ == "__main__":
    main(online=False)

