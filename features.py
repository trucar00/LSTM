import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

df = pd.read_csv(
    "ais_with_ers_labels_01.csv",
    usecols=["mmsi", "trajectory_id", "date_time_utc", "lon", "lat", "speed", "cog", "label"],
    low_memory=False
)

df = df.fillna(value={"label": "no_fishing"})

# Include all fishing as FISHING
gears = ["Trål", "Snurrevad", "Garn", "Not", "Krokredskap"]
for gear in gears:
    df.loc[df["label"] == gear, "label"] = "fishing"

# Build features

def add_features(df):
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

    # Binary label
    df["y"] = (df["label"] == "fishing").astype(np.int8)

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
    for col in SMOOTH_COLS:
        df[f"ra_{col}"] = (
            df.groupby("trajectory_id")[col]
              .transform(lambda x: x.rolling(window=WINDOW, center=True, min_periods=1).mean())
        )

    return df

def remove_trajectories_w_low_avg_speed(df):
    MIN_AVG_SPEED_KNOTS = 1

    avg_speed = df.groupby("trajectory_id")["speed"].mean()

    stationary_traj_ids = avg_speed[avg_speed < MIN_AVG_SPEED_KNOTS].index

    df = df[~df["trajectory_id"].isin(stationary_traj_ids)].copy()

    print(f"Removed {len(stationary_traj_ids)} trajectories with avg speed < {MIN_AVG_SPEED_KNOTS} knots")
    print(f"Remaining trajectories: {df['trajectory_id'].nunique()}")
    
    return df

df = remove_trajectories_w_low_avg_speed(df)
df = add_features(df)

counts = df["label"].value_counts().reset_index()
counts.columns = ["gear", "nr_messages"]
print(counts)

print(df[FEATURES].isna().sum())
print(np.isinf(df[FEATURES]).sum())
print(df["y"].value_counts(dropna=False))

print(df[FEATURES].describe().T[["mean", "std", "min", "max"]])
print(df[FEATURES].abs().max().sort_values(ascending=False))

df.to_parquet("ais_labeled_features.csv", index=False)

""" for traj_id, d in df.groupby("trajectory_id"):
    d = d.sort_values("date_time_utc")
    t = d["date_time_utc"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Trajectory {traj_id}", fontsize=13)

    for ax, col in zip(axes, ["accel", "jerk", "dcog"]):

        ax.plot(t, d[col], alpha=0.3, linewidth=1, label="raw")
        ax.plot(t, d[f"ra_{col}"], linewidth=1, label=f"MA({5})")
        ax.set_ylabel(col)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linewidth=0.4)
    
    plt.show() """


