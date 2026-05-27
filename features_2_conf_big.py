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
GEAR = ["Trål", "Not", "Krokredskap", "Snurrevad", "Garn", "Traps"]


def column_fixing(df):
    df["gear_report"] = df["report"]
    df.loc[df["conf_no_fishing"], "report"] = "conf_no_fishing"
    df.loc[df["unknown_no_fishing"], "report"] = "unknown"

    df = df.drop(columns=["row_id", "high_speed", "no_fish_cl", "close_to_shore", "passed_any_rule", "conf_no_fishing", "unknown_no_fishing"])

    counts = df["report"].value_counts()
    print(counts)

    # Include all fishing as FISHING
   
    for gear in GEAR:
        df.loc[df["report"] == gear, "report"] = "fishing"

    print(df["report"].unique())
    return df

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
    df["y"] = np.nan
    df.loc[df["report"] == "fishing", "y"] = 1
    df.loc[df["report"] == "conf_no_fishing", "y"] = 0
    
    # Sample weight, unknown = 0
    df["sample_weight"] = df["y"].notna().astype(np.float32)
    df["y_train"] = df["y"].fillna(0).astype(np.float32) # replacing NaN with 0, now the unknowns have y_train = 0 and sample weight = 0

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
    feature_cols = ["dt", "dist_to_prev", "speed_calc_ms", "accel", "jerk", "dcog", "dist_to_shore_km"]
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

#df = add_features(df)

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


#df.to_parquet("ais_all_msgs_labeled_features_05_all_gear.parquet", index=False)


def concat():
    for i in range(1, 12+1, 3):
        
        for year in range(2023, 2025+1):
            dfs = []
            for gear in GEAR:
                df = pd.read_parquet(f"../../Label-ais-ers/Master-prework/label_ais_pts_w_ers/confident2/{gear}_{year}_{i}_{i+2}.parquet", engine="pyarrow")
                dfs.append(df)
        
            all_gear_full_month_df = pd.concat(dfs, ignore_index=True)
            all_gear_full_month_df.to_parquet(f"three_months/all_gear2/{year}_{i}_{i+2}.parquet", index=False)

def concat2():
    for i in range(1, 12+1, 3):
        dfs = []
        for j in range(i, i+3):
            df = pd.read_parquet(f"../../Label-ais-ers/Master-prework/label_ais_pts_w_ers/ais_ers_labels_all_{j}_2025.parquet", engine="pyarrow")
            dfs.append(df)

        all_vessels_three_months = pd.concat(dfs, ignore_index=True)
        all_vessels_three_months.to_parquet(f"three_months/all_vessels_2025/all_vessels_2025_{i}_{i+2}.parquet", index=False)

def main():
    for year in range(2023, 2025+1):
        for i in range(1, 12+1, 3):
            df = pd.read_parquet(f"three_months/all_gear2/{year}_{i}_{i+2}.parquet", engine="pyarrow")
            print("Fixing columns")
            df = column_fixing(df)
            df = add_features(df)
            check_feats(df)
            df.to_parquet(f"three_months/feats_all_gear2/{year}_{i}_{i+2}_feats.parquet", index=False)


if __name__ == "__main__":
    #main()
    #concat()
    concat2()
