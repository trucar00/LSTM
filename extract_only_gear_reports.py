import pandas as pd
import pyarrow.parquet as pq

GEARS = ["Trål", "Krokredskap", "Bur og ruser", "Not", "Snurrevad", "Garn"]
BASE = "three_months/feats_new_rule_online"
SAVE = "three_months/only_gear_reports"
FILES = [
    f"{BASE}/2023_1_3_feats.parquet",     # Q1 2023
    f"{BASE}/2023_4_6_feats.parquet",     # Q2 2023
    f"{BASE}/2023_7_9_feats.parquet",     # Q3 2023
    f"{BASE}/2023_10_12_feats.parquet",   # Q4 2023
    f"{BASE}/2024_1_3_feats.parquet",     # Q1 2024
    f"{BASE}/2024_4_6_feats.parquet",     # Q2 2024
    f"{BASE}/2024_7_9_feats.parquet",     # Q3 2024
    f"{BASE}/2024_10_12_feats.parquet",   # Q4 2024
    f"{BASE}/2025_1_3_feats.parquet",     # Q1 2025
    f"{BASE}/2025_4_6_feats.parquet",     # Q2 2025
    f"{BASE}/2025_7_9_feats.parquet",     # Q3 2025
    f"{BASE}/2025_10_12_feats.parquet",   # Q4 2025
]


def get_fishing_segments(df, seg_id_end):
    df = df.sort_values(["trajectory_id", "date_time_utc"]).reset_index(drop=True)
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    new_traj  = df["trajectory_id"].ne(df["trajectory_id"].shift())
    gear_flip = df["gear_report"].ne(df["gear_report"].shift())

    df["segment_id"] = ((new_traj | gear_flip ).cumsum()).astype(str) + seg_id_end
    return df[df["gear_report"].isin(GEARS)].copy()

def get_file_name(filepath):
    split_str = filepath.split("/")
    return "onl_" + split_str[2]

def seg_id_ending(filepath):
    split_str = filepath.split("/")
    date = split_str[2].split("_")[0:3]
    date_str = "-" + date[0] + "-" + date[1] + "-" + date[2]
    return date_str


def get_msgs_reported_fishing(files):
    for f in files:
        df = pd.read_parquet(f, engine="pyarrow")
        seg_id_end = seg_id_ending(f)
        df = get_fishing_segments(df, seg_id_end)
        file_name = get_file_name(f)
        save_path = f"{SAVE}/{file_name}"
        df.to_parquet(save_path, index=False)

    return "DONE!"


get_msgs_reported_fishing(FILES)