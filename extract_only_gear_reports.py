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


def get_time_name(filepath):
    split_str = filepath.split("/")
    return "onl_" + split_str[2]


def get_msgs_reported_fishing(files):
    for f in files:
        table = pq.read_table(
            f,
            filters=[("gear_report", "in", GEARS)]
        )
        file_name = get_time_name(f)
        save_path = f"{SAVE}/{file_name}"

        df = table.to_pandas()
        
        df.to_parquet(save_path, index=False)

    return "DONE!"


get_msgs_reported_fishing(FILES)