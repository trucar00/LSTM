import pandas as pd
import numpy as np

GEAR_TYPE_FILE = ["Not", "Krokredskap", "Traps", "Garn", "Trål", "Snurrevad"]

GEAR_DICT = {
    "Not": set(),
    "Krokredskap": set(),
    "Garn": set(),
    "Trål": set(),
    "Traps": set(),
    "Snurrevad": set()
}

TOTAL_MMSIS = set()


def mmsis_per_gear_type(df, gear, year, month1, month2):
    mmsis = set(df["mmsi"].dropna().unique())
    print(f"Nr of mmsis registered for {gear} in {year} between month {month1} and month {month2}: {len(mmsis)}")
    
    return mmsis

def print_total_mmsis_per_gear(gear_dict):
    for key, values in gear_dict.items():
        print(key, len(values))
    return 0

def main():
    for year in range(2023, 2025+1):
        print("CHECKING GEAR COUNT FOR ", year)
        for i in range(1, 12+1, 3):

            for gear in GEAR_TYPE_FILE:
                df = pd.read_parquet(f"../../Label-ais-ers/Master-prework/label_ais_pts_w_ers/confident_new_rule_new_duration/{gear}_{year}_{i}_{i+2}.parquet", engine="pyarrow")
                mmsis_gear = mmsis_per_gear_type(df, gear, year, month1=i, month2=i+2)
                GEAR_DICT[gear].update(mmsis_gear)
                TOTAL_MMSIS.update(mmsis_gear)

        print("PER GEAR: ")
        print_total_mmsis_per_gear(GEAR_DICT)
        print(f"TOTAL MMSIS: {len(TOTAL_MMSIS)}")

    return 0

if __name__ == "__main__":
    main()
