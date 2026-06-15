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
    print(f"{gear} in {year} between month {month1} and month {month2}: {len(mmsis)}")
    
    return mmsis

def print_total_mmsis_per_gear(gear_dict):
    for key, values in gear_dict.items():
        print(key, len(values))
    return 0

def main():
    for year in range(2024, 2024+1):
        print("CHECKING GEAR COUNT FOR ", year)
        for i in range(1, 12+1, 3):

            for gear in GEAR_TYPE_FILE:
                df = pd.read_parquet(f"../../Label-ais-ers/Master-prework/label_ais_pts_w_ers/confident_new_rule_new_duration/{gear}_{year}_{i}_{i+2}.parquet", engine="pyarrow")
                mmsis_gear = mmsis_per_gear_type(df, gear, year, month1=i, month2=i+2)
                GEAR_DICT[gear].update(mmsis_gear)
                TOTAL_MMSIS.update(mmsis_gear)

        print("PER GEAR: ")
        print_total_mmsis_per_gear(GEAR_DICT)
        print(f"TOTAL MMSIS in {year}: {len(TOTAL_MMSIS)}")

    mmsis = np.array(list(TOTAL_MMSIS))
    split_rng = np.random.default_rng(10)
    split_rng.shuffle(mmsis)
    n = len(mmsis)
    train_mmsi = set(mmsis[:int(0.70 * n)])
    val_mmsi   = set(mmsis[int(0.70 * n):int(0.85 * n)])
    test_mmsi  = set(mmsis[int(0.85 * n):])

    for key, values in GEAR_DICT.items():
        train_mmsis_in_gear = values.intersection(train_mmsi)
        val_mmsis_in_gear = values.intersection(val_mmsi)
        test_mmsis_in_gear = values.intersection(test_mmsi)

        total = len(values)

        print(f"\n{key}")
        print("-" * 40)
        print(f"Total:      {total}")
        print(f"Train:      {len(train_mmsis_in_gear)}")
        print(f"Validation: {len(val_mmsis_in_gear)}")
        print(f"Test:       {len(test_mmsis_in_gear)}")

    print("-" * 40)
    print("Train total: ", len(train_mmsi))
    print("Val total: ", len(val_mmsi))
    print("Test total: ", len(test_mmsi))

    df_mmsis = pd.DataFrame({
        "mmsi": np.concatenate([
            list(train_mmsi),
            list(val_mmsi),
            list(test_mmsi),
        ]),
        "split": (
            ["train"] * len(train_mmsi)
            + ["validation"] * len(val_mmsi)
            + ["test"] * len(test_mmsi)
        ),
    })

    print(df_mmsis.head())
    df_mmsis.to_csv("../train_val_test_mmsis_FINAL.csv", index=False)
   

if __name__ == "__main__":
    main()
