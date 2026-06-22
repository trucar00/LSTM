import pandas as pd
import numpy as np
from collections import defaultdict

# ============================================================
# Files
# ============================================================

files = [
    "training_FINAL/test_vessels_2024/BiLSTM_2024_seen_test_seed0.parquet",
]

columns = [
    "mmsi",
    "trajectory_id",
    "date_time_utc",
    "report",
    "gear_report",
    "pred_fishing",
]

gears = [
    "Bur og ruser",
    "Krokredskap",
    "Trål",
    "Not",
    "Snurrevad",
    "Garn",
]


# ============================================================
# Count-based metric functions
# ============================================================

def compute_counts(df):
    pred_fishing = df["pred_fishing"].to_numpy().astype(bool)

    report = df["report"].to_numpy()
    rep_fish = report == "fishing"
    rep_conf = report == "conf_no_fishing"
    rep_unknown = report == "unknown"

    return {
        "n_rows": len(df),

        "tp": int(np.sum(pred_fishing & rep_fish)),
        "fp": int(np.sum(pred_fishing & rep_conf)),
        "tn": int(np.sum(~pred_fishing & rep_conf)),
        "fn": int(np.sum(~pred_fishing & rep_fish)),

        "n_pred_fish": int(np.sum(pred_fishing)),
        "n_pred_no_fish": int(np.sum(~pred_fishing)),

        "n_reported_fish": int(np.sum(rep_fish)),
        "n_reported_no_fish": int(np.sum(rep_conf)),
        "n_unknowns": int(np.sum(rep_unknown)),

        "n_pred_fish_of_unknown": int(np.sum(pred_fishing & rep_unknown)),
        "n_pred_no_fish_of_unknown": int(np.sum(~pred_fishing & rep_unknown)),
    }


def empty_counts():
    return {
        "n_rows": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "n_pred_fish": 0,
        "n_pred_no_fish": 0,
        "n_reported_fish": 0,
        "n_reported_no_fish": 0,
        "n_unknowns": 0,
        "n_pred_fish_of_unknown": 0,
        "n_pred_no_fish_of_unknown": 0,
        "mmsis": set(),
    }


def add_counts(total, new):
    for key, value in new.items():
        total[key] += value


def finalize_metrics(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

    return {
        "N vessels": len(counts["mmsis"]),
        "N rows": counts["n_rows"],

        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,

        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision,
        "Specificity": specificity,
        "F1": f1,

        "Predicted fishing": counts["n_pred_fish"],
        "Predicted no fishing": counts["n_pred_no_fish"],
        "Reported fishing": counts["n_reported_fish"],
        "Reported no fishing": counts["n_reported_no_fish"],
        "Unknown": counts["n_unknowns"],
        "Predicted fishing of unknown": counts["n_pred_fish_of_unknown"],
        "Predicted no fishing of unknown": counts["n_pred_no_fish_of_unknown"],
    }


# ============================================================
# Your gear-filtering function
# ============================================================

def filter_for_gear_vs_no_fishing(
    df,
    gear_type,
    no_gear="no_fishing",
    gear_col="gear_report",
    time_col="date_time_utc",
):
    allowed_gear = [gear_type, no_gear]

    allowed_mask = df[gear_col].isin(allowed_gear)
    has_gear_mask = df[gear_col].eq(gear_type)

    valid_by_traj = (
        allowed_mask.groupby(df["trajectory_id"]).all()
        &
        has_gear_mask.groupby(df["trajectory_id"]).any()
    )

    valid_ids = valid_by_traj[valid_by_traj].index

    df_out = df[df["trajectory_id"].isin(valid_ids)].copy()
    df_out[time_col] = pd.to_datetime(df_out[time_col])

    df_out = (
        df_out
        .sort_values(["trajectory_id", time_col])
        .reset_index(drop=True)
    )

    df_out["row_id"] = np.arange(len(df_out))

    return df_out


# ============================================================
# Aggregation over all 2025 files
# ============================================================

overall_counts = empty_counts()
gear_counts = {gear: empty_counts() for gear in gears}

for file in files:
    print(f"\nProcessing {file}")

    df_predict = pd.read_parquet(
        file,
        engine="pyarrow",
        columns=columns,
    )

    print("Rows:", len(df_predict))
    print("Vessels:", df_predict["mmsi"].nunique())

    # --------------------------
    # Overall counts
    # --------------------------

    file_overall_counts = compute_counts(df_predict)
    add_counts(overall_counts, file_overall_counts)
    overall_counts["mmsis"].update(df_predict["mmsi"].dropna().unique())

    # --------------------------
    # Gear-specific counts
    # --------------------------

    for gear in gears:
        gear_df = filter_for_gear_vs_no_fishing(
            df_predict,
            gear_type=gear,
        )

        file_gear_counts = compute_counts(gear_df)
        add_counts(gear_counts[gear], file_gear_counts)
        gear_counts[gear]["mmsis"].update(gear_df["mmsi"].dropna().unique())

        print(
            gear,
            "rows:", len(gear_df),
            "vessels:", gear_df["mmsi"].nunique(),
        )

    del df_predict


# ============================================================
# Final 2025 results
# ============================================================

rows = []

overall_metrics = finalize_metrics(overall_counts)
overall_metrics["Gear type"] = "Overall"
rows.append(overall_metrics)

for gear in gears:
    metrics = finalize_metrics(gear_counts[gear])
    metrics["Gear type"] = gear
    rows.append(metrics)

results = pd.DataFrame(rows)

# Reorder columns
cols_first = [
    "Gear type",
    "N vessels",
    "N rows",
    "Accuracy",
    "Recall",
    "Precision",
    "Specificity",
    "F1",
    "TP",
    "FP",
    "TN",
    "FN",
]

results = results[cols_first + [c for c in results.columns if c not in cols_first]]

# Convert metrics to percentages
metric_cols = ["Accuracy", "Recall", "Precision", "Specificity", "F1"]
results[metric_cols] = results[metric_cols] * 100

print("\nFinal 2024 results:")
print(results)

results.to_csv("training_FINAL/test_vessels_2024/bilstm_2024_gear_specific_statistics_seen_vessels.csv", index=False)