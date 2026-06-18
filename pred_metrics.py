import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import numpy as np
import geopandas as gpd
import contextily as ctx
from shapely.geometry import box

def compute_metrics(df, prefix=""):
    pred_fishing = df["pred_fishing"].to_numpy().astype(bool)

    report = df["report"].to_numpy()
    rep_fish = report == "fishing"
    rep_conf = report == "conf_no_fishing"
    rep_unknown = report == "unknown"

    tp = int(np.sum(pred_fishing & rep_fish))
    fp = int(np.sum(pred_fishing & rep_conf))
    tn = int(np.sum(~pred_fishing & rep_conf))
    fn = int(np.sum(~pred_fishing & rep_fish))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

    return {
        f"{prefix}n_rows": len(df),
        f"{prefix}tp": tp,
        f"{prefix}fp": fp,
        f"{prefix}tn": tn,
        f"{prefix}fn": fn,
        f"{prefix}accuracy": accuracy,
        f"{prefix}recall": recall,
        f"{prefix}specificity": specificity,
        f"{prefix}precision": precision,
        f"{prefix}f1": f1,
        f"{prefix}n_pred_fish": int(np.sum(pred_fishing)),
        f"{prefix}n_pred_no_fish": int(np.sum(~pred_fishing)),
        f"{prefix}n_reported_fish": int(np.sum(rep_fish)),
        f"{prefix}n_reported_no_fish": int(np.sum(rep_conf)),
        f"{prefix}n_unknowns": int(np.sum(rep_unknown)),
        f"{prefix}n_pred_fish_of_unknown": int(np.sum(pred_fishing & rep_unknown)),
        f"{prefix}n_pred_no_fish_of_unknown": int(np.sum(~pred_fishing & rep_unknown)),
    }

df_predict = pd.read_parquet("2025_1_3_w_full_2023_2024_model_tuned.parquet")

# Overall metrics
overall_metrics = compute_metrics(df_predict, prefix="ext_")
print(
    "Overall:",
    "a:", overall_metrics["ext_accuracy"],
    "r:", overall_metrics["ext_recall"],
    "f1:", overall_metrics["ext_f1"],
    "p:", overall_metrics["ext_precision"],
    "spec:", overall_metrics["ext_specificity"],
)



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

gears = ["Bur og ruser"] # "Krokredskap", "Trål", "Snurrevad", "Garn", "Not"

for gear in gears:


    gear_df = filter_for_gear_vs_no_fishing(df_predict, gear_type=gear)
    print(gear_df.head())
    print(gear_df.shape)
    print(gear_df["gear_report"].value_counts())
    print(gear_df["pred_fishing"].value_counts())

    metrics_gear = compute_metrics(gear_df, prefix="ext_")
    print(
        gear,
        "a:", metrics_gear["ext_accuracy"],
        "r:", metrics_gear["ext_recall"],
        "f1:", metrics_gear["ext_f1"],
        "p:", metrics_gear["ext_precision"],
        "spec:", metrics_gear["ext_specificity"],
    )

    tp_mask = (
        gear_df["pred_fishing"]
        & (gear_df["gear_report"] == "Bur og ruser")
    )

    fn_mask = (
        ~gear_df["pred_fishing"].astype(bool)
        & (gear_df["gear_report"] == "Bur og ruser")
    )

    fp_mask = (
        gear_df["pred_fishing"]
        & (gear_df["gear_report"] == "no_fishing")
    )


    for traj_id, d in gear_df.groupby("trajectory_id"):
        plt.figure(figsize=(12, 10))

        # Background
        plt.scatter(
            gear_df["lon"],
            gear_df["lat"],
            s=1,
            c="lightgray",
            alpha=0.2,
            label="All points"
        )

        # Actual Bur og ruser
        bur_mask = gear_df["gear_report"] == "Bur og ruser"

        plt.scatter(
            gear_df.loc[bur_mask, "lon"],
            gear_df.loc[bur_mask, "lat"],
            s=3,
            c="hotpink",
            alpha=0.7,
            label="Reported Bur og ruser"
        )

        # Predicted fishing
        pred_mask = gear_df["pred_fishing"].astype(bool)

        plt.scatter(
            gear_df.loc[pred_mask, "lon"],
            gear_df.loc[pred_mask, "lat"],
            s=3,
            c="red",
            alpha=0.7,
            label="Predicted fishing"
        )

        plt.legend()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Bur og ruser trajectories")
        plt.show()