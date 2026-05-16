import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import numpy as np
import geopandas as gpd
import contextily as ctx
from shapely.geometry import box

df = pd.read_parquet("may_predictions_bilstm_w_dist.parquet")
print(df.columns)

fishing_df = df[df["report"] == "fishing"]

n_total = len(fishing_df)
n_correct = (fishing_df["pred_fishing"] == 1).sum()

recall = n_correct / n_total if n_total > 0 else 0

print(f"Fishing points: {n_total}")
print(f"Predicted fishing: {n_correct}")
print(f"Recall: {recall:.3f}")

conf_df = df[df["report"] == "conf_no_fishing"]

n_total = len(conf_df)
n_correct = (conf_df["pred_fishing"] == 0).sum()

accuracy_conf = n_correct / n_total if n_total > 0 else 0

print(f"conf_no_fishing points: {n_total}")
print(f"Predicted non-fishing: {n_correct}")
print(f"Accuracy (conf_no_fishing): {accuracy_conf:.3f}")

""" for traj_id, traj in df.groupby("trajectory_id"):
    plt.figure(figsize=(6, 6))

    fishing = traj[traj["pred_fishing"] == 1]
    non_fishing = traj[traj["pred_fishing"] == 0]

    plt.scatter(non_fishing["lon"], non_fishing["lat"],
                s=5, color="blue", label="Non-fishing")

    plt.scatter(fishing["lon"], fishing["lat"],
                s=5, color="red", label="Fishing")

    plt.title(f"Trajectory {traj_id}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show() """

# Get unique trajectories
traj_ids = df["trajectory_id"].unique()

# Sample 25%
rng = np.random.default_rng(42)
sampled_traj_ids = rng.choice(
    traj_ids,
    size=int(0.1 * len(traj_ids)),
    replace=False
)

# Filter dataframe
df_sample = df[df["trajectory_id"].isin(sampled_traj_ids)]

plt.figure(figsize=(10, 10))

fishing = df_sample[df_sample["pred_fishing"] == 1]
non_fishing = df_sample[df_sample["pred_fishing"] == 0]

plt.scatter(non_fishing["lon"], non_fishing["lat"],
            s=1, color="blue", alpha=0.5, label="Non-fishing")

plt.scatter(fishing["lon"], fishing["lat"],
            s=1, color="red", alpha=0.5, label="Fishing")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("25% of Trajectories — May Predictions")
plt.legend()
plt.show()

# False positives
fp_df = df_sample[
    (df_sample["pred_fishing"] == 1) &
    (df_sample["report"] != "fishing")
].copy()

# True positives
tp_df = df_sample[
    (df_sample["pred_fishing"] == 1) &
    (df_sample["report"] == "fishing")
].copy()

# Trajectories that contain ANY predicted fishing (both TP + FP)
traj_ids = df_sample[
    df_sample["pred_fishing"] == 1
]["trajectory_id"].unique()

# Context: all points from those trajectories
context_df = df_sample[df_sample["trajectory_id"].isin(traj_ids)]

# Plot
plt.figure(figsize=(10, 10))

# Context (all trajectory points)
plt.scatter(
    context_df["lon"],
    context_df["lat"],
    s=1,
    color="blue",
    alpha=0.3,
    label="Trajectory context"
)

# True positives (green)
plt.scatter(
    tp_df["lon"],
    tp_df["lat"],
    s=2,
    color="green",
    alpha=0.7,
    label="True positives"
)

# False positives (red)
plt.scatter(
    fp_df["lon"],
    fp_df["lat"],
    s=2,
    color="red",
    alpha=0.7,
    label="False positives"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted fishing with context (TP=green, FP=red)")
plt.legend()
plt.show()


# HEATMAP
# Only messages predicted as fishing

def heatmap(df):
    fishing_pred = df[df["pred_fishing"] == 1].copy()

    # Lon/lat -> Web Mercator, same projection as contextily
    gdf = gpd.GeoDataFrame(
        fishing_pred,
        geometry=gpd.points_from_xy(fishing_pred["lon"], fishing_pred["lat"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y

    # Fixed full grid in EPSG:3857
    n_bins = 300
    x_edges = np.linspace(gdf["x"].min(), gdf["x"].max(), n_bins + 1)
    y_edges = np.linspace(gdf["y"].min(), gdf["y"].max(), n_bins + 1)

    gdf["x_bin"] = np.searchsorted(x_edges, gdf["x"], side="right") - 1
    gdf["y_bin"] = np.searchsorted(y_edges, gdf["y"], side="right") - 1

    gdf = gdf[
        (gdf["x_bin"] >= 0) & (gdf["x_bin"] < n_bins) &
        (gdf["y_bin"] >= 0) & (gdf["y_bin"] < n_bins)
    ]

    counts = (
        gdf.groupby(["x_bin", "y_bin"])["trajectory_id"]
        .nunique()
        .reset_index(name="count")
    )

    # Important: full fixed-size grid, not compressed unstack output
    Z = np.zeros((n_bins, n_bins), dtype=float)
    Z[counts["x_bin"], counts["y_bin"]] = counts["count"]

    Z = gaussian_filter(Z, sigma=0.5)
    Z = np.ma.masked_where(Z < 0.2, Z)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(x_edges.min(), x_edges.max())
    ax.set_ylim(y_edges.min(), y_edges.max())

    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        zoom=6
    )

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(alpha=0)

    img = ax.imshow(
        Z.T,
        origin="lower",
        extent=[x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()],
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=Z.max()),
        alpha=0.7,
        interpolation="nearest",
        zorder=10
    )

    plt.colorbar(img, ax=ax, label="Number of unique trajectories")
    ax.set_title("Fishing Activity Heatmap over OpenStreetMap")
    ax.set_axis_off()
    plt.show()

heatmap(df)