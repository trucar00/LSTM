import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_parquet("may_predictions_bilstm_w_dist.parquet")

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
    size=int(0.25 * len(traj_ids)),
    replace=False
)

# Filter dataframe
df_sample = df[df["trajectory_id"].isin(sampled_traj_ids)]

plt.figure(figsize=(10, 10))

fishing = df_sample[df_sample["pred_fishing"] == 1]
non_fishing = df_sample[df_sample["pred_fishing"] == 0]

plt.scatter(non_fishing["lon"], non_fishing["lat"],
            s=2, color="blue", alpha=0.5, label="Non-fishing")

plt.scatter(fishing["lon"], fishing["lat"],
            s=2, color="red", alpha=0.5, label="Fishing")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("25% of Trajectories — May Predictions")
plt.legend()
plt.show()