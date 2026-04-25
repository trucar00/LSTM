import pandas as pd
import matplotlib.pyplot as plt

""" first_50_mmsis = df["mmsi"].drop_duplicates().head(10)

df = df[df["mmsi"].isin(first_50_mmsis)] """

pred_df = pd.read_csv("test_predictions.csv")

for traj_id, d in pred_df.groupby("trajectory_id"):

    plt.figure(figsize=(10, 6))

    colors = d["pred_fishing"].map({0: "blue", 1: "red"})

    plt.scatter(d["lon"], d["lat"],
                c=colors,
                s=10, alpha=0.7)

    plt.title("Trajectory with fishing predictions")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()