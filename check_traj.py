import pandas as pd
import matplotlib.pyplot as plt

check = "257770000-0"

df = pd.read_parquet("Data/ais_ers_labels_clean_05_w_dist.parquet")
df_traj = df.loc[df["trajectory_id"] == check].copy()

df_traj["date_time_utc"] = pd.to_datetime(df_traj["date_time_utc"])
df_traj = df_traj.sort_values("date_time_utc")

plt.scatter(df_traj["lon"], df_traj["lat"], s=5)

plt.show()


print(df_traj.tail())