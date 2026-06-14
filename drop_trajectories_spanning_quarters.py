import pandas as pd

df = pd.read_parquet("2024_1_3.parquet", engine="pyarrow")

print(len(df))

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

print(df["trajectory_id"].nunique())

print(df["date_time_utc"].min(), df["date_time_utc"].max())

df_late_march = df[df["date_time_utc"].between("2024-3-31 23:00:00", "2024-4-1 00:00:00")]

print(len(df_late_march))
print(df_late_march["trajectory_id"].nunique())