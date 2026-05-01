import pandas as pd
get_cols = ["mmsi", "trajectory_id", "date_time_utc", "lon", "lat", "dt", "y", "cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk","log_dist", "ra_dcog", "log_dt"]

df = pd.read_csv("ais_labeled_features.csv", usecols=get_cols)

df.to_csv("ais_labeled_features.csv", index=False)