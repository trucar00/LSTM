import pandas as pd

""" df = pd.read_parquet("line_trawl_purse_conf_negs_ais.parquet")
df.loc[df["conf_no_fishing"], "report"] = "conf_no_fishing"
df.loc[df["unknown_no_fishing"], "report"] = "unknown"
df = df.drop(columns=["conf_no_fishing", "passed_any_rule", "unknown_no_fishing", "close_to_shore", "high_speed", "row_id", "no_fish_cl"])
print(df.columns)

df.to_parquet("ais_conf_negs_w_reports.parquet", index=False) """

df = pd.read_parquet("ais_conf_labeled_features_01_04_all_gear.parquet")
print(len(df))
df = df.drop(columns=["row_id", "high_speed", "no_fish_cl", "close_to_shore", "dist_to_shore_km", "passed_any_rule", "conf_no_fishing", "unknown_no_fishing"])
print(df["report"].unique())
print(df.head())
print(df.columns)
#df.to_parquet("upload_me.parquet", index=False)