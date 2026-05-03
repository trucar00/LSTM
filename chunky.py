import pandas as pd
import math
import glob

file = "ais_conf_labeled_features_01_04_all_gear.parquet"
chunk_size = 20_000_000  # rows per chunk

df = pd.read_parquet(file)
n_chunks = math.ceil(len(df) / chunk_size)

for i in range(n_chunks):
    chunk = df.iloc[i*chunk_size:(i+1)*chunk_size]
    chunk.to_parquet(f"chunks/chunk_{i:03d}.parquet")

print("Done splitting")



""" files = sorted(glob.glob("chunks/chunk_*.parquet"))
dfs = [pd.read_parquet(f) for f in files]

df = pd.concat(dfs, ignore_index=True)
df.to_parquet("ais_conf_labeled_features_01_04_all_gear.parquet")

print("Recombined!") """