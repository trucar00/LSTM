import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from pathlib import Path
import json

USE_TUNED_PARAMS = True
tuned_params_path = Path(f"tuning/best_params.json")

if USE_TUNED_PARAMS and tuned_params_path.exists():
    
    with open(tuned_params_path, "r") as file:
        best_params = json.load(file)["best_params"]
    print("Loaded tuned params ", best_params)
    WINDOW = best_params["window"]
    STRIDE = best_params["stride"]
    N_LAYERS = best_params["n_layers"]
    HIDDEN = best_params["hidden"]
    DENSE = best_params["dense"]
    DROPOUT = best_params["dropout"]
    BATCH = best_params["batch"]
    LR = best_params["lr"]

# Previous base parameters
else:
    print("Using default (non-tuned) params")
    WINDOW = 256
    STRIDE = 128
    HIDDEN = 128
    DROPOUT = 0.3
    N_LAYERS = 2
    DENSE = 64
    BATCH = 128
    LR = 1e-4

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt", "dist_to_shore_km"]

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES

MODEL_PATH = "models/model_full_all_gear2_tuned_2023_2024.pt"

# --------------------------------------------------
# Same model class as training
# --------------------------------------------------
class FishingBiLSTM(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT, dense=DENSE):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        logits = self.head(h).squeeze(-1)
        return logits


# --------------------------------------------------
# Recompute the same normalization stats from Jan-Apr
# Better: save mu/sigma during training and load them.
# --------------------------------------------------


with open("parameters_full_all_gear2_2023_2024.pkl", "rb") as f:
    params = pickle.load(f)

mu = params["mu"]
sigma = params["sigma"]
print("Read mu and sigma from file")

# --------------------------------------------------
# Load data with already-built features
# If not built yet: run df_predict = add_features(raw_may_df)
# --------------------------------------------------

df_predict = pd.read_parquet("three_months/feats_all_gear2/2025_10_12_feats.parquet")
df_predict["date_time_utc"] = pd.to_datetime(df_predict["date_time_utc"])
month = df_predict["date_time_utc"].dt.month

df_predict["month_sin"] = np.sin(2 * np.pi * month / 12)
df_predict["month_cos"] = np.cos(2 * np.pi * month / 12)

# Normalize exactly like training
for col in FEATURES:
    df_predict[col] = (df_predict[col] - mu[col]) / sigma[col]

# Same clipping as training
df_predict["ra_accel"] = df_predict["ra_accel"].clip(-5, 5)
df_predict["ra_jerk"]  = df_predict["ra_jerk"].clip(-5, 5)
df_predict["ra_dcog"]  = df_predict["ra_dcog"].clip(-5, 5)


# --------------------------------------------------
# Load model
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FishingBiLSTM(
    n_features=len(FEATURES),
    hidden=HIDDEN,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    dense=DENSE,
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --------------------------------------------------
# Predict May and merge overlapping window predictions
# --------------------------------------------------
df_predict = df_predict.sort_values(["trajectory_id", "date_time_utc"]).copy()
df_predict["pred_sum"] = 0.0
df_predict["pred_count"] = 0.0

with torch.no_grad():
    for traj_id, traj in tqdm(df_predict.groupby("trajectory_id", sort=False)):
        idx = traj.index.to_numpy()
        X_all = traj[FEATURES].to_numpy(dtype=np.float32)

        n = len(traj)
        if n < 8:
            continue

        starts = list(range(0, max(1, n - WINDOW + 1), STRIDE))
        final_start = max(0, n - WINDOW)
        if starts[-1] != final_start:
            starts.append(final_start) # Makes sure that the end of each trajectory is included. 

        for start in starts:
            end = start + WINDOW

            x = X_all[start:end]
            L = len(x)

            if L < WINDOW:
                pad = WINDOW - L
                x = np.vstack([
                    x,
                    np.zeros((pad, x.shape[1]), dtype=np.float32)
                ])

            x_tensor = torch.from_numpy(x[None, :, :]).to(device)

            logits = model(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            valid_idx = idx[start:start + L]
            valid_probs = probs[:L]

            df_predict.loc[valid_idx, "pred_sum"] += valid_probs
            df_predict.loc[valid_idx, "pred_count"] += 1

df_predict["p_fishing"] = df_predict["pred_sum"] / df_predict["pred_count"]
df_predict["pred_fishing"] = (df_predict["p_fishing"] > 0.5).astype(int)

df_predict = df_predict.drop(columns=["pred_sum", "pred_count"])

df_predict.to_parquet("predictions/2025_10_12_w_full_2023_2024_model_tuned.parquet", index=False)

print(df_predict[["trajectory_id", "date_time_utc", "mmsi", "p_fishing", "pred_fishing"]].head())
print(df_predict["pred_fishing"].value_counts())