import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from pathlib import Path
import json

USE_TUNED_PARAMS = True
tuned_params_path = Path(f"tuning_FINAL/best_params_LSTM_tune-2023-no-val-test.json")

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
    exit()

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt"]

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES

MODEL_PATH = "training_FULL_MODEL/model_lstm_train_2023_and_2024_FULL_FINAL.pt"

# --------------------------------------------------
# Same model class as training
# --------------------------------------------------
class FishingLSTM(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, n_layers=N_LAYERS,
                 dropout=DROPOUT, dense=DENSE):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        logits = self.head(h).squeeze(-1)
        return logits


# --------------------------------------------------
# Recompute the same normalization stats from 2023 and 2024
# Better: save mu/sigma during training and load them.
# --------------------------------------------------

mu_sigma_path = Path(f"training_FULL_MODEL/parameters_lstm_train_2023_and_2024.pkl")
with open(mu_sigma_path, "rb") as f:
    params = pickle.load(f)

mu = params["mu"]
sigma = params["sigma"]
print("Read mu and sigma from file")

# --------------------------------------------------
# Load data with already-built features
# If not built yet: run df_predict = add_features(raw_may_df)
# --------------------------------------------------

df_predict = pd.read_parquet("three_months/feats_new_rule_online/2025_7_9_feats.parquet")
#df_predict = pd.read_parquet("other_preds/russian_svalbard_trawler_feats.parquet")
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

model = FishingLSTM(
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

# ---------- 6. ONLINE sliding-window inference ----------
#
# For each message at time t we feed the model the window x_{t-W+1} ... x_t
# (zero-padded on the LEFT for the first W-1 messages, which is equivalent to
# starting from a cold LSTM state — exactly what the model saw at training time
# at the start of each window). We take only the prediction at the LAST position.
#
# This means every message gets exactly ONE prediction, using only past info.
# No averaging over overlapping windows.

INFER_BATCH = 256   # how many "ending-at-t" windows to forward in one pass

df_predict = df_predict.sort_values(["trajectory_id", "date_time_utc"]).copy()
df_predict["p_fishing"] = np.nan

with torch.no_grad():
    for traj_id, traj in df_predict.groupby("trajectory_id", sort=False):
        idx = traj.index.to_numpy()
        X_all = traj[FEATURES].to_numpy(dtype=np.float32)
        n, F = X_all.shape
        if n < 1:
            continue

        # --- Cold-start phase: positions t = 0 .. min(WINDOW, n) - 1 ---
        # A single forward pass on X_all[:WINDOW] gives the causal predictions
        # at all those positions (because the LSTM is unidirectional, the
        # output at position k uses only inputs 0..k — equivalent to feeding
        # sequences of length 1, 2, ..., WINDOW from a zero state).
        head_len = min(WINDOW, n)
        x_head = torch.from_numpy(X_all[:head_len][None, :, :]).to(device)
        probs_head = torch.sigmoid(model(x_head))[0].cpu().numpy()  # (head_len,)

        traj_probs = np.empty(n, dtype=np.float32)
        traj_probs[:head_len] = probs_head

        # --- Steady-state phase: positions t = WINDOW .. n-1 ---
        # For each such t, window is X_all[t-WINDOW+1 : t+1], take last pred.
        if n > WINDOW:
            # Build all sliding windows in one go using stride tricks
            sw = np.lib.stride_tricks.sliding_window_view(
                X_all, window_shape=WINDOW, axis=0
            )                                  # shape: (n - WINDOW + 1, F, WINDOW)
            sw = sw.transpose(0, 2, 1)         # -> (n - WINDOW + 1, WINDOW, F)
            sw = sw[1:]                        # drop window ending at WINDOW-1
                                               #   (already covered by head)
            # Now sw[k] is the window ending at position WINDOW + k.

            for i in range(0, len(sw), INFER_BATCH):
                batch_np = np.ascontiguousarray(sw[i:i+INFER_BATCH])
                batch = torch.from_numpy(batch_np).to(device)
                logits = model(batch)          # (B, WINDOW)
                last_probs = torch.sigmoid(logits[:, -1]).cpu().numpy()
                t_start = WINDOW + i
                traj_probs[t_start : t_start + len(last_probs)] = last_probs

        df_predict.loc[idx, "p_fishing"] = traj_probs

df_predict["pred_fishing"] = (df_predict["p_fishing"] > 0.5).astype(int)
df_predict.to_parquet("predictions_seen/2025_7_9_w_lstm_full_model.parquet", index=False)

print(df_predict[["trajectory_id", "date_time_utc", "mmsi",
                  "p_fishing", "pred_fishing"]].head())
print(df_predict["pred_fishing"].value_counts())