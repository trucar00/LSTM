import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

WINDOW = 256
STRIDE = 128

FEATURES = [
    "cog_sin", "cog_cos", "speed_calc_ms", "ra_accel",
    "ra_jerk", "log_dist", "ra_dcog", "log_dt", "dist_to_shore_km"
]

MODEL_PATH = "models/bilstm_best_10e_all_gear_01_04_dist.pt"

# --------------------------------------------------
# Same model class as training
# --------------------------------------------------
class FishingBiLSTM(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.3):
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
            nn.Linear(2 * hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        logits = self.head(h).squeeze(-1)
        return logits


# --------------------------------------------------
# Recompute the same normalization stats from Jan-Apr
# Better: save mu/sigma during training and load them.
# --------------------------------------------------
df_train_period = pd.read_parquet("ais_conf_labeled_features_01_04_all_gear.parquet")

# Split into train test and validation set by mmsi so that no vessel appear in both.
rng = np.random.default_rng(42)
mmsis = df_train_period["mmsi"].unique().copy()
rng.shuffle(mmsis)
n = len(mmsis)
train_mmsi = set(mmsis[: int(0.70 * n)])

# Fit normalization on TRAIN ONLY
train_df = df_train_period[df_train_period["mmsi"].isin(train_mmsi)]

mu    = train_df[FEATURES].mean()
sigma = train_df[FEATURES].std().replace(0, 1)

# --------------------------------------------------
# Load May data with already-built features
# If not built yet: run df_may = add_features(raw_may_df)
# --------------------------------------------------
df_may = pd.read_parquet("ais_conf_labeled_features_05_all_gear.parquet")
df_may["date_time_utc"] = pd.to_datetime(df_may["date_time_utc"])

# Normalize exactly like training
for col in FEATURES:
    df_may[col] = (df_may[col] - mu[col]) / sigma[col]

# Same clipping as training
df_may["ra_accel"] = df_may["ra_accel"].clip(-5, 5)
df_may["ra_jerk"]  = df_may["ra_jerk"].clip(-5, 5)
df_may["ra_dcog"]  = df_may["ra_dcog"].clip(-5, 5)


# --------------------------------------------------
# Load model
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FishingBiLSTM(
    n_features=len(FEATURES),
    hidden=128,
    n_layers=2,
    dropout=0.3,
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --------------------------------------------------
# Predict May and merge overlapping window predictions
# --------------------------------------------------
df_may = df_may.sort_values(["trajectory_id", "date_time_utc"]).copy()
df_may["pred_sum"] = 0.0
df_may["pred_count"] = 0.0

with torch.no_grad():
    for traj_id, traj in tqdm(df_may.groupby("trajectory_id", sort=False)):
        idx = traj.index.to_numpy()
        X_all = traj[FEATURES].to_numpy(dtype=np.float32)

        n = len(traj)
        if n < 8:
            continue

        for start in range(0, max(1, n - WINDOW + 1), STRIDE):
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

            df_may.loc[valid_idx, "pred_sum"] += valid_probs
            df_may.loc[valid_idx, "pred_count"] += 1

df_may["p_fishing"] = df_may["pred_sum"] / df_may["pred_count"]
df_may["pred_fishing"] = (df_may["p_fishing"] > 0.5).astype(int)

df_may = df_may.drop(columns=["pred_sum", "pred_count"])

df_may.to_parquet("may_predictions_bilstm_w_dist.parquet", index=False)

print(df_may[["trajectory_id", "date_time_utc", "mmsi", "p_fishing", "pred_fishing"]].head())
print(df_may["pred_fishing"].value_counts())