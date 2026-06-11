import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
import os
import pickle
import json
from pathlib import Path


# Setting of parameters

USE_TUNED_PARAMS = True
tuned_params_path = Path(f"tuning/best_params_ALL_GEAR.json")

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


#files = sorted(glob.glob("three_months/feats/*.parquet"))

files = [
    #"three_months/feats_all_w_traps/2023_1_3_feats.parquet",
    #"three_months/feats_all_w_traps/2023_4_6_feats.parquet",
    #"three_months/feats_all_w_traps/2023_7_9_feats.parquet",
    #"three_months/feats_all_w_traps/2023_10_12_feats.parquet",
    "three_months/feats_all_w_traps/2024_1_3_feats.parquet",
    "three_months/feats_all_w_traps/2024_4_6_feats.parquet",
    #"three_months/feats_all_w_traps/2024_7_9_feats.parquet",
    #"three_months/feats_all_w_traps/2024_10_12_feats.parquet"
]

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt"] # removed dist to shore

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES

all_mmsis = set()

for f in files:
    print("reading ", f)
    m = pd.read_parquet(f, columns=["mmsi"])["mmsi"].unique()
    all_mmsis.update(m)

mmsis = np.array(list(all_mmsis))
rng = np.random.default_rng(42)
rng.shuffle(mmsis)

# Split into train test and validation set by mmsi so that no vessel appear in both.
n = len(mmsis)
train_mmsi = set(mmsis[:int(0.70*n)])
val_mmsi   = set(mmsis[int(0.70*n):int(0.85*n)])
test_mmsi  = set(mmsis[int(0.85*n):])

mu_sigma_path = Path(f"parameters_2024_1_3_4_6_ALL_GEAR.pkl") # USE CORRECT PARAMETERS
if mu_sigma_path.exists():
    print(f"Loading mu/sigma from {mu_sigma_path}")
    with open(mu_sigma_path, "rb") as f:
        params = pickle.load(f)

    mu = params["mu"]
    sigma = params["sigma"]

else:

    # Fit normalization on TRAIN ONLY
    sum_x = pd.Series(0.0, index=FEATURES)
    sum_x2 = pd.Series(0.0, index=FEATURES)
    count = 0

    needed_cols = ["mmsi", "date_time_utc"] + BASE_FEATURES

    for f in files:
        df = pd.read_parquet(f, columns=needed_cols)
        df = df[df["mmsi"].isin(train_mmsi)].copy()

        df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
        month = df["date_time_utc"].dt.month

        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        x = df[FEATURES]
        sum_x += x.sum()
        sum_x2 += (x ** 2).sum()
        count += len(x)

    mu = sum_x / count
    sigma = np.sqrt((sum_x2 / count) - mu**2).replace(0, 1)
    print("mu and sigma found")

    with open(mu_sigma_path, "wb") as f:
        pickle.dump({"mu": mu, "sigma": sigma}, f)


class AISWindowDataset(IterableDataset):
    def __init__(self, files, mmsi_set, features, mu, sigma,
                 window=WINDOW, stride=STRIDE, shuffle_files=False):
        self.files = files
        self.mmsi_set = mmsi_set
        self.features = features
        self.mu = mu
        self.sigma = sigma
        self.window = window
        self.stride = stride
        self.shuffle_files = shuffle_files

    def make_windows(self, traj_df):
        X_all = traj_df[self.features].to_numpy(dtype=np.float32)
        y_all = traj_df["y_train"].to_numpy(dtype=np.float32)
        w_all = traj_df["sample_weight"].to_numpy(dtype=np.float32)

        n = len(traj_df)
        if n < 8:
            return

        for start in range(0, max(1, n - self.window + 1), self.stride):
            end = start + self.window

            x = X_all[start:end]
            y = y_all[start:end]
            w = w_all[start:end]

            if len(x) < self.window:
                pad = self.window - len(x)
                x = np.vstack([x, np.zeros((pad, x.shape[1]), dtype=np.float32)])
                y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
                w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])

            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle_files:
            np.random.shuffle(files)

        cols = ["mmsi", "trajectory_id", "date_time_utc", "y_train", "sample_weight"] + BASE_FEATURES

        for f in files:
            df = pd.read_parquet(f, columns=cols)
            df = df[df["mmsi"].isin(self.mmsi_set)].copy()

            if len(df) == 0:
                continue
            
            df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

            month = df["date_time_utc"].dt.month

            df["month_sin"] = np.sin(2 * np.pi * month / 12)
            df["month_cos"] = np.cos(2 * np.pi * month / 12)

            df[self.features] = (df[self.features] - self.mu) / self.sigma

            df["ra_accel"] = df["ra_accel"].clip(-5, 5)
            df["ra_jerk"]  = df["ra_jerk"].clip(-5, 5)
            df["ra_dcog"]  = df["ra_dcog"].clip(-5, 5)

            df = df.sort_values(["trajectory_id", "date_time_utc"])

            for _, traj in df.groupby("trajectory_id", sort=False):
                yield from self.make_windows(traj)



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
            nn.Linear(dense, 1),   # binary logit per timestep
        )

    def forward(self, x):
        # x: (B, T, F)
        h, _ = self.lstm(x)          # (B, T, 2*hidden)
        logits = self.head(h).squeeze(-1)  # (B, T)
        return logits

train_ds = AISWindowDataset(files, train_mmsi, FEATURES, mu, sigma, shuffle_files=True, stride=STRIDE, window=WINDOW)
val_ds   = AISWindowDataset(files, val_mmsi, FEATURES, mu, sigma, stride=STRIDE, window=WINDOW)
test_ds  = AISWindowDataset(files, test_mmsi, FEATURES, mu, sigma, stride=STRIDE, window=WINDOW)

# 3. DataLoaders

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False,
                          num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True)

val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, pin_memory=False)

test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                         num_workers=0, pin_memory=False)


if torch.cuda.is_available():
    print("Cuda available.")
    device = torch.device("cuda")
else:
    print("Cuda NOT available. Using CPU.")
    device = torch.device("cpu")

# ------------------------------------------------------------------
# Re-instantiate before training (prevents stale state on cell re-run)
# ------------------------------------------------------------------

torch.manual_seed(42)

model = FishingBiLSTM(n_features=len(FEATURES),
                      hidden=HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT, dense=DENSE).to(device)

neg = 0
pos = 0

cols = ["mmsi", "sample_weight", "y_train"]

for f in files:
    df_tmp = pd.read_parquet(f, columns=cols)
    df_tmp = df_tmp[df_tmp["mmsi"].isin(train_mmsi)].copy()
    df_tmp = df_tmp[df_tmp["sample_weight"] == 1]

    neg += (df_tmp["y_train"] == 0).sum()
    pos += (df_tmp["y_train"] == 1).sum()

pos_weight = torch.tensor([neg / max(pos, 1)], device=device, dtype=torch.float32)
print("pos_weight:", pos_weight.item())

bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

def masked_loss(logits, y, mask):
    m = mask.float()
    per = bce(logits, y)
    return (per * m).sum() / m.sum().clamp_min(1.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2)

# ------------------------------------------------------------------
# Epoch runner — per-message metrics (padding ignored), NaN-safe
# ------------------------------------------------------------------
print("Starting epochs")
def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    tot_loss, tot_n = 0.0, 0
    tp = fp = fn = tn = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, m in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            logits = model(x)
            loss = masked_loss(logits, y, m)

            if train:
                # Guard: skip any bad step instead of poisoning weights
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n = m.sum().item()
            tot_loss += loss.item() * n
            tot_n    += n

            pred = (torch.sigmoid(logits) > 0.5).int()
            yi, mb = y.int(), m.bool()
            tp += ((pred == 1) & (yi == 1) & mb).sum().item()
            fp += ((pred == 1) & (yi == 0) & mb).sum().item()
            fn += ((pred == 0) & (yi == 1) & mb).sum().item()
            tn += ((pred == 0) & (yi == 0) & mb).sum().item()

    avg  = tot_loss / max(tot_n, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
    return avg, prec, rec, f1, acc

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

model_name = "models/model_bilstm_tuned_2024_1_3_4_6_ALL_GEAR_no_DIST.pt"

best_val = float("inf")
bad, patience = 0, 3

history = []
for epoch in range(1, 16):
    tr = run_epoch(train_loader, train=True)
    vl = run_epoch(val_loader,   train=False)
    scheduler.step(vl[0])
    print(f"Ep{epoch:02d} | train loss {tr[0]:.4f} f1 {tr[3]:.3f} | "
          f"val loss {vl[0]:.4f} p {vl[1]:.3f} r {vl[2]:.3f} f1 {vl[3]:.3f}")
    history.append({
        "epoch": epoch,
        "train_loss": tr[0], "train_f1": tr[3],
        "val_loss": vl[0],   "val_p": vl[1], "val_r": vl[2],
        "val_f1": vl[3],     "val_acc": vl[4],
    })
    #pd.DataFrame(history).to_csv("training_stats/training_history_tuned_2024_1_3_4_6_ALL_GEAR.csv", index=False)
    if vl[0] < best_val:
        best_val = vl[0]
        torch.save(model.state_dict(), model_name)
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ------------------------------------------------------------------
# Final test
# ------------------------------------------------------------------
import os
if os.path.exists(model_name):
    model.load_state_dict(torch.load(model_name))
te = run_epoch(test_loader, train=False)
print(f"TEST | loss {te[0]:.4f}  p {te[1]:.3f}  r {te[2]:.3f}  f1 {te[3]:.3f}  acc {te[4]:.3f}")

""" with open("training_stats/results_tuned_2024_1_3_4_6_ALL_GEAR.json", "w") as f:
    json.dump({"loss": te[0], "precision": te[1], "recall": te[2],
               "f1": te[3], "accuracy": te[4], "params": best_params}, f, indent=2) """