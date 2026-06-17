import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from sklearn.metrics import log_loss
import gc
import random

# Load tuned parameters
tuned_params_path = Path("tuning_FINAL/best_params_LSTM_tune-2023-no-val-test.json")

if tuned_params_path.exists():
    with open(tuned_params_path, "r") as file:
        best_params = json.load(file)["best_params"]
    print("Loaded tuned params ", best_params)
    WINDOW   = best_params["window"]
    STRIDE   = best_params["stride"]
    N_LAYERS = best_params["n_layers"]
    HIDDEN   = best_params["hidden"]
    DENSE    = best_params["dense"]
    DROPOUT  = best_params["dropout"]
    BATCH    = best_params["batch"]
    LR       = best_params["lr"]
else:
    print("Tuned parameters not found, using base params ...")
    WINDOW   = 256
    STRIDE   = 128
    N_LAYERS = 3
    HIDDEN   = 256
    DENSE    = 128
    DROPOUT  = 0.321
    BATCH    = 256
    LR       = 4.56e-4
    exit()

BASE = "three_months/feats_new_rule_online"

# TRAIN: all of 2023
TRAIN_FILES = [
    f"{BASE}/2023_1_3_feats.parquet",     # Q1 2023
    f"{BASE}/2023_4_6_feats.parquet",     # Q2 2023
    f"{BASE}/2023_7_9_feats.parquet",     # Q3 2023
    f"{BASE}/2023_10_12_feats.parquet",   # Q4 2023
    f"{BASE}/2024_1_3_feats.parquet",     # Q1 2024
    f"{BASE}/2024_4_6_feats.parquet",     # Q2 2024
    f"{BASE}/2024_7_9_feats.parquet",     # Q3 2024
    f"{BASE}/2024_10_12_feats.parquet",   # Q4 2024
]

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt"]

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES

EPOCHS = 10

FOLDER = "training_FULL_MODEL"
TAG = "lstm_train_2023_and_2024"

# ------------------------------------------------------------------
# Normalization stats -- fit on TRAIN (2023) only
# ------------------------------------------------------------------
mu_sigma_path = Path(f"{FOLDER}/parameters_{TAG}.pkl")
if mu_sigma_path.exists():
    print(f"Loading mu/sigma from {mu_sigma_path}")
    with open(mu_sigma_path, "rb") as f:
        params = pickle.load(f)
    mu, sigma = params["mu"], params["sigma"]
else:
    sum_x  = pd.Series(0.0, index=FEATURES)
    sum_x2 = pd.Series(0.0, index=FEATURES)
    count = 0
    needed_cols = ["mmsi", "date_time_utc"] + BASE_FEATURES
    for f in TRAIN_FILES:
        df = pd.read_parquet(f, columns=needed_cols)
        print("mmsis in training param df before: ", df["mmsi"].nunique())
        df["mmsi"] = df["mmsi"].astype("int64")
        print("mmsis in training param df after: ", df["mmsi"].nunique())
        df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
        month = df["date_time_utc"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        x = df[FEATURES]
        sum_x  += x.sum()
        sum_x2 += (x ** 2).sum()
        count  += len(x)
        del df, x, month
        gc.collect()
    mu = sum_x / count
    sigma = np.sqrt((sum_x2 / count) - mu ** 2).replace(0, 1)
    with open(mu_sigma_path, "wb") as f:
        pickle.dump({"mu": mu, "sigma": sigma}, f)
    print(f"Fit mu/sigma on Q1 and saved to {mu_sigma_path}")

# ============================================================
# Dataset + Model
# ============================================================

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
        cols = ["mmsi", "trajectory_id", "date_time_utc",
                "y_train", "sample_weight"] + BASE_FEATURES
        for f in files:
            df = pd.read_parquet(f, columns=cols)
            #df = df[df["mmsi"].isin(self.mmsi_set)].copy()
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


# ============================================================
# Loaders + device + class imbalance (all fixed across seeds)
# ============================================================

train_ds = AISWindowDataset(TRAIN_FILES, None, FEATURES, mu, sigma,
                            shuffle_files=True, stride=STRIDE, window=WINDOW)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False,
                          num_workers=0,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=True)


if torch.cuda.is_available():
    print("Cuda available.")
    device = torch.device("cuda")
else:
    print("Cuda NOT available. Using CPU.")
    device = torch.device("cpu")

neg = 0
pos = 0
cols_w = ["mmsi", "sample_weight", "y_train"]
for f in TRAIN_FILES:
    df_tmp = pd.read_parquet(f, columns=cols_w)
    df_tmp = df_tmp[df_tmp["sample_weight"] == 1]
    neg += (df_tmp["y_train"] == 0).sum()
    pos += (df_tmp["y_train"] == 1).sum()
    del df_tmp
    gc.collect()

pos_weight = torch.tensor([neg / max(pos, 1)], device=device, dtype=torch.float32)
print("pos_weight:", pos_weight.item())

bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

def masked_loss(logits, y, mask):
    m = mask.float()
    per = bce(logits, y)
    return (per * m).sum() / m.sum().clamp_min(1.0)

# ============================================================
# Epoch runner
# ============================================================
 
def run_epoch(model, loader, optimizer=None, train=False):
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
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n_m = m.sum().item()
            tot_loss += loss.item() * n_m
            tot_n    += n_m
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

seed = 42
print(f"\n========== TRAINING FULL MODEL (seed {seed}) ==========")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
 
model = FishingLSTM(
    n_features=len(FEATURES),
    hidden=HIDDEN, n_layers=N_LAYERS,
    dropout=DROPOUT, dense=DENSE,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# No validation set, so step the scheduler on the (training) loss.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2)
 
model_name = f"{FOLDER}/model_{TAG}_seed{seed}.pt"
history = []
 
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_prec, tr_rec, tr_f1, tr_acc = run_epoch(
        model, train_loader, optimizer=optimizer, train=True)
 
    scheduler.step(tr_loss)
    lr_now = optimizer.param_groups[0]["lr"]
 
    print(f"[epoch {epoch:02d}/{EPOCHS}] "
          f"loss={tr_loss:.4f} prec={tr_prec:.3f} rec={tr_rec:.3f} "
          f"f1={tr_f1:.3f} acc={tr_acc:.3f} lr={lr_now:.2e}")
 
    history.append({
        "epoch": epoch, "train_loss": tr_loss,
        "train_prec": tr_prec, "train_rec": tr_rec,
        "train_f1": tr_f1, "train_acc": tr_acc, "lr": lr_now,
    })
 
    # Save the latest checkpoint every epoch (crash-safe; final epoch = final model).
    torch.save(model.state_dict(), model_name)
 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
# Save a clearly-named final copy + the training history
final_name = f"{FOLDER}/model_{TAG}_FULL_FINAL.pt"
torch.save(model.state_dict(), final_name)
pd.DataFrame(history).to_csv(f"{FOLDER}/history_{TAG}.csv", index=False)
 
print(f"\nDone. Final model: {final_name}")
print(f"Normalization stats: {mu_sigma_path}")
print(f"History: {FOLDER}/history_{TAG}.csv")


######################