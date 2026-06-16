import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
import itertools, json, time
import os
import pickle
import optuna
from pathlib import Path

WINDOW = 256
STRIDE = 128

# TAG FOR FILES
TAG = "BILSTM_tune-2023-no-val-test"
FOLDER = "tuning_FINAL"

print("--------", TAG, "----------")

# ------------------------------------------------------------------
# FILES and FEATURES
# ------------------------------------------------------------------
TUNING_FILES = ["three_months/feats_new_rule_online/2023_1_3_feats.parquet",
                "three_months/feats_new_rule_online/2023_4_6_feats.parquet",
                "three_months/feats_new_rule_online/2023_7_9_feats.parquet",
                "three_months/feats_new_rule_online/2023_10_12_feats.parquet"] 

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt"]

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES

needed_cols = ["mmsi", "date_time_utc"] + BASE_FEATURES

def all_mmsis_in(files):
    s = set()
    for f in files:
        mmsis = pd.read_parquet(f, columns=["mmsi"])["mmsi"]
        mmsis = pd.to_numeric(mmsis, errors="coerce").dropna().astype("int64")
        s.update(mmsis.unique())
    return s

def get_global_val_test_mmsis(which, path="../train_val_test_mmsis_FINAL.csv"):
    split_df = pd.read_csv(path)
    split_df["mmsi"] = split_df["mmsi"].astype("int64")
    mmsis = set(split_df.loc[split_df["split"] == which,"mmsi"])
    return mmsis
 
# validation mmsis from the whole of 2024 so we dont validate on the mmsis saved for testing/validation only
GLOB_val_mmsis = get_global_val_test_mmsis(which="validation")
GLOB_test_mmsis = get_global_val_test_mmsis(which="test")
print(f"{len(GLOB_val_mmsis)} mmsis are reserved for validation, and {len(GLOB_test_mmsis)} are reserved for testing. We do not tune on these!")
all_mmsis_in_tuning = all_mmsis_in(TUNING_FILES)

tuning_mmsis = all_mmsis_in_tuning - GLOB_val_mmsis - GLOB_test_mmsis # REMOVE all validation and test mmsis, so these vessel are not seen by the tuning.
assert tuning_mmsis.isdisjoint(GLOB_val_mmsis), "Tuning MMSIS include val MMSIs!"
assert tuning_mmsis.isdisjoint(GLOB_test_mmsis), "Tuning MMSIS include test MMSIs!"

print(f"MMSIs available for tuning after excluding val/test mmsis training file: {len(tuning_mmsis)}")

# Split tuning mmsis in 80/20 for tuning
tuning_mmsis = np.array(list(tuning_mmsis))
rng = np.random.default_rng(42)
rng.shuffle(tuning_mmsis)

# Split into train test and validation set by mmsi so that no vessel appear in both.
n = len(tuning_mmsis)
train_mmsis = set(tuning_mmsis[:int(0.80*n)])
val_mmsis   = set(tuning_mmsis[int(0.80*n):])


print(f"Train 80% of Q1 Q3 2023 vessels excluding the global val and test mmsis: {len(train_mmsis)} | Val 20% of the available vessels: {len(val_mmsis)}")
print(f"Are there vessels in both train and val?: "
      f"{len(train_mmsis & val_mmsis)}")

# ------------------------------------------------------------------
# Normalization stats -- fit on TRAIN only
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
    for f in TUNING_FILES:
        df = pd.read_parquet(f, columns=needed_cols)
        df = df[df["mmsi"].isin(train_mmsis)]
        df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
        month = df["date_time_utc"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        x = df[FEATURES]
        sum_x  += x.sum()
        sum_x2 += (x ** 2).sum()
        count  += len(x)
    mu = sum_x / count
    sigma = np.sqrt((sum_x2 / count) - mu ** 2).replace(0, 1)
    with open(mu_sigma_path, "wb") as f:
        pickle.dump({"mu": mu, "sigma": sigma}, f)
    print(f"Fit mu/sigma on Q1 and saved to {mu_sigma_path}")
 


class AISWindowDataset(IterableDataset):
    def __init__(self, files, mmsi_set, features, mu, sigma,
                 window=128, stride=64, shuffle_files=False):
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
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.3, dense=64):
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

# ------------------------------------------------------------------
# Device + class imbalance (pos_weight fit on TRAIN / Q1)
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
 
neg, pos = 0, 0
for f in TUNING_FILES:
    df_tmp = pd.read_parquet(f, columns=["mmsi", "sample_weight", "y_train"])
    df_tmp = df_tmp[df_tmp["mmsi"].isin(train_mmsis)]
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

# ------------------------------------------------------------------
# Caching: train caches come from Q1, val caches from Q3
# ------------------------------------------------------------------
def cache_windows(files, mmsi_set, name, window, stride):
    out_path = Path(f"{FOLDER}/cache_{name}_w{window}_s{stride}_{TAG}.pt")
    if out_path.exists():
        print(f"  already cached: {out_path.name}")
        return
    print(f"  caching {out_path.name} ...")
    ds = AISWindowDataset(files, mmsi_set, FEATURES, mu, sigma,
                          window=window, stride=stride)
    xs, ys, ms = [], [], []
    for x, y, m in ds:
        xs.append(x); ys.append(y); ms.append(m)
    torch.save({"x": torch.stack(xs),
                "y": torch.stack(ys),
                "m": torch.stack(ms)}, out_path)
 
 
print("Building caches...")
# Only the (window, stride) combos the search can actually use
# (stride > window // 2 is pruned in the objective).
for w, s in [(128, 64), (128, 128), (256, 128), (256, 256)]:
    cache_windows(TUNING_FILES, train_mmsis, "train", w, s)
    cache_windows(TUNING_FILES,   val_mmsis,   "val",   w, s)
print("Caches ready.\n")

def run_epoch(model, loader, optimizer, device, train: bool):
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
            n = m.sum().item()
            tot_loss += loss.item() * n
            tot_n += n
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
# Train one config (Optuna pruning supported)
# ------------------------------------------------------------------
def train_one_config(cfg, trial=None, max_epochs=6):
    torch.manual_seed(42)
 
    train_cache = torch.load(
        f"{FOLDER}/cache_train_w{cfg['window']}_s{cfg['stride']}_{TAG}.pt")
    val_cache = torch.load(
        f"{FOLDER}/cache_val_w{cfg['window']}_s{cfg['stride']}_{TAG}.pt")
 
    train_ds = TensorDataset(train_cache["x"], train_cache["y"], train_cache["m"])
    val_ds   = TensorDataset(val_cache["x"],   val_cache["y"],   val_cache["m"])
 
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                              num_workers=0, drop_last=True,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg["batch"], num_workers=0)
 
    model = FishingBiLSTM(
        n_features=len(FEATURES),
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        dense=cfg["dense"],
    ).to(device)
 
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1)
 
    best = {"val_loss": float("inf"), "val_f1": 0.0, "epoch": -1}
    bad, patience = 0, 2
 
    for ep in range(1, max_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, train=True)
        vl = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step(vl[0])
        print(f"  ep{ep} train_loss {tr[0]:.4f} | "
              f"val_loss {vl[0]:.4f} p {vl[1]:.3f} r {vl[2]:.3f} f1 {vl[3]:.3f}")
        if vl[0] < best["val_loss"]:
            best = {"val_loss": vl[0], "val_f1": vl[3],
                    "val_p": vl[1], "val_r": vl[2], "epoch": ep}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if trial is not None:
            trial.report(vl[0], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
 
    return best

# ------------------------------------------------------------------
# Optuna objective + study
# ------------------------------------------------------------------
def objective(trial):
    cfg = {
        "hidden":   trial.suggest_categorical("hidden", [64, 128, 256]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout":  trial.suggest_float("dropout", 0.1, 0.5),
        "batch":    trial.suggest_categorical("batch", [64, 128, 256]),
        "lr":       trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "window":   trial.suggest_categorical("window", [128, 256]),
        "stride":   trial.suggest_categorical("stride", [64, 128, 256]),
        "dense":    trial.suggest_categorical("dense", [32, 64, 128]),
    }
    if cfg["stride"] not in (cfg["window"] // 2, cfg["window"]):
        raise optuna.TrialPruned()
    print(f"\nTrial {trial.number}: {cfg}")
    best = train_one_config(cfg, trial=trial, max_epochs=6)
    return best["val_loss"]
 
 
study = optuna.create_study(
    direction="minimize",
    study_name=f"fishing_{TAG}",
    storage=f"sqlite:///{FOLDER}/optuna_{TAG}.db",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2, n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(seed=42),
)
 
study.optimize(objective, n_trials=6, show_progress_bar=False)
 
print("\n=== BEST ===")
print("val_loss:", study.best_value)
print("params:  ", study.best_params)
 
with open(f"{FOLDER}/best_params_{TAG}.json", "w") as f:
    json.dump({"best_value": study.best_value,
               "best_params": study.best_params}, f, indent=2)
print(f"Saved best params to {FOLDER}/best_params_{TAG}.json")