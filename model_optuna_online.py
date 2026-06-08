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

#files = sorted(glob.glob("three_months/feats/*.parquet"))

files = [
    "three_months/feats_all_w_traps/2024_1_3_feats.parquet",
    "three_months/feats_all_w_traps/2024_7_9_feats.parquet",
]

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt", "dist_to_shore_km"]

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


mu_sigma_path = Path(f"tuning/parameters_2024_optuna_ALL_GEAR.pkl")
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
    print("Mu and sigma found")

    with open(mu_sigma_path, "wb") as f:
        pickle.dump({"mu": mu, "sigma": sigma}, f)
    print(f"Saved mu/sigma to {mu_sigma_path}")


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


class FishingLSTM(nn.Module):
    def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.3, dense=64):
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
            nn.Linear(dense, 1),   # binary logit per timestep
        )

    def forward(self, x):
        # x: (B, T, F)
        h, _ = self.lstm(x)          # (B, T, 2*hidden)
        logits = self.head(h).squeeze(-1)  # (B, T)
        return logits


BATCH = 128
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
# Class imbalance (computed once, shared across trials)
# ------------------------------------------------------------------
neg, pos = 0, 0
cols = ["mmsi", "sample_weight", "y_train"]
for f in files:
    df_tmp = pd.read_parquet(f, columns=cols)
    df_tmp = df_tmp[df_tmp["mmsi"].isin(train_mmsi)]
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
# Generic epoch runner (no globals)
# ------------------------------------------------------------------

def cache_windows(mmsi_set, name, window, stride):
    out_path = Path(f"tuning/cache_{name}_w{window}_s{stride}_ALL_GEAR.pt")
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
for w, s in [(128, 64), (128, 128), (256, 64), (256, 128)]:
    cache_windows(train_mmsi, "train", w, s)
    cache_windows(val_mmsi,   "val",   w, s)
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
# Train one hyperparameter config (with Optuna pruning support)
# ------------------------------------------------------------------
def train_one_config(cfg, trial=None, max_epochs=6):
    torch.manual_seed(42)

    # Load cached windows instead of re-reading parquet
    train_cache = torch.load(f"tuning/cache_train_w{cfg['window']}_s{cfg['stride']}_ALL_GEAR.pt")
    val_cache   = torch.load(f"tuning/cache_val_w{cfg['window']}_s{cfg['stride']}_ALL_GEAR.pt")

    train_ds = TensorDataset(train_cache["x"], train_cache["y"], train_cache["m"])
    val_ds   = TensorDataset(val_cache["x"],   val_cache["y"],   val_cache["m"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                              num_workers=0, drop_last=True,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch"], num_workers=0)

    model = FishingLSTM(
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

        # Optuna pruning: report and possibly stop early
        if trial is not None:
            trial.report(vl[0], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best

# ------------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------------
def objective(trial):
    cfg = {
        "hidden":   trial.suggest_categorical("hidden", [64, 128, 256]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout":  trial.suggest_float("dropout", 0.1, 0.5),
        "batch":    trial.suggest_categorical("batch", [64, 128, 256]),
        "lr":       trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        #"wd":       trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "window":   trial.suggest_categorical("window", [128, 256]),
        "stride":   trial.suggest_categorical("stride", [64, 128]),
        "dense":    trial.suggest_categorical("dense", [32, 64, 128]),
    }
    # Guard against silly combos
    if cfg["stride"] > cfg["window"]:
        raise optuna.TrialPruned()

    print(f"\nTrial {trial.number}: {cfg}")
    best = train_one_config(cfg, trial=trial, max_epochs=6)
    return best["val_loss"]

# ------------------------------------------------------------------
# Run the study
# ------------------------------------------------------------------

study = optuna.create_study(
    direction="minimize",
    study_name="fishing_lstm_search_ALL_GEAR",
    storage="sqlite:///tuning/optuna_fishing_online_ALL_GEAR.db",   # so you can resume / inspect
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2, n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(seed=42),
)

study.optimize(objective, n_trials=35, show_progress_bar=False)

print("\n=== BEST ===")
print("val_loss:", study.best_value)
print("params:  ", study.best_params)

# Persist best params to use later when scaling up
with open("tuning/best_params_online_ALL_GEAR.json", "w") as f:
    json.dump({"best_value": study.best_value,
               "best_params": study.best_params}, f, indent=2)