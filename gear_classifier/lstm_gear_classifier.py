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

GEARS = ["Trål", "Krokredskap", "Not", "Snurrevad", "Garn"]

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt"]

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES
FILE_BASE = "../three_months/only_gear_reports"
TRAIN_FILES = [f"{FILE_BASE}/onl_2023_1_3_feats.parquet", f"{FILE_BASE}/onl_2023_4_6_feats.parquet"]
VAL_TEST_FILES = [f"{FILE_BASE}/onl_2024_1_3_feats.parquet"]
#FOLDER = "gear_classifier"
TAG = "gear_class"

# ============================================================
# Hyperparameters (load tuned if present, else mirror binary)
# ============================================================

def all_mmsis_in(files):
    s = set()
    for f in files:
        mmsis = pd.read_parquet(f, columns=["mmsi"])["mmsi"]
        mmsis = pd.to_numeric(mmsis, errors="coerce").dropna().astype("int64")
        s.update(mmsis.unique())
    return s

def get_global_val_test_mmsis(which, path=f"../../train_val_test_mmsis_FINAL.csv"):
    split_df = pd.read_csv(path)
    split_df["mmsi"] = split_df["mmsi"].astype("int64")
    mmsis = set(split_df.loc[split_df["split"] == which,"mmsi"])
    return mmsis
 
# All vessels in each quarter (no MMSI split -- the split is by TIME).
val_mmsis = get_global_val_test_mmsis(which="validation")
test_mmsis = get_global_val_test_mmsis(which="test")
all_mmsis_in_train = all_mmsis_in(TRAIN_FILES)
train_mmsis = all_mmsis_in_train - val_mmsis - test_mmsis
assert train_mmsis.isdisjoint(val_mmsis), "Train/val MMSIs overlap!"
assert train_mmsis.isdisjoint(test_mmsis), "Train/test MMSIs overlap!"
#print(f"Train (all 2023) vessels: {len(train_mmsis)} | Val (2024) vessels: {len(val_mmsis)} | Test (2024) vessels: {len(test_mmsis)}")

# ------------------------------------------------------------------
# Normalization stats -- fit on TRAIN (2023) only
# ------------------------------------------------------------------
mu_sigma_path = Path(f"parameters_{TAG}.pkl")
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
        df = df[df["mmsi"].isin(train_mmsis)]
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
# Hyperparameters (load tuned if present, else mirror binary)
# ============================================================
tuned_params_path = Path(f"best_params_gear_class.json")
if tuned_params_path.exists():
    bp = json.load(open(tuned_params_path))["best_params"]
    WINDOW, STRIDE   = bp["window"], bp["stride"]
    N_LAYERS, HIDDEN = bp["n_layers"], bp["hidden"]
    DENSE, DROPOUT   = bp["dense"], bp["dropout"]
    BATCH, LR        = bp["batch"], bp["lr"]
    print("Loaded tuned params", bp)
else:
    WINDOW, STRIDE   = 256, 128
    N_LAYERS, HIDDEN = 2, 256
    DENSE, DROPOUT   = 128, 0.2
    BATCH, LR        = 256, 4.56e-4

MAX_EPOCHS, PATIENCE = 10, 3
INFER_BATCH = 128

GEAR_TO_IDX = {g: i for i, g in enumerate(GEARS)}
IDX_TO_GEAR = {i: g for i, g in enumerate(GEARS)}
N_CLASSES   = len(GEARS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# Class weights (inverse-frequency, 'balanced' style)
#   Computed over valid, gear-labelled TRAIN messages only.
# ============================================================
class_counts = np.zeros(N_CLASSES, dtype=np.float64)
for f in TRAIN_FILES:
    d = pd.read_parquet(f, columns=["mmsi", "gear_report", "sample_weight"])
    d["mmsi"] = d["mmsi"].astype("int64")
    d = d[d["mmsi"].isin(train_mmsis) & (d["sample_weight"] == 1)]
    idx = d["gear_report"].map(GEAR_TO_IDX).dropna().astype(int)
    for c, n in idx.value_counts().items():
        class_counts[c] += n
    del d; gc.collect()

print("Class counts:", dict(zip(GEARS, class_counts.astype(int))))
class_weights = class_counts.sum() / (N_CLASSES * np.maximum(class_counts, 1.0))
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
print("Class weights:", [round(w, 3) for w in class_weights.tolist()])

# ============================================================
# Dataset  (one gear type per segment; per-message target)
# ============================================================
class GearWindowDataset(IterableDataset):
    def __init__(self, files, mmsi_set, features, mu, sigma,
                 window=WINDOW, stride=STRIDE, shuffle_files=False):
        self.files, self.mmsi_set = files, mmsi_set
        self.features, self.mu, self.sigma = features, mu, sigma
        self.window, self.stride = window, stride
        self.shuffle_files = shuffle_files

    def make_windows(self, traj_df):
        X_all = traj_df[self.features].to_numpy(dtype=np.float32)
        y_all = traj_df["gear_idx"].to_numpy(dtype=np.int64)    # -1 = invalid
        w_all = traj_df["loss_mask"].to_numpy(dtype=np.float32) # 0/1
        n = len(traj_df)
        if n < 8:
            return
        for start in range(0, max(1, n - self.window + 1), self.stride):
            end = start + self.window
            x, y, w = X_all[start:end], y_all[start:end], w_all[start:end]
            if len(x) < self.window:
                pad = self.window - len(x)
                x = np.vstack([x, np.zeros((pad, x.shape[1]), np.float32)])
                y = np.concatenate([y, np.zeros(pad, np.int64)])
                w = np.concatenate([w, np.zeros(pad, np.float32)])
            y = np.where(y >= 0, y, 0).astype(np.int64)  # clamp; masked anyway
            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle_files:
            np.random.shuffle(files)
        cols = ["mmsi", "segment_id", "date_time_utc",
                "gear_report", "sample_weight"] + BASE_FEATURES
        for f in files:
            df = pd.read_parquet(f, columns=cols)
            df["mmsi"] = df["mmsi"].astype("int64")
            df = df[df["mmsi"].isin(self.mmsi_set)].copy()
            df = df[df["gear_report"].isin(GEARS)]   
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
            # target + loss mask
            df["gear_idx"] = df["gear_report"].map(GEAR_TO_IDX)
            df["loss_mask"] = ((df["sample_weight"] == 1) &
                               df["gear_idx"].notna()).astype(np.float32)
            df["gear_idx"] = df["gear_idx"].fillna(-1).astype(np.int64)
            df = df.sort_values(["segment_id", "date_time_utc"])
            for _, traj in df.groupby("segment_id", sort=False):
                yield from self.make_windows(traj)

# ============================================================
# Model — same backbone, multiclass head
# ============================================================
class GearLSTM(nn.Module):
    def __init__(self, n_features, n_classes, hidden=HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT, dense=DENSE):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True,
                            bidirectional=False,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, dense), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dense, n_classes),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h)          # (B, T, n_classes)

# ============================================================
# Weighted, masked cross-entropy
# ============================================================
ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

def masked_loss(logits, y, mask):
    B, T, C = logits.shape
    per = ce(logits.reshape(B * T, C), y.reshape(B * T)).reshape(B, T)
    m = mask.float()
    return (per * m).sum() / m.sum().clamp_min(1.0)

# ============================================================
# Epoch runner — accumulates a confusion matrix for metrics
# ============================================================
def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.maximum(tp + fp, 1)
    rec  = tp / np.maximum(tp + fn, 1)
    f1   = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    acc  = tp.sum() / max(cm.sum(), 1)
    return acc, f1.mean(), prec, rec, f1   # macro-F1

def run_epoch(model, loader, optimizer=None, train=False):
    model.train() if train else model.eval()
    tot_loss, tot_n = 0.0, 0
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)  # cm[true, pred]
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            loss = masked_loss(logits, y, m)
            if train:
                if not torch.isfinite(loss):
                    optimizer.zero_grad(); continue
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n_m = m.sum().item()
            tot_loss += loss.item() * n_m; tot_n += n_m
            valid = m.bool()
            t = y[valid].cpu().numpy()
            p = logits.argmax(-1)[valid].cpu().numpy()
            np.add.at(cm, (t, p), 1)
    avg = tot_loss / max(tot_n, 1)
    acc, macro_f1, _, _, _ = metrics_from_cm(cm)
    return avg, acc, macro_f1, cm

# ============================================================
# Loaders
# ============================================================
train_ds = GearWindowDataset(TRAIN_FILES, train_mmsis, FEATURES, mu, sigma,
                             shuffle_files=True)
val_ds   = GearWindowDataset(VAL_TEST_FILES, val_mmsis, FEATURES, mu, sigma)
train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=0,
                          pin_memory=torch.cuda.is_available(), drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH, num_workers=0)

# ============================================================
# Causal streaming prediction (mirrors your fishing inference)
# ============================================================
TEST_COLUMNS = list(dict.fromkeys(
    ["mmsi", "segment_id", "date_time_utc", "gear_report",
     "sample_weight", "report", "lon", "lat"] + BASE_FEATURES))

def get_test_df(files, mmsi_list):
    dfs = [pd.read_parquet(f, engine="pyarrow", columns=TEST_COLUMNS,
                           filters=[("mmsi", "in", mmsi_list), ("gear_report", "in", GEARS)]) for f in files]
    return pd.concat(dfs, ignore_index=True)

def prepare_test_df(df):
    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    m = df["date_time_utc"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    for col in FEATURES:
        df[col] = (df[col] - mu[col]) / sigma[col]
    df["ra_accel"] = df["ra_accel"].clip(-5, 5)
    df["ra_jerk"]  = df["ra_jerk"].clip(-5, 5)
    df["ra_dcog"]  = df["ra_dcog"].clip(-5, 5)
    return df.sort_values(["segment_id", "date_time_utc"]).reset_index(drop=True)

def predict_probs(model, X_all):
    n = len(X_all)
    probs = np.zeros((n, N_CLASSES), dtype=np.float32)
    head = min(WINDOW, n)
    x_head = torch.from_numpy(X_all[:head][None]).to(device)
    probs[:head] = torch.softmax(model(x_head)[0], -1).cpu().numpy()
    if n > WINDOW:
        sw = np.lib.stride_tricks.sliding_window_view(X_all, WINDOW, axis=0)
        sw = sw.transpose(0, 2, 1)[1:]            # (n-WINDOW, WINDOW, F)
        for i in range(0, len(sw), INFER_BATCH):
            b = torch.from_numpy(np.ascontiguousarray(sw[i:i+INFER_BATCH])).to(device)
            last = torch.softmax(model(b)[:, -1], -1).cpu().numpy()
            t0 = WINDOW + i
            probs[t0:t0+len(last)] = last
    return probs

def predict_and_score(model, df, prefix, seed):
    df = df.copy()
    prob_cols = [f"p_{g}" for g in GEARS]
    df[prob_cols] = np.nan
    model.eval()
    with torch.inference_mode():
        for _, traj in df.groupby("segment_id", sort=False):
            X = traj[FEATURES].to_numpy(np.float32)
            if len(X) < 1:
                continue
            p = predict_probs(model, X)
            # running-mean of probs => increasingly confident segment estimate
            p_cum = np.cumsum(p, 0) / np.arange(1, len(p) + 1)[:, None]
            df.loc[traj.index, prob_cols] = p
            df.loc[traj.index, [f"cum_{c}" for c in prob_cols]] = p_cum

    P = df[prob_cols].to_numpy()
    df["pred_idx"]  = P.argmax(1)
    df["pred_gear"] = df["pred_idx"].map(IDX_TO_GEAR)

    if seed == 0:
        df.to_parquet(f"predictions/gear_{prefix}_test_seed{seed}.parquet", index=False)

    # evaluate on valid, gear-labelled messages
    y_idx = df["gear_report"].map(GEAR_TO_IDX)
    mask = (df["sample_weight"] == 1) & y_idx.notna()
    y_true = y_idx[mask].astype(int).to_numpy()
    y_prob = P[mask.to_numpy()]
    y_pred = df.loc[mask, "pred_idx"].to_numpy()

    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    acc, macro_f1, prec, rec, f1 = metrics_from_cm(cm)
    ll = log_loss(y_true, y_prob, labels=list(range(N_CLASSES)))

    res = {f"{prefix}_acc": acc, f"{prefix}_macro_f1": macro_f1,
           f"{prefix}_loss": ll}
    for i, g in enumerate(GEARS):
        res[f"{prefix}_f1_{g}"] = f1[i]
        res[f"{prefix}_recall_{g}"] = rec[i]
        res[f"{prefix}_support_{g}"] = int(cm[i].sum())
    print(f"[{prefix}] acc {acc:.4f} macro-F1 {macro_f1:.4f} logloss {ll:.4f}")
    return res

# build seen / unseen test sets once (mu/sigma are seed-independent)
df_unseen = prepare_test_df(get_test_df(VAL_TEST_FILES, list(test_mmsis)))
random.seed(42)
seen_list = random.sample(sorted(train_mmsis), k=len(train_mmsis) // 4)
df_seen   = prepare_test_df(get_test_df(VAL_TEST_FILES, seen_list))

# ============================================================
# Train (single seed shown; wrap in your seed loop like the binary)
# ============================================================
seed = 0
torch.manual_seed(seed); np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model = GearLSTM(len(FEATURES), N_CLASSES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2)

best_val, bad = float("inf"), 0
model_name = f"models/gear_lstm_seed{seed}.pt"
for epoch in range(1, MAX_EPOCHS + 1):
    tr = run_epoch(model, train_loader, optimizer, train=True)
    vl = run_epoch(model, val_loader, train=False)
    scheduler.step(vl[0])
    print(f"Ep{epoch:02d} | train loss {tr[0]:.4f} macroF1 {tr[2]:.4f} "
          f"| val loss {vl[0]:.4f} acc {vl[1]:.4f} macroF1 {vl[2]:.4f}")
    if vl[0] < best_val:
        best_val, bad = vl[0], 0
        #torch.save(model.state_dict(), model_name)
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop at epoch {epoch}"); break

model.load_state_dict(torch.load(model_name, map_location=device))
res = {**predict_and_score(model, df_unseen, "unseen", seed),
       **predict_and_score(model, df_seen,   "seen",   seed)}



# Want an LSTM that runs over each trajectory the same way the fishing/non-fishing does. Same strucuture, only now it is gear classification. column gear_report is the target
# and belongs to one of the GEARS. Every trajectory only includes ONE gear type. It should predict one gear type per message, So it simulates that the fishing/non-fishing predicts
#fishing. This model should, from the first fishing message, predict the gear type. This prediction will get better and better as more fishing messages are fed to the model.
# Class weights should be used as it is far more Trål messages than garn for example. 