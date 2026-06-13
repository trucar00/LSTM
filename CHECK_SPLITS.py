"""
Multi-seed BiLSTM training + evaluation on 2025_1_3 external test set.

For each seed:
  1. Re-initialize the model / optimizer / scheduler.
  2. Train with early stopping (same logic as the single-seed script).
  3. Reload best checkpoint, compute the internal 15% test metrics (sanity check).
  4. Predict on the 2025_1_3 file (overlapping windows averaged) and score
     against y_train on sample_weight == 1 rows.
  5. Append one row to multi_seeds_results/seed_results.csv.

What is FIXED across seeds:
  - MMSI split (rng seed = 42)
  - mu / sigma (loaded from pickle)
  - pos_weight
  - the pre-normalized 2025 dataframe (computed once)

What VARIES per seed:
  - torch / numpy seeding
  - model weight init
  - dataloader iteration order (IterableDataset uses np.random.shuffle)
"""

import os
import json
import pickle
from pathlib import Path
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import gc

# ============================================================
# Config — same loading logic as your training script
# ============================================================

USE_TUNED_PARAMS = True
tuned_params_path = Path("tuning/best_params_BILSTM_NEW_RULE_NO_DIST.json")

if USE_TUNED_PARAMS and tuned_params_path.exists():
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
    print("Using default (non-tuned) params")
    WINDOW   = 256
    STRIDE   = 128
    HIDDEN   = 128
    DROPOUT  = 0.3
    N_LAYERS = 2
    DENSE    = 64
    BATCH    = 128
    LR       = 1e-4
    best_params = None

files = [
    "three_months/feats_new_rule_bilstm/2024_1_3_feats.parquet",
    "three_months/feats_new_rule_bilstm/2024_4_6_feats.parquet",
    "three_months/feats_new_rule_bilstm/2024_7_9_feats.parquet",
    "three_months/feats_new_rule_bilstm/2024_10_12_feats.parquet",
]

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk",
                 "log_dist", "ra_dcog", "log_dt"]
SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES

#SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SEEDS = [0, 1]
MAX_EPOCHS = 15
PATIENCE = 3

# ============================================================
# MMSI split — fixed across seeds so train/val/test are identical
# ============================================================

all_mmsis = set()
for f in files:
    print("reading", f)
    m = pd.read_parquet(f, columns=["mmsi"])["mmsi"].unique()
    all_mmsis.update(m)

mmsis = np.array(list(all_mmsis))
split_rng = np.random.default_rng(5)
split_rng.shuffle(mmsis)
n = len(mmsis)
train_mmsi = set(mmsis[:int(0.70 * n)])
val_mmsi   = set(mmsis[int(0.70 * n):int(0.85 * n)])
test_mmsi  = set(mmsis[int(0.85 * n):])

# Read only the necessary columns
split_df = pd.concat(
    [
        pd.read_parquet(f, columns=["mmsi", "gear_report"])
        for f in files
    ],
    ignore_index=True,
)

# Assign each message to a split based on MMSI
split_df["split"] = np.select(
    [
        split_df["mmsi"].isin(train_mmsi),
        split_df["mmsi"].isin(val_mmsi),
        split_df["mmsi"].isin(test_mmsi),
    ],
    [
        "Train",
        "Validation",
        "Test",
    ],
    default="Unknown",
)

# Number of unique MMSIs per gear type and split
mmsis_per_gear = (
    split_df
    .groupby(["gear_report", "split"])["mmsi"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(columns=["Train", "Validation", "Test"], fill_value=0)
)

mmsis_per_gear["Total"] = mmsis_per_gear[
    ["Train", "Validation", "Test"]
].sum(axis=1)

print(mmsis_per_gear)

# ============================================================
# Normalization stats
# ============================================================

mu_sigma_path = Path("NEW_SPLIT/parameters_2024_BILSTM.pkl")
if mu_sigma_path.exists():
    print(f"Loading mu/sigma from {mu_sigma_path}")
    with open(mu_sigma_path, "rb") as f:
        params = pickle.load(f)
    mu, sigma = params["mu"], params["sigma"]
else:
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
        sum_x  += x.sum()
        sum_x2 += (x ** 2).sum()
        count  += len(x)
    mu = sum_x / count
    sigma = np.sqrt((sum_x2 / count) - mu ** 2).replace(0, 1)
    with open(mu_sigma_path, "wb") as f:
        pickle.dump({"mu": mu, "sigma": sigma}, f)
    print("mu and sigma found")

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
    def __init__(self, n_features, hidden=HIDDEN, n_layers=N_LAYERS,
                 dropout=DROPOUT, dense=DENSE):
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


# ============================================================
# Loaders + device + class imbalance (all fixed across seeds)
# ============================================================

train_ds = AISWindowDataset(files, train_mmsi, FEATURES, mu, sigma,
                            shuffle_files=True, stride=STRIDE, window=WINDOW)
val_ds   = AISWindowDataset(files, val_mmsi, FEATURES, mu, sigma,
                            stride=STRIDE, window=WINDOW)
test_ds  = AISWindowDataset(files, test_mmsi, FEATURES, mu, sigma,
                            stride=STRIDE, window=WINDOW)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False,
                          num_workers=0,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=True)
val_loader  = DataLoader(val_ds,  batch_size=BATCH, shuffle=False,
                         num_workers=0, pin_memory=False)

test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                         num_workers=0, pin_memory=False)

if torch.cuda.is_available():
    print("Cuda available.")
    device = torch.device("cuda")
else:
    print("Cuda NOT available. Using CPU.")
    device = torch.device("cpu")

neg = 0
pos = 0
cols_w = ["mmsi", "sample_weight", "y_train"]
for f in files:
    df_tmp = pd.read_parquet(f, columns=cols_w)
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


# ============================================================
# Epoch runner — takes model + optional optimizer
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


# ============================================================
# Pre-normalize the 2025 external test file ONCE
# (mu/sigma are seed-independent, so this is safe)
# ============================================================

print(f"Preparing test set")

test_mmsi_list = list(test_mmsi)

df_test = pd.concat(
    [
        pd.read_parquet(
            f,
            filters=[("mmsi", "in", test_mmsi_list)],
            engine="pyarrow",
        )
        for f in files
    ],
    ignore_index=True,
)

df_test["date_time_utc"] = pd.to_datetime(df_test["date_time_utc"])
_month = df_test["date_time_utc"].dt.month
df_test["month_sin"] = np.sin(2 * np.pi * _month / 12)
df_test["month_cos"] = np.cos(2 * np.pi * _month / 12)
for col in FEATURES:
    df_test[col] = (df_test[col] - mu[col]) / sigma[col]
df_test["ra_accel"] = df_test["ra_accel"].clip(-5, 5)
df_test["ra_jerk"]  = df_test["ra_jerk"].clip(-5, 5)
df_test["ra_dcog"]  = df_test["ra_dcog"].clip(-5, 5)
df_test = df_test.sort_values(["trajectory_id", "date_time_utc"]).reset_index(drop=True)



def predict_and_score_test(model):
    """Run overlapping-window prediction on the pre-normalized 2025 frame,
    average overlapping predictions, then score using the `report` column.

    Metric definitions (match the user's manual evaluation):
      - recall:        correctly predicted fishing among report == "fishing"
      - acc_conf:      correctly predicted non-fishing among report == "conf_no_fishing"
      - precision:     report == "fishing" hits among ALL rows with pred_fishing == 1
                       (rows whose report is anything else — including unknowns —
                        count as false positives in the denominator)
      - f1:            harmonic mean of precision and recall above
    """
    df = df_test.copy()
    df["pred_sum"]   = 0.0
    df["pred_count"] = 0.0

    model.eval()
    with torch.no_grad():
        for traj_id, traj in df.groupby("trajectory_id", sort=False):
            idx = traj.index.to_numpy()
            X_all = traj[FEATURES].to_numpy(dtype=np.float32)
            n_traj = len(traj)
            if n_traj < 8:
                continue
            starts = list(range(0, max(1, n_traj - WINDOW + 1), STRIDE))
            final_start = max(0, n_traj - WINDOW)
            if starts[-1] != final_start:
                starts.append(final_start)  # ensure end of trajectory is covered
            for start in starts:
                end = start + WINDOW
                x = X_all[start:end]
                L = len(x)
                if L < WINDOW:
                    pad = WINDOW - L
                    x = np.vstack([x, np.zeros((pad, x.shape[1]), dtype=np.float32)])
                x_tensor = torch.from_numpy(x[None, :, :]).to(device)
                logits = model(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                valid_idx   = idx[start:start + L]
                valid_probs = probs[:L]
                df.loc[valid_idx, "pred_sum"]   += valid_probs
                df.loc[valid_idx, "pred_count"] += 1

    # Rows we never predicted on (e.g. trajectories with < 8 points) get NaN here,
    # which becomes pred_fishing = 0. That's fine for the report-based scoring
    # below since those rows just look like "not predicted as fishing".
    df["p_fishing"]    = df["pred_sum"] / df["pred_count"]
    #df["pred_fishing"] = (df["p_fishing"] > 0.5).astype(int)

    pred_fishing = (df["p_fishing"].to_numpy() > 0.5)

    n_pred_fish = int(np.sum(pred_fishing))
    n_pred_no_fish = int(np.sum(~pred_fishing))

    report = df["report"].to_numpy()
    rep_fish = report == "fishing"
    rep_conf = report == "conf_no_fishing"

    n_reported_fish = int(np.sum(rep_fish))
    n_reported_conf_no_fish = int(np.sum(rep_conf))

    tp = int(np.sum(pred_fishing & rep_fish))
    fp = int(np.sum(pred_fishing & rep_conf))
    tn = int(np.sum(~pred_fishing & rep_conf))
    fn = int(np.sum(~pred_fishing & rep_fish))

    # unknowns
    rep_unknown = report == "unknown"
    n_unknown = int(rep_unknown.sum())
    n_pred_fish_of_unknown    = int(np.sum(pred_fishing & rep_unknown))
    n_pred_no_fish_of_unknown = int(np.sum(~pred_fishing & rep_unknown))

    # Precision of confirmed.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Accuracy of confirmed.
    accuracy = (tp + tn) / (tp + tn + fp +fn) if (tp + tn + fp +fn) > 0 else 0.0

    # Recall (sensitvity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity, true negative rate, on confirmed non-fishing rows
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    eval_mask = (
            (df["sample_weight"] == 1)
            & df["p_fishing"].notna()
        )
    

    y_true = df.loc[eval_mask, "y_train"].to_numpy(dtype=int)
    y_prob = df.loc[eval_mask, "p_fishing"].to_numpy()

    del df
    gc.collect()

    test_logloss = log_loss(
        y_true,
        y_prob,
        labels=[0, 1],
    )


    return {
        "tp":                  tp,
        "fp":                  fp,
        "tn":                  tn,
        "fn":                  fn,
        "accuracy":            accuracy,
        "recall":              recall,
        "specificity":         specificity,
        "precision":           precision,           
        "f1":                  f1,
        "logloss":             test_logloss,
        "n_pred_fish":         n_pred_fish,
        "n_pred_no_fish":      n_pred_no_fish,
        "n_reported_fish":     n_reported_fish,
        "n_reported_no_fish":  n_reported_conf_no_fish,
        "n_unknowns":          n_unknown,
        "n_pred_fish_of_unknown": n_pred_fish_of_unknown,
        "n_pred_no_fish_of_unknown": n_pred_no_fish_of_unknown
    }


# ============================================================
# Multi-seed loop
# ============================================================

results_csv_path = "NEW_SPLIT/bilstm_seed_results.csv"

# Resume support: skip seeds already in the CSV
done_seeds = set()
all_results = []
if os.path.exists(results_csv_path):
    try:
        existing = pd.read_csv(results_csv_path)
        done_seeds = set(existing["seed"].tolist())
        all_results = existing.to_dict("records")
        print(f"Resuming. Already-completed seeds: {sorted(done_seeds)}")
    except Exception as e:
        print(f"Could not read existing results ({e}); starting fresh.")

for seed in SEEDS:
    if seed in done_seeds:
        print(f"\n[seed {seed}] Already done. Skipping.")
        continue

    print(f"\n========== SEED {seed} ==========")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Fresh model / optimizer / scheduler per seed
    model = FishingBiLSTM(
        n_features=len(FEATURES),
        hidden=HIDDEN, n_layers=N_LAYERS,
        dropout=DROPOUT, dense=DENSE,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    model_name = f"NEW_SPLIT/models/model_bilstm_seed{seed}.pt"
    best_val = float("inf")
    bad = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer=optimizer, train=True)
        vl = run_epoch(model, val_loader, train=False)
        scheduler.step(vl[0])
        print(f"[seed {seed}] Ep{epoch:02d} | "
              f"train loss {tr[0]:.4f} f1 {tr[3]:.3f} | "
              f"val loss {vl[0]:.4f} p {vl[1]:.3f} r {vl[2]:.3f} f1 {vl[3]:.3f}")
        history.append({
            "epoch": epoch,
            "train_loss": tr[0], "train_f1": tr[3],
            "val_loss":   vl[0], "val_p":    vl[1],
            "val_r":      vl[2], "val_f1":   vl[3], "val_acc": vl[4],
        })
        pd.DataFrame(history).to_csv(
            f"NEW_SPLIT/training_history_bilstm_seed{seed}.csv", index=False
        )
        if vl[0] < best_val:
            best_val = vl[0]
            torch.save(model.state_dict(), model_name)
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[seed {seed}] Early stopping at epoch {epoch}")
                break

    # Reload best checkpoint for evaluation
    model.load_state_dict(torch.load(model_name, map_location=device))

    # Internal 15% test split (sanity check)
    te = run_epoch(model, test_loader, train=False)
    print(f"[seed {seed}] INTERNAL TEST | "
          f"loss {te[0]:.4f} p {te[1]:.3f} r {te[2]:.3f} "
          f"f1 {te[3]:.3f} acc {te[4]:.3f}")
    
    # External loss
    # External 2025_1_3 test — the metric that matters
    
    print("Predicting on test set")
    test_metrics = predict_and_score_test(model)
    print(f"[seed {seed}] TEST SET 15% of fishing vessels in 2024 | "
          f"precision {test_metrics['precision']:.3f} "
          f"recall {test_metrics['recall']:.3f} "
          f"specificity {test_metrics['specificity']:.3f} "
          f"f1 {test_metrics['f1']:.3f} "
          f"accuracy {test_metrics['accuracy']:.3f} "
          f"logloss {test_metrics['logloss']:.3f} ")

    row = {
        "seed": seed,
        "best_val_loss": best_val,
        "epochs_trained": len(history),
        "int_window_loss": te[0],
        **test_metrics,
    }
    
    all_results.append(row)

    # Save incrementally so a crash doesn't lose everything
    pd.DataFrame(all_results).to_csv(results_csv_path, index=False)
    torch.cuda.synchronize()
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# Summary across seeds
# ============================================================

df_res = pd.DataFrame(all_results)
print("\n========== SUMMARY ==========")
print(df_res.to_string(index=False))

metric_cols = [
    "int_window_loss", "f1", "precision", "recall", "specificity", "accuracy", "logloss"
]
summary = df_res[metric_cols].agg(["mean", "std"]).T
summary.columns = ["mean", "std"]
print("\nMean / Std across seeds:")
print(summary)
summary.to_csv("NEW_SPLIT/bilstm_seed_results_summary.csv")
print(f"\nPer-seed rows: {results_csv_path}")
print(f"Summary:       NEW_SPLIT/bilstm_seed_results_summary.csv")