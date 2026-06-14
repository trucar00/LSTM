

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

# Load tuned parameters
tuned_params_path = Path("tuning_temporal/best_params_BiLSTM_2023_train_2024_val.json")

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
    N_LAYERS = 2
    HIDDEN   = 64
    DENSE    = 128
    DROPOUT  = 0.326
    BATCH    = 64
    LR       = 9.27e-4
    #exit()

BASE = "three_months/feats_new_rule_bilstm"

# TRAIN: all of 2023
TRAIN_FILES = [
    f"{BASE}/2023_1_3_feats.parquet",     # Q1 2023
    f"{BASE}/2023_4_6_feats.parquet",     # Q2 2023
    f"{BASE}/2023_7_9_feats.parquet",     # Q3 2023
    f"{BASE}/2023_10_12_feats.parquet",   # Q4 2023
]

# VALIDATION and TEST on 2024. We have MMSIS for validation and MMSIS for testing
VAL_TEST_FILES = [
    f"{BASE}/2024_1_3_feats.parquet",     # Q1 2024
    f"{BASE}/2024_4_6_feats.parquet",     # Q2 2024
    f"{BASE}/2024_7_9_feats.parquet",     # Q3 2024
    f"{BASE}/2024_10_12_feats.parquet",   # Q4 2024
]


BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt"]

SEASON_FEATURES = ["month_sin", "month_cos"]

FEATURES = BASE_FEATURES + SEASON_FEATURES

SEEDS = [0, 1]
MAX_EPOCHS = 15
PATIENCE = 3

FOLDER = "training_temporal/"
TAG = "all_2023_bilstm"

def all_mmsis_in(files):
    s = set()
    for f in files:
        s.update(pd.read_parquet(f, columns=["mmsi"])["mmsi"].unique())
    return s

def get_val_test_mmsis(test_or_val, path="../split_mmsis_val_test.csv"):
    split_df = pd.read_csv(path)
    return set(split_df.loc[split_df["split"] == test_or_val, "mmsi"])
 
# All vessels in each quarter (no MMSI split -- the split is by TIME).
train_mmsi = all_mmsis_in(TRAIN_FILES)
val_mmsi = get_val_test_mmsis(test_or_val="validation")
test_mmsi = get_val_test_mmsis(test_or_val="test")
print(f"Train (all 2023) vessels: {len(train_mmsi)} | Val (2024) vessels: {len(val_mmsi)} | Test (2024) vessels: {len(test_mmsi)}")

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
    needed_cols = ["date_time_utc"] + BASE_FEATURES
    for f in TRAIN_FILES:
        df = pd.read_parquet(f, columns=needed_cols)
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

train_ds = AISWindowDataset(TRAIN_FILES, train_mmsi, FEATURES, mu, sigma,
                            shuffle_files=True, stride=STRIDE, window=WINDOW)
val_ds   = AISWindowDataset(VAL_TEST_FILES, val_mmsi, FEATURES, mu, sigma,
                            stride=STRIDE, window=WINDOW)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False,
                          num_workers=0,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=True)
val_loader  = DataLoader(val_ds,  batch_size=BATCH, shuffle=False,
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
for f in TRAIN_FILES:
    df_tmp = pd.read_parquet(f, columns=cols_w)
    #df_tmp = df_tmp[df_tmp["mmsi"].isin(train_mmsi)].copy()
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
# Pre-normalize the 2025 testernal test file ONCE
# (mu/sigma are seed-independent, so this is safe)
# ============================================================

test_mmsi_list = list(test_mmsi)

print(f"Preparing test set: {VAL_TEST_FILES}")

dfs = []
for f in VAL_TEST_FILES:
    df_part = pd.read_parquet(
        f,
        engine="pyarrow",
        filters=[("mmsi", "in", test_mmsi_list)]
    )
    dfs.append(df_part)

df_test = pd.concat(dfs, ignore_index=True)

# Load the test set for finding the loss


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

lens = df_test.groupby("trajectory_id").size()
print((lens < WINDOW).mean(), "of trajectories shorter than WINDOW")


def predict_and_score_external(model):
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
                x = X_all[start:end]          # length L, leave it short
                L = len(x)
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

    df["pred_fishing"] = (df["p_fishing"] > 0.5).astype(int)

    eval_mask = (
            (df["sample_weight"] == 1)
            & df["p_fishing"].notna()
        )
    

    y_true = df.loc[eval_mask, "y_train"].to_numpy(dtype=int)
    y_prob = df.loc[eval_mask, "p_fishing"].to_numpy()

    test_logloss = log_loss(
        y_true,
        y_prob,
        labels=[0, 1],
    )

    # True positives (TP) of labeled
    pred_pos_df = df[df["pred_fishing"] == 1]
    tp = int((pred_pos_df["report"] == "fishing").sum())

    # False positives (FP) of labeled
    fp = int((pred_pos_df["report"] == "conf_no_fishing").sum())

    # True negatives (TN) of labeled
    pred_neg_df = df[df["pred_fishing"] == 0]
    tn = int((pred_neg_df["report"] == "conf_no_fishing").sum())

    # False negatives (FN) of labeled
    fn = int((pred_neg_df["report"] == "fishing").sum())

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

    # Number of predictions for each class
    n_pred_fish = int((df["pred_fishing"] == 1).sum())
    n_pred_no_fish = int((df["pred_fishing"] == 0).sum())

    # Number of reports for each class
    n_reported_fish = int((df["report"] == "fishing").sum())
    n_reported_conf_no_fish = int((df["report"] == "conf_no_fishing").sum())

    # Number of unknowns
    unknown_df = df[df["report"] == "unknown"]
    n_unknown = int(len(unknown_df))

    n_pred_fish_of_unknown = int((unknown_df["pred_fishing"] == 1).sum())
    n_pred_no_fish_of_unknown = int((unknown_df["pred_fishing"] == 0).sum())

    del df
    gc.collect()


    return {
        "test_tp":                  tp,
        "test_fp":                  fp,
        "test_tn":                  tn,
        "test_fn":                  fn,
        "test_accuracy":            accuracy,
        "test_recall":              recall,
        "test_specificity":         specificity,
        "test_precision":           precision,           
        "test_f1":                  f1,
        "test_loss":                test_logloss,
        "test_n_pred_fish":         n_pred_fish,
        "test_n_pred_no_fish":      n_pred_no_fish,
        "test_n_reported_fish":     n_reported_fish,
        "test_n_reported_no_fish":  n_reported_conf_no_fish,
        "test_n_unknowns":          n_unknown,
        "test_n_pred_fish_of_unknown": n_pred_fish_of_unknown,
        "test_n_pred_no_fish_of_unknown": n_pred_no_fish_of_unknown
    }


# ============================================================
# Multi-seed loop
# ============================================================

results_csv_path = "multi_seeds_results/BiLSTM_temporal_train_2023_val_0.5_2024_test_0.5_2024.csv"

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

    model_name = f"models/seed/model_bilstm_seed{seed}_temp_split.pt"
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
            f"training_stats/training_history_BiLSTM_seed{seed}_temp_split.csv", index=False
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
    
    # testernal 2025_1_3 test — the metric that matters
    test = predict_and_score_external(model)
    print(f"[seed {seed}] TEST on test vessels from 2024 | "
          f"precision {test['test_precision']:.3f} "
          f"recall {test['test_recall']:.3f} "
          f"specificity {test['test_specificity']:.3f} "
          f"f1 {test['test_f1']:.3f} "
          f"accuracy {test['test_accuracy']:.3f} ")

    row = {
        "seed": seed,
        "best_val_loss": best_val,
        "epochs_trained": len(history),
        **test,
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
    "test_loss", "test_f1", "test_precision", "test_recall", "test_specificity", "test_accuracy",
]
summary = df_res[metric_cols].agg(["mean", "std"]).T
summary.columns = ["mean", "std"]
print("\nMean / Std across seeds:")
print(summary)
summary.to_csv("multi_seeds_results/BiLSTM_seed_results_summary.csv")
print(f"\nPer-seed rows: {results_csv_path}")
print(f"Summary:       multi_seeds_results/BiLSTM_seed_results_summary.csv")