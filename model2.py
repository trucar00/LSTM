import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import glob

WINDOW = 256
STRIDE = 128

files = sorted(glob.glob("chunks/chunk_*.parquet"))
dfs = [pd.read_parquet(f) for f in files]

df = pd.concat(dfs, ignore_index=True)
df.to_parquet("ais_conf_labeled_features_01_04_all_gear.parquet")

print("Recombined!")

if os.path.exists("fishing_bilstm_best.pt"):
    os.remove("fishing_bilstm_best.pt")

FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk", "log_dist", "ra_dcog", "log_dt"]

#df = pd.read_parquet("ais_conf_labeled_features_01_04_all_gear.parquet")

# Split into train test and validation set by mmsi so that no vessel appear in both.
rng = np.random.default_rng(42)
mmsis = df["mmsi"].unique().copy()
rng.shuffle(mmsis)
n = len(mmsis)
train_mmsi = set(mmsis[: int(0.70 * n)])
val_mmsi   = set(mmsis[int(0.70 * n) : int(0.85 * n)])
test_mmsi  = set(mmsis[int(0.85 * n) :])
print(f"Train {len(train_mmsi)} | Val {len(val_mmsi)} | Test {len(test_mmsi)}")
print("All sample weights:")
print(df["sample_weight"].value_counts())

print("Labeled y distribution:")
print(df[df["sample_weight"] == 1]["y_train"].value_counts())

print(df[FEATURES].describe().T[["mean", "std", "min", "max"]])
print(df[FEATURES].abs().max().sort_values(ascending=False))

# Fit normalization on TRAIN ONLY
train_df = df[df["mmsi"].isin(train_mmsi)]
print(train_df.groupby(["report", "sample_weight", "y_train"]).size())

mu    = train_df[FEATURES].mean()
sigma = train_df[FEATURES].std().replace(0, 1)
for col in FEATURES:
    df[col] = (df[col] - mu[col]) / sigma[col]

#print(df[FEATURES].isna().sum())
#print(np.isinf(df[FEATURES]).sum())
print(df["y_train"].value_counts(dropna=False))

print(df[FEATURES].describe().T[["mean", "std", "min", "max"]])
print(df[FEATURES].abs().max().sort_values(ascending=False))

df["ra_accel"] = df["ra_accel"].clip(-5, 5)
df["ra_jerk"]  = df["ra_jerk"].clip(-5, 5)
df["ra_dcog"]  = df["ra_dcog"].clip(-5, 5)

def make_windows(traj_df, FEATURES, window=WINDOW, stride=STRIDE):
    X_all = traj_df[FEATURES].to_numpy(dtype=np.float32)
    y_all = traj_df["y_train"].to_numpy(dtype=np.float32)
    w_all = traj_df["sample_weight"].to_numpy(dtype=np.float32)

    n = len(traj_df)
    if n < 8:
        return

    for start in range(0, max(1, n - window + 1), stride):
        end = start + window

        x = X_all[start:end]
        y = y_all[start:end]
        w = w_all[start:end]

        L = len(x)

        if L < window:
            pad = window - L
            x = np.vstack([x, np.zeros((pad, x.shape[1]), dtype=np.float32)])
            y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
            w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])

        yield x, y, w

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
            nn.Linear(64, 1),   # binary logit per timestep
        )

    def forward(self, x):
        # x: (B, T, F)
        h, _ = self.lstm(x)          # (B, T, 2*hidden)
        logits = self.head(h).squeeze(-1)  # (B, T)
        return logits
    
# 2. Build windows for a split
def build_split(df, mmsi_set, FEATURES):
    Xs, Ys, Ms = [], [], []
    sub = df[df["mmsi"].isin(mmsi_set)]
    for _, traj in tqdm(sub.groupby("trajectory_id", sort=False),
                        desc="windowing"):
        traj = traj.sort_values("date_time_utc")
        for x, y, m in make_windows(traj, FEATURES):
            Xs.append(x); Ys.append(y); Ms.append(m)
    if not Xs:
        return None
    X = torch.from_numpy(np.stack(Xs))
    Y = torch.from_numpy(np.stack(Ys)).float()
    W = torch.from_numpy(np.stack(Ms)).float()
    return TensorDataset(X, Y, W)

train_ds = build_split(df, train_mmsi, FEATURES)
val_ds   = build_split(df, val_mmsi,   FEATURES)
test_ds  = build_split(df, test_mmsi,  FEATURES)
print(f"Windows — train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")

# 3. DataLoaders
BATCH = 128
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=True)

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
                      hidden=128, n_layers=2, dropout=0.3).to(device)

train_labeled = train_df[train_df["sample_weight"] == 1]
neg = (train_labeled["y_train"] == 0).sum()
pos = (train_labeled["y_train"] == 1).sum()

pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)
print("pos_weight:", pos_weight.item())

bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

def masked_loss(logits, y, mask):
    m = mask.float()
    per = bce(logits, y)
    return (per * m).sum() / m.sum().clamp_min(1.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2)

# ------------------------------------------------------------------
# Epoch runner — per-message metrics (padding ignored), NaN-safe
# ------------------------------------------------------------------
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
best_val = float("inf")
for epoch in range(1, 20):
    tr = run_epoch(train_loader, train=True)
    vl = run_epoch(val_loader,   train=False)
    scheduler.step(vl[0])
    print(f"Ep{epoch:02d} | train loss {tr[0]:.4f} f1 {tr[3]:.3f} | "
          f"val loss {vl[0]:.4f} p {vl[1]:.3f} r {vl[2]:.3f} f1 {vl[3]:.3f} acc {vl[4]:.3f}")
    if vl[0] < best_val:
        best_val = vl[0]
        torch.save(model.state_dict(), "bilstm_best_IDUN_all_gear_01_04.pt")

# ------------------------------------------------------------------
# Final test
# ------------------------------------------------------------------
import os
if os.path.exists("bilstm_best_IDUN_all_gear_01_04.pt"):
    model.load_state_dict(torch.load("bilstm_best_IDUN_all_gear_01_04.pt"))
te = run_epoch(test_loader, train=False)
print(f"TEST | loss {te[0]:.4f}  p {te[1]:.3f}  r {te[2]:.3f}  f1 {te[3]:.3f}  acc {te[4]:.3f}")