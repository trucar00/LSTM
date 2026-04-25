import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os

# ── Time-based window config ────────────────────────────────────────────────
WINDOW   = pd.Timedelta(hours=1)
STRIDE   = pd.Timedelta(minutes=30)
MAX_LEN  = 256   # max AIS points per window (pad/truncate to this)

FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel",
            "ra_jerk", "log_dist", "ra_dcog", "log_dt"]

if os.path.exists("fishing_bilstm_best.pt"):
    os.remove("fishing_bilstm_best.pt")

df = pd.read_csv("first_50_feats.csv")
df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])   # ← ensure datetime

# ── Train / Val / Test split by MMSI ────────────────────────────────────────
rng = np.random.default_rng(42)
mmsis = df["mmsi"].unique().copy()
rng.shuffle(mmsis)
n = len(mmsis)
train_mmsi = set(mmsis[: int(0.70 * n)])
val_mmsi   = set(mmsis[int(0.70 * n) : int(0.85 * n)])
test_mmsi  = set(mmsis[int(0.85 * n) :])
print(f"Train {len(train_mmsi)} | Val {len(val_mmsi)} | Test {len(test_mmsi)}")

# ── Normalise on train split only ───────────────────────────────────────────
train_df = df[df["mmsi"].isin(train_mmsi)]
mu    = train_df[FEATURES].mean()
sigma = train_df[FEATURES].std().replace(0, 1)
for col in FEATURES:
    df[col] = (df[col] - mu[col]) / sigma[col]

print(df[FEATURES].isna().sum())
print(np.isinf(df[FEATURES]).sum())
print(df["y"].value_counts(dropna=False))
print(df[FEATURES].describe().T[["mean", "std", "min", "max"]])

# ── Time-based sliding window ────────────────────────────────────────────────
def make_windows(traj_df, features, window=WINDOW, stride=STRIDE, max_len=MAX_LEN):
    """
    Slide a fixed-duration window over one trajectory.
    Each yielded window is padded/truncated to max_len rows.
    """
    traj_df = traj_df.sort_values("date_time_utc").reset_index(drop=True)
    times  = traj_df["date_time_utc"]          # already datetime
    X_all  = traj_df[features].to_numpy(dtype=np.float32)
    y_all  = traj_df["y"].to_numpy(dtype=np.int8)

    t_start = times.iloc[0]
    t_end   = times.iloc[-1]

    win_start = t_start
    while win_start <= t_end:
        win_end  = win_start + window
        in_win   = (times >= win_start) & (times < win_end)
        x, y     = X_all[in_win], y_all[in_win]
        L        = len(x)

        if L < 2:                               # skip near-empty windows
            win_start += stride
            continue

        # Truncate if the window somehow contains more than max_len points
        if L > max_len:
            x, y, L = x[:max_len], y[:max_len], max_len

        # Pad tail to max_len
        pad = max_len - L
        if pad:
            x = np.vstack([x, np.zeros((pad, x.shape[1]), dtype=np.float32)])
            y = np.concatenate([y, np.zeros(pad,           dtype=np.int8)])

        mask = np.concatenate([np.ones(L,   dtype=np.bool_),
                               np.zeros(pad, dtype=np.bool_)])
        yield x, y, mask
        win_start += stride


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
        return self.head(h).squeeze(-1)   # (B, T) logits


def build_split(df, mmsi_set, features):
    Xs, Ys, Ms = [], [], []
    sub = df[df["mmsi"].isin(mmsi_set)]
    for _, traj in tqdm(sub.groupby("trajectory_id", sort=False), desc="windowing"):
        for x, y, m in make_windows(traj, features):
            Xs.append(x); Ys.append(y); Ms.append(m)
    if not Xs:
        return None
    X = torch.from_numpy(np.stack(Xs))
    Y = torch.from_numpy(np.stack(Ys)).float()
    M = torch.from_numpy(np.stack(Ms))
    return TensorDataset(X, Y, M)


train_ds = build_split(df, train_mmsi, FEATURES)
val_ds   = build_split(df, val_mmsi,   FEATURES)
test_ds  = build_split(df, test_mmsi,  FEATURES)
print(f"Windows — train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")

# ── DataLoaders ──────────────────────────────────────────────────────────────
BATCH = 128
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=0, pin_memory=False)

# ── Model, loss, optimiser ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
model = FishingBiLSTM(n_features=len(FEATURES),
                      hidden=128, n_layers=2, dropout=0.3).to(device)

pos_weight = torch.tensor([16_443_814 / 6_107_465], device=device)
bce        = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

def masked_loss(logits, y, mask):
    m = mask.float()
    return (bce(logits, y) * m).sum() / m.sum().clamp_min(1.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2)

def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    tot_loss = tot_n = tp = fp = fn = tn = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits   = model(x)
            loss     = masked_loss(logits, y, m)
            if train:
                if not torch.isfinite(loss):
                    optimizer.zero_grad(); continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n      = m.sum().item()
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

best_val = float("inf")
for epoch in range(1, 3):
    tr = run_epoch(train_loader, train=True)
    vl = run_epoch(val_loader,   train=False)
    scheduler.step(vl[0])
    print(f"Ep{epoch:02d} | train loss {tr[0]:.4f} f1 {tr[3]:.3f} | "
          f"val loss {vl[0]:.4f} p {vl[1]:.3f} r {vl[2]:.3f} "
          f"f1 {vl[3]:.3f} acc {vl[4]:.3f}")
    if vl[0] < best_val:
        best_val = vl[0]
        torch.save(model.state_dict(), "fishing_bilstm_best.pt")

if os.path.exists("fishing_bilstm_best.pt"):
    model.load_state_dict(torch.load("fishing_bilstm_best.pt"))
te = run_epoch(test_loader, train=False)
print(f"TEST | loss {te[0]:.4f}  p {te[1]:.3f}  r {te[2]:.3f}  "
      f"f1 {te[3]:.3f}  acc {te[4]:.3f}")

def make_windows_indexed(traj_df, features, window=WINDOW, stride=STRIDE, max_len=MAX_LEN):
    """Like make_windows but also yields the integer row indices of real points."""
    traj_df = traj_df.sort_values("date_time_utc").reset_index(drop=True)
    times  = traj_df["date_time_utc"]
    X_all  = traj_df[features].to_numpy(dtype=np.float32)
    y_all  = traj_df["y"].to_numpy(dtype=np.int8)

    t_start = times.iloc[0]
    t_end   = times.iloc[-1]

    win_start = t_start
    while win_start <= t_end:
        win_end = win_start + window
        in_win  = (times >= win_start) & (times < win_end)
        idxs    = np.where(in_win)[0]          # ← real row indices
        x, y    = X_all[idxs], y_all[idxs]
        L       = len(x)

        if L < 2:
            win_start += stride
            continue

        if L > max_len:
            idxs, x, y, L = idxs[:max_len], x[:max_len], y[:max_len], max_len

        pad  = max_len - L
        if pad:
            x = np.vstack([x, np.zeros((pad, x.shape[1]), dtype=np.float32)])
            y = np.concatenate([y, np.zeros(pad, dtype=np.int8)])

        mask = np.concatenate([np.ones(L, dtype=np.bool_),
                               np.zeros(pad, dtype=np.bool_)])
        yield x, y, mask, idxs          # ← extra yield
        win_start += stride


# ── Run inference and collect per-row predictions ───────────────────────────
model.load_state_dict(torch.load("fishing_bilstm_best.pt"))
model.eval()

test_df = df[df["mmsi"].isin(test_mmsi)].copy()
all_preds = []

with torch.no_grad():
    for _, traj in tqdm(test_df.groupby("trajectory_id", sort=False), desc="predicting"):
        traj = traj.sort_values("date_time_utc").reset_index(drop=True)

        # Accumulate prob scores — a row can appear in multiple overlapping windows
        row_scores = np.zeros(len(traj), dtype=np.float64)
        row_counts = np.zeros(len(traj), dtype=np.int32)

        for x, y, m, idxs in make_windows_indexed(traj, FEATURES):
            x_t    = torch.from_numpy(x).unsqueeze(0).to(device)
            probs  = torch.sigmoid(model(x_t)).squeeze(0).cpu().numpy()
            L      = len(idxs)
            row_scores[idxs] += probs[:L]
            row_counts[idxs] += 1

        # Average score across windows, threshold at 0.5
        avg_prob = np.divide(row_scores, row_counts,
                             out=np.zeros_like(row_scores),
                             where=row_counts > 0)

        traj["pred_prob"]    = avg_prob
        traj["pred_fishing"] = (avg_prob > 0.5).astype(int)
        all_preds.append(traj)

pred_df = pd.concat(all_preds, ignore_index=True)

# Save — include whatever columns you need for plotting
pred_df[["mmsi", "trajectory_id", "date_time_utc",
         "lat", "lon", "y", "pred_prob", "pred_fishing"]].to_csv(
    "test_predictions.csv", index=False
)
print(f"Saved {len(pred_df)} rows to test_predictions.csv")