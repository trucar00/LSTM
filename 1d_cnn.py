import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("first_50_feats.csv")

WINDOW = 128
STRIDE = 64

FEATURES = [
    "cog_sin", "cog_cos",
    "speed_calc_ms",
    "ra_accel", "ra_jerk",
    "log_dist", "ra_dcog", "log_dt",
]


def create_windows(df, features, window=128, stride=64):
    X, y, groups = [], [], []
    meta = []

    df = df.sort_values(["mmsi", "trajectory_id", "date_time_utc"]).copy()

    for traj_id, d in df.groupby("trajectory_id", sort=False):
        d = d.reset_index(drop=True)

        if len(d) < window:
            continue

        feat_values = d[features].to_numpy(dtype=np.float32)
        labels = d["y"].to_numpy(dtype=np.int64)
        mmsi = d["mmsi"].iloc[0]

        for start in range(0, len(d) - window + 1, stride):
            end = start + window
            mid = start + window // 2

            X.append(feat_values[start:end])
            y.append(labels[mid])
            groups.append(mmsi)

            meta.append({
                "mmsi": mmsi,
                "trajectory_id": traj_id,
                "start_idx": start,
                "mid_idx": mid,
                "end_idx": end,
                "mid_time": d.loc[mid, "date_time_utc"],
                "mid_lon": d.loc[mid, "lon"],
                "mid_lat": d.loc[mid, "lat"],
                "true_y": labels[mid],
            })

    return np.array(X), np.array(y), np.array(groups), pd.DataFrame(meta)


def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalMaxPooling1D(),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


def evaluate(y_true, y_pred, name):
    print(f"\n{name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# ------------------------------------------------------------------
# Create windows
# ------------------------------------------------------------------

X, y, groups, meta = create_windows(
    df,
    features=FEATURES,
    window=WINDOW,
    stride=STRIDE,
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)
print("Positive rate:", y.mean())
print("Number of vessels:", len(np.unique(groups)))


# ------------------------------------------------------------------
# One grouped split: train_full / test
# ------------------------------------------------------------------

outer_split = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=1,
)

train_full_idx, test_idx = next(
    outer_split.split(X, y, groups)
)

X_train_full, X_test = X[train_full_idx], X[test_idx]
y_train_full, y_test = y[train_full_idx], y[test_idx]
groups_train_full, groups_test = groups[train_full_idx], groups[test_idx]

meta_train_full = meta.iloc[train_full_idx].reset_index(drop=True)
meta_test = meta.iloc[test_idx].reset_index(drop=True)


# ------------------------------------------------------------------
# One grouped split: train / val
# ------------------------------------------------------------------

inner_split = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_idx, val_idx = next(
    inner_split.split(X_train_full, y_train_full, groups_train_full)
)

X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
groups_train, groups_val = groups_train_full[train_idx], groups_train_full[val_idx]

print("\nSplit summary")
print("Train vessels:", len(np.unique(groups_train)))
print("Val vessels:  ", len(np.unique(groups_val)))
print("Test vessels: ", len(np.unique(groups_test)))

print("Train positive rate:", y_train.mean())
print("Val positive rate:  ", y_val.mean())
print("Test positive rate: ", y_test.mean())


# ------------------------------------------------------------------
# Scale using train only
# ------------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])
).reshape(X_train.shape)

X_val = scaler.transform(
    X_val.reshape(-1, X_val.shape[-1])
).reshape(X_val.shape)

X_test = scaler.transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)


# ------------------------------------------------------------------
# Train model
# ------------------------------------------------------------------

tf.keras.backend.clear_session()

model = build_model(
    input_shape=(X_train.shape[1], X_train.shape[2])
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)


# ------------------------------------------------------------------
# Optional: tune threshold on validation set
# ------------------------------------------------------------------

val_prob = model.predict(X_val, verbose=0).ravel()

thresholds = np.linspace(0.1, 0.9, 81)

best_threshold = 0.5
best_f1 = -1

for t in thresholds:
    val_pred = (val_prob >= t).astype(int)
    f1 = f1_score(y_val, val_pred, zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\nBest threshold from val:", best_threshold)
print("Best val F1:", best_f1)


# ------------------------------------------------------------------
# Evaluate on held-out test set
# ------------------------------------------------------------------

test_prob = model.predict(X_test, verbose=0).ravel()
test_pred = (test_prob >= best_threshold).astype(int)

evaluate(y_test, test_pred, name="HELD-OUT TEST")


# ------------------------------------------------------------------
# Save predictions for plotting vessels
# ------------------------------------------------------------------

test_results = meta_test.copy()
test_results["y_prob"] = test_prob
test_results["y_pred"] = test_pred
test_results["y_true"] = y_test

test_results.to_csv("cnn_test_predictions.csv", index=False)

print("\nSaved predictions to cnn_test_predictions.csv")
print(test_results.head())