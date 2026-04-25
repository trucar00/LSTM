import pandas as pd
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold

#read_cols = "mmsi","trajectory_id","date_time_utc","lon","lat", "label","dt","dist_to_prev","log_dt","log_dist","cog_sin","cog_cos","y","speed_calc_ms", "ra_accel","ra_jerk","ra_dcog"
df = pd.read_csv("first_50_feats.csv")

WINDOW = 128
STRIDE = 64

FEATURES = [
    "cog_sin",
    "cog_cos",
    "speed_calc_ms",
    "ra_accel",
    "ra_jerk",
    "log_dist",
    "ra_dcog",
    "log_dt",
]

def create_windows(df, features, window=128, stride=64):
    X, y, groups = [], [], []

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

    return np.array(X), np.array(y), np.array(groups)

def evaluate_subset(y_true, y_pred, name=""):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    print(f"\n{name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    return metrics

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

X, y, groups = create_windows(
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

kf = StratifiedGroupKFold(n_splits=5, random_state=1, shuffle=True)
cv_scores_total = []

i=1

for train_index, test_index in kf.split(X, y, groups=groups):
    print(f"Fold: {i} ===================================================")
    X_train_full, X_test = X[train_index], X[test_index]
    y_train_full, y_test = y[train_index], y[test_index]
    groups_train_full, groups_test = groups[train_index], groups[test_index]

    inner_split = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42 + i
    )

    train_idx, val_idx = next(
        inner_split.split(X_train_full, y_train_full, groups_train_full)
    )

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
    
    groups_train = groups_train_full[train_idx]
    groups_val = groups_train_full[val_idx]
    
    print("Train positive rate:", y_train.mean())
    print("Val positive rate:  ", y_val.mean())
    print("Test positive rate: ", y_test.mean())

    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)

    X_val = scaler.transform(
        X_val.reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape)

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight = dict(zip(classes, weights))

    tf.keras.backend.clear_session()
    model = build_model((X_train.shape[1], X_train.shape[2]))


    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        #class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1
    )

    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)
    
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Total
    total_metrics = evaluate_subset(y_test, y_pred, name=f"Fold {i} TOTAL")
    total_metrics["fold"] = i
    cv_scores_total.append(total_metrics)

    i += 1


cv_scores_total = pd.DataFrame(cv_scores_total)
print("\nCV results")
print(cv_scores_total)
print("\nMean CV results")
print(cv_scores_total.mean(numeric_only=True))