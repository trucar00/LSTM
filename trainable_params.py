import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
import os
import pickle
import json
from pathlib import Path

USE_TUNED_PARAMS = True
tuned_params_path = Path("tuning_params/best_params_ALL_GEAR.json")

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

BASE_FEATURES = ["cog_sin", "cog_cos", "speed_calc_ms", "ra_accel", "ra_jerk",
                 "log_dist", "ra_dcog", "log_dt", "dist_to_shore_km"]
SEASON_FEATURES = ["month_sin", "month_cos"]
FEATURES = BASE_FEATURES + SEASON_FEATURES



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
    

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_parameter_count(model, model_name):
    total, trainable = count_parameters(model)

    print(f"\n===== {model_name} =====")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    print("\nPer layer:")
    for name, param in model.named_parameters():
        print(f"{name:35s} {param.numel():10,d}  trainable={param.requires_grad}")


n_features = len(FEATURES)

#lstm_model = FishingLSTM(n_features=n_features)
bilstm_model = FishingBiLSTM(n_features=n_features)

#print_parameter_count(lstm_model, "LSTM")
print_parameter_count(bilstm_model, "BiLSTM")