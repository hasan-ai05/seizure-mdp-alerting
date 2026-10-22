"""Observation layer: three classifiers, AUC-weighted fusion, history buffer.

Lifted from notebook cells 8b / 9 (CNN-TCN, LSTM, Random Forest on PSD
features; early stopping on an inner 20 % validation split; fusion weights
w_i = AUC_i / sum AUC on that split; causal moving average of length H).

torch is imported lazily so that the rest of the package (and the decision
layer) works on a machine without PyTorch / GPU.
"""
import copy
import random

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

_torch = None
_nn = None


def _lazy_torch():
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch, _nn = torch, nn
    return _torch, _nn


def get_device():
    torch, _ = _lazy_torch()
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    """Reset every RNG the training path touches (per-fold determinism)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch, _ = _lazy_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def zscore_downsample(windows, factor):
    """Per-window z-score + temporal downsampling -> (N, C, T/factor) float32."""
    ds = windows[:, :, ::factor].astype(np.float32)
    mean = ds.mean(axis=2, keepdims=True)
    std = ds.std(axis=2, keepdims=True) + 1e-8
    return (ds - mean) / std


def extract_psd(windows, config):
    """Welch band powers (delta, theta, alpha, beta, gamma) x channels. No z-score."""
    fs = config['fs']
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    feats = []
    for w in windows:
        row = []
        for ch in w:
            f, psd = welch(ch, fs=fs, window='hann', nperseg=min(256, len(ch)))
            for lo, hi in bands:
                m = (f >= lo) & (f <= hi)
                row.append(np.mean(psd[m]) if m.any() else 0.0)
        feats.append(row)
    return np.array(feats)


# ---------------------------------------------------------------------------
# Model definitions (built on first use so torch is only needed then)
# ---------------------------------------------------------------------------
def _build_classes():
    torch, nn = _lazy_torch()

    class LSTMNet(nn.Module):
        def __init__(self, n_ch, config):
            super().__init__()
            self.lstm = nn.LSTM(n_ch, 64, batch_first=True)
            self.fc1 = nn.Linear(64, 32)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(config['dropout'])
            self.out = nn.Linear(32, 1)

        def forward(self, x):                       # x: (B, T, C)
            o, _ = self.lstm(x)
            o = self.drop(self.relu(self.fc1(o[:, -1, :])))
            return self.out(o).squeeze(1)

    class TCNBlock(nn.Module):
        def __init__(self, ic, oc, k=3, d=1, drop=0.4):
            super().__init__()
            pad = (k - 1) * d
            self.conv = nn.Conv1d(ic, oc, k, padding=pad, dilation=d)
            self.bn = nn.BatchNorm1d(oc)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(drop)
            self.down = nn.Conv1d(ic, oc, 1) if ic != oc else None

        def forward(self, x):
            o = self.conv(x)[:, :, :x.size(2)]      # causal crop
            o = self.drop(self.relu(self.bn(o)))
            res = x if self.down is None else self.down(x)
            return o + res

    class CNNTCNNet(nn.Module):
        def __init__(self, n_ch, config):
            super().__init__()
            d = config['dropout_cnn']
            self.cnn = nn.Sequential(
                nn.Conv1d(n_ch, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(d),
                nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(d))
            self.tcn = nn.Sequential(
                TCNBlock(64, 64, d=1, drop=d), TCNBlock(64, 64, d=2, drop=d),
                TCNBlock(64, 64, d=4, drop=d), TCNBlock(64, 64, d=8, drop=d))
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(d), nn.Linear(32, 1))

        def forward(self, x):                       # x: (B, C, T)
            o = self.tcn(self.cnn(x))
            o = self.pool(o).squeeze(-1)
            return self.fc(o).squeeze(1)

    return LSTMNet, TCNBlock, CNNTCNNet


def LSTMNet(n_ch, config):
    return _build_classes()[0](n_ch, config)


def TCNBlock(ic, oc, k=3, d=1, drop=0.4):
    return _build_classes()[1](ic, oc, k, d, drop)


def CNNTCNNet(n_ch, config):
    return _build_classes()[2](n_ch, config)


# ---------------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------------
def train_torch_model(model, X, y, config, input_mode, device=None, seed=42):
    """Adam + BCEWithLogits(pos_weight) with early stopping on a 20 % split
    (seeded shuffle); restores the best weights. input_mode is 'lstm' or 'cnn'
    (layout is prepared by the caller)."""
    torch, nn = _lazy_torch()
    from torch.utils.data import DataLoader, TensorDataset
    device = get_device() if device is None else device

    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * config['val_split'])
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xt, yt = torch.FloatTensor(X[tr_idx]), torch.FloatTensor(y[tr_idx])
    Xv, yv = torch.FloatTensor(X[val_idx]).to(device), torch.FloatTensor(y[val_idx]).to(device)

    n_pos = yt.sum().item()
    n_neg = len(yt) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)]).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=config['batch_size'], shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config.get('weight_decay', 0))
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state, patience_ctr = float('inf'), None, 0
    for _ in range(config['epochs']):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()
        if val_loss < best_val:
            best_val, best_state, patience_ctr = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience_ctr += 1
            if patience_ctr >= config['patience']:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_torch(model, X, device=None):
    """Sigmoid probabilities for X."""
    torch, _ = _lazy_torch()
    device = get_device() if device is None else device
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.FloatTensor(X).to(device))).cpu().numpy()


def inner_split(n, config, seed=42):
    """The seeded inner train / validation split shared by training, fusion
    weights and the training-stream regeneration."""
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * config['val_split'])
    return idx[n_val:], idx[:n_val]


def fit_three_classifiers(X_tr, y_tr, config, device=None, seed=42):
    """Train CNN-TCN, LSTM and RF on (X_tr, y_tr). Returns dict of models."""
    device = get_device() if device is None else device
    n_ch = X_tr.shape[1]
    cnn = CNNTCNNet(n_ch, config).to(device)
    cnn = train_torch_model(cnn, zscore_downsample(X_tr, config['cnn_downsample']),
                            y_tr, config, 'cnn', device, seed)
    lstm = LSTMNet(n_ch, config).to(device)
    lstm = train_torch_model(lstm, zscore_downsample(X_tr, config['lstm_downsample']).transpose(0, 2, 1),
                             y_tr, config, 'lstm', device, seed)
    rf = RandomForestClassifier(n_estimators=config['rf_n_estimators'],
                                max_depth=config['rf_max_depth'],
                                class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(extract_psd(X_tr, config), y_tr)
    return {'cnn': cnn, 'lstm': lstm, 'rf': rf}


def predict_three(models, X, config, device=None):
    """Probabilities of the three classifiers on windows X -> dict."""
    device = get_device() if device is None else device
    return {'cnn': predict_torch(models['cnn'], zscore_downsample(X, config['cnn_downsample']), device),
            'lstm': predict_torch(models['lstm'],
                                  zscore_downsample(X, config['lstm_downsample']).transpose(0, 2, 1), device),
            'rf': models['rf'].predict_proba(extract_psd(X, config))[:, 1]}


def train_classifiers_with_val(fold_data, config, device=None, seed=42):
    """Train all three classifiers on the inner training split and return
    probabilities on test, FAR and the inner validation split (+ its labels),
    so fusion weights never see test or FAR data."""
    Xtr, ytr = fold_data['X_train'], fold_data['y_train']
    tr_idx, val_idx = inner_split(len(Xtr), config, seed)
    models = fit_three_classifiers(Xtr[tr_idx], ytr[tr_idx], config, device, seed)
    P_test = predict_three(models, fold_data['X_test'], config, device)
    P_FAR = predict_three(models, fold_data['X_FAR'], config, device)
    P_val = predict_three(models, Xtr[val_idx], config, device)
    return P_test, P_FAR, P_val, ytr[val_idx]


# ---------------------------------------------------------------------------
# Fusion and history buffer
# ---------------------------------------------------------------------------
def compute_fusion_weights(P_val, y_val):
    """w_i = AUC_i / sum_j AUC_j on the validation split (0.5 if one class)."""
    aucs = {}
    for name in ['cnn', 'lstm', 'rf']:
        aucs[name] = 0.5 if len(np.unique(y_val)) < 2 else roc_auc_score(y_val, P_val[name])
    total = sum(aucs.values())
    return {name: aucs[name] / total for name in aucs}, aucs


def apply_fusion(P_dict, weights):
    """Convex combination of the three probability streams."""
    return (weights['cnn'] * P_dict['cnn'] + weights['lstm'] * P_dict['lstm']
            + weights['rf'] * P_dict['rf'])


def history_buffer(P_fusion, H):
    """Causal (trailing) moving average of length H."""
    P_smooth = np.zeros_like(P_fusion)
    for t in range(len(P_fusion)):
        P_smooth[t] = np.mean(P_fusion[max(0, t - H + 1):t + 1])
    return P_smooth
