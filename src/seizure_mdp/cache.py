"""Per-fold cache builder (notebook Cell 12B / build_patient_frozen_with_streams).

Training dominates runtime but is independent of the MDP cost parameters, so
each LOSO fold is trained once and everything the decision layer needs is
stored:

    fold_id, test_seizure, t_test, t_FAR, test_seizures,
    P_smooth_test, P_smooth_FAR      smoothed fused probability streams
    P_fusion_test, P_fusion_FAR      pre-smoothing (for H sweeps)
    TRAIN_STREAMS                    per-recording smoothed streams of the
                                     training recordings (T_base source)
    T_base                           24x24 training-stream estimate (bins in force)
    T                                the same expanded to the legacy
                                     (4 actions, 96, 96) layout of the
                                     original caches, so T_base_from_cache works
    base_coverage, fusion_weights, val_aucs, seed, level_bins

Each fold is seeded with SEED + 1000*fold_id before training (and
SEED + 1000*fold_id + 500 before the regeneration pass), so its classifiers
depend on the fold alone, never on execution history.

The cached ``T`` / ``T_base`` are discretised with the bins that were in force
when every notebook cache was built, ``CACHE_LEVEL_BINS = [0.30, 0.60, 0.85]``
(the notebook's module-level ``LEVEL_BINS``), NOT the adopted decision-layer
bins in ``config['level_bins']``; the adopted bins only enter at evaluation
time, where ``T`` is re-estimated from ``TRAIN_STREAMS`` (``T_source='streams'``).
"""
import os
import pickle
import time

import numpy as np

from .data import build_fold_data
from .models import (seed_everything, get_device, inner_split, fit_three_classifiers,
                     predict_three, train_classifiers_with_val, compute_fusion_weights,
                     apply_fusion, history_buffer)
from .mdp import T_base_from_streams, features, NB

# Bins used for the cached transition matrix of every notebook cache
# (notebook Cell 10A: LEVEL_BINS = CONFIG['level_bins'] = [0.30, 0.60, 0.85]).
CACHE_LEVEL_BINS = (0.30, 0.60, 0.85)


def base_coverage(streams, bins, tau=0.05, dur_thr=3):
    """Number of the 24 base states observed as a transition source."""
    seen = np.zeros(NB, bool)
    for st in streams:
        if len(st) < 2:
            continue
        lv, tr, du = features(st, bins, tau, dur_thr)
        seen[np.unique((lv * 6 + tr * 2 + du)[:-1])] = True
    return int(seen.sum())


# Legacy 4-mode / 4-action mode table (Idle, EnhMonitor, SoftRefr, HardRefr) used
# to lay out ``T`` exactly as the original caches do.
_MODE_T_LEGACY = np.array([[0, 0, 2, 3],
                           [1, 1, 2, 3],
                           [2, 2, 2, 3],
                           [3, 3, 3, 3]], dtype=int)


def build_transition_matrix_legacy(T_base):
    """Expand T_base (24x24) to the original (4, 96, 96) layout."""
    T = np.zeros((4, 96, 96))
    for a in range(4):
        for s in range(96):
            b, m = divmod(s, 4)
            m_next = int(_MODE_T_LEGACY[a][m])
            for b2 in range(NB):
                T[a, s, b2 * 4 + m_next] = T_base[b, b2]
    return T


def regenerate_training_smoothed_streams(fold, fold_data, seg_eval, weights, config,
                                         device=None, seed=42):
    """Reconstruct the fold's classifiers (same inner split, same seed) and
    score each TRAINING recording separately, in temporal order, at 0 %
    overlap. Returns a list of smoothed streams (one per recording). Test and
    FAR recordings are never touched."""
    Xtr, ytr = fold_data['X_train'], fold_data['y_train']
    tr_idx, _ = inner_split(len(Xtr), config, seed)
    models = fit_three_classifiers(Xtr[tr_idx], ytr[tr_idx], config, device, seed)
    streams = []
    for fname in fold['train_seizures'] + fold['interictal_train']:
        if fname not in seg_eval:
            continue
        W = seg_eval[fname]['windows']
        if len(W) == 0:
            continue
        P = predict_three(models, W, config, device)
        streams.append(history_buffer(apply_fusion(P, weights), config['history_H']))
    return streams


def build_fold_cache(folds, seg_train, seg_eval, config, cache_path=None, force=False,
                     keep_streams=True, verbose=True, level_bins=None):
    """Train once per fold and cache everything the MDP layer needs.

    Parameters
    ----------
    folds, seg_train, seg_eval : from data.build_loso_folds / segment_patient
                                 (folds are built from seg_eval, as in the notebook)
    config : flat config (see config.load_config); uses config['seed'],
             config['trend_threshold'], config['duration_threshold'], ...
    cache_path : pickle to read (if present and not force) / write
    keep_streams : store TRAIN_STREAMS (needed to re-estimate T under other bins)
    level_bins : bins for the cached T / T_base; defaults to CACHE_LEVEL_BINS,
                 the bins every notebook cache was built with (NOT the adopted
                 decision-layer bins in config['level_bins'])
    """
    if cache_path and os.path.exists(cache_path) and not force:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if verbose:
            print(f"Loaded cache: {len(cache)} folds from {cache_path}")
        return cache

    seed = config.get('seed', 42)
    bins = list(CACHE_LEVEL_BINS if level_bins is None else level_bins)
    tau, dur = config['trend_threshold'], config['duration_threshold']
    device = get_device()
    cache = []
    for fold in folds:
        t0 = time.time()
        if verbose:
            print(f"  Fold {fold['fold_id']} ({fold['test_seizure']}) ...", end=" ", flush=True)

        seed_everything(seed + 1000 * fold['fold_id'])
        fd = build_fold_data(fold, seg_train, seg_eval, config, seed)
        P_te, P_fa, P_va, y_va = train_classifiers_with_val(fd, config, device, seed)
        w, aucs = compute_fusion_weights(P_va, y_va)
        ps_test = history_buffer(apply_fusion(P_te, w), config['history_H'])
        ps_far = history_buffer(apply_fusion(P_fa, w), config['history_H'])

        seed_everything(seed + 1000 * fold['fold_id'] + 500)
        streams = regenerate_training_smoothed_streams(fold, fd, seg_eval, w, config, device, seed)
        T_base = T_base_from_streams(streams, bins, tau, dur)
        coverage = base_coverage(streams, bins, tau, dur)

        entry = {
            'fold_id': fold['fold_id'],
            'test_seizure': fold['test_seizure'],
            't_test': fd['t_test'], 't_FAR': fd['t_FAR'],
            'test_seizures': fd['test_seizures'],
            'P_smooth_test': ps_test, 'P_smooth_FAR': ps_far,
            'P_fusion_test': apply_fusion(P_te, w), 'P_fusion_FAR': apply_fusion(P_fa, w),
            'T_base': T_base,
            'T': build_transition_matrix_legacy(T_base),
            'base_coverage': coverage,
            'fusion_weights': w, 'val_aucs': aucs,
            'seed': seed + 1000 * fold['fold_id'],
            'level_bins': bins, 'tau': tau,
        }
        if keep_streams:
            entry['TRAIN_STREAMS'] = streams
        cache.append(entry)
        if verbose:
            print(f"done [{time.time() - t0:.0f}s]")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        if verbose:
            print(f"Cache written to {cache_path}")
    return cache


def load_master(cache_dir, filename='MASTER_complete_session.pkl'):
    """Load the released session pickle (all patient caches + adopted config)."""
    with open(os.path.join(cache_dir, filename), 'rb') as f:
        return pickle.load(f)
