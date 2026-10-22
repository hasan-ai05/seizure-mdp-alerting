"""Event-level clinical evaluation of any decision layer on cached folds.

A fold cache entry holds (see cache.py): P_smooth_test, t_test, test_seizures,
P_smooth_FAR, t_FAR. The FAR stream is a concatenation of interictal
recordings; the refractory environment restarts at every recording boundary
(detected where the window time does not increase).
"""
from math import comb, exp

import numpy as np

from . import mdp as _mdp
from .baselines import rollout_threshold, BASELINES
from .mdp import rollout_mdp, solve_fold


def far_segments(t_far):
    """Boundaries of the recordings concatenated in the FAR stream."""
    n = len(t_far)
    return [0] + [i for i in range(1, n) if t_far[i] <= t_far[i - 1]] + [n]


def score(fold_cache, alert_fn, SOP=1800, SPH=300, window_sec=5):
    """(hit, FAR per hour) of ``alert_fn`` (stream -> bool alerts) on one fold.

    hit : >= 1 alert inside [onset - SOP - SPH, onset - SPH)
    FAR : alerts over the FAR recordings (each rolled out independently)
          divided by the total FAR duration in hours
    """
    on = fold_cache['test_seizures'][0]['start']
    tt = np.asarray(fold_cache['t_test'])
    al = alert_fn(fold_cache['P_smooth_test'])
    valid = al & (tt >= on - SOP - SPH) & (tt < on - SPH)
    hit = bool(np.any(valid))
    pf, tf = fold_cache['P_smooth_FAR'], fold_cache['t_FAR']
    bd = far_segments(tf)
    nfa = sum(int(alert_fn(pf[bd[j]:bd[j + 1]]).sum()) for j in range(len(bd) - 1))
    hours = len(pf) * window_sec / 3600.0
    return hit, nfa / hours if hours > 0 else 0.0


def warning_time(fold_cache, alert_fn, SOP=1800, SPH=300):
    """Seconds between the first valid alert and onset (None if missed)."""
    on = fold_cache['test_seizures'][0]['start']
    tt = np.asarray(fold_cache['t_test'])
    al = alert_fn(fold_cache['P_smooth_test'])
    idx = np.nonzero(al & (tt >= on - SOP - SPH) & (tt < on - SPH))[0]
    return float(on - tt[idx.min()]) if len(idx) else None


def chance_p_value(n_hit, n_folds, far_mean, SOP=1800):
    """Winterhalder / Truong unspecific-random-predictor test.

    P0 = 1 - exp(-FAR * SOP)  is the chance of a random alert landing in a
    SOP; the p-value is the binomial tail of >= n_hit hits out of n_folds.
    """
    P0 = 1.0 - exp(-far_mean * SOP / 3600.0)
    return sum(comb(n_folds, k) * P0 ** k * (1 - P0) ** (n_folds - k)
               for k in range(n_hit, n_folds + 1)), P0


def summarize(per_fold, SOP=1800):
    """Aggregate [(hit, far), ...] -> dict with sens, FAR mean/std, p_chance."""
    hits = sum(int(h) for h, _ in per_fold)
    fars = np.array([f for _, f in per_fold], float)
    n = len(per_fold)
    p, P0 = chance_p_value(hits, n, fars.mean(), SOP)
    return {'n_hit': hits, 'n_folds': n, 'sensitivity': hits / n,
            'far_mean': float(fars.mean()), 'far_std': float(fars.std()),
            'far_vec': fars, 'p_chance': p, 'P0': P0}


def evaluate_mdp(patient_cache, bins, base, alpha_fa, T_source='cache',
                 SOP=1800, SPH=300, window_sec=5):
    """LOSO evaluation of the MDP on one patient's fold cache."""
    per, pols = [], []
    for c in patient_cache:
        _, pol = solve_fold(c, bins, base, alpha_fa, T_source)
        pols.append(pol)
        fn = lambda p, pol=pol: rollout_mdp(p, pol, bins, base['soft_arp'], base['hard_arp'],
                                            base['trend_threshold'], base['duration_threshold'])
        per.append(score(c, fn, SOP, SPH, window_sec))
    out = summarize(per, SOP)
    out['policies'] = pols
    out['policy_counts'] = [int(np.sum(pols[0] == a)) for a in range(_mdp.NA)]
    return out


def evaluate_baseline(patient_cache, kw, soft_w=60, hard_w=360,
                      SOP=1800, SPH=300, window_sec=5):
    """LOSO evaluation of one threshold-type arm (kwargs of rollout_threshold)."""
    fn = lambda p: rollout_threshold(p, soft_w=soft_w, hard_w=hard_w, **kw)
    return summarize([score(c, fn, SOP, SPH, window_sec) for c in patient_cache], SOP)


def best_baseline_at_sensitivity(patient_cache, n_hit, baselines=BASELINES,
                                 soft_w=60, hard_w=360, SOP=1800, SPH=300, window_sec=5):
    """Lowest-FAR arm among those reaching at least ``n_hit`` hits."""
    arms = []
    for name, kw in baselines:
        r = evaluate_baseline(patient_cache, kw, soft_w, hard_w, SOP, SPH, window_sec)
        if r['n_hit'] >= n_hit:
            arms.append((r['far_mean'], name, r['n_hit'], r['n_folds']))
    return min(arms) if arms else (float('nan'), '-', 0, len(patient_cache))


def screen_patient(patient_cache, peak_threshold=0.60, catch_fraction=0.60,
                   auc_threshold=0.70, SOP=1800, SPH=300):
    """Classifier-quality gate applied before the decision layer.

    A seizure is 'caught' if the smoothed stream peaks >= peak_threshold inside
    its preictal window; AUC is preictal windows vs all FAR windows.
    """
    from sklearn.metrics import roc_auc_score
    caught = total = 0
    pre, far = [], []
    for c in patient_cache:
        ps, tt = c['P_smooth_test'], np.asarray(c['t_test'])
        on = c['test_seizures'][0]['start']
        win = np.nonzero((tt >= on - SOP - SPH) & (tt < on - SPH))[0]
        if len(win):
            total += 1
            caught += int(ps[win].max() >= peak_threshold)
            pre.extend(ps[win])
        far.extend(c['P_smooth_FAR'])
    pre, far = np.array(pre), np.array(far)
    y = np.concatenate([np.ones(len(pre)), np.zeros(len(far))])
    auc = float(roc_auc_score(y, np.concatenate([pre, far])))
    cf = caught / max(total, 1)
    return {'auc': auc, 'catch': caught, 'total': total, 'catch_fraction': cf,
            'valid': (cf >= catch_fraction) and (auc >= auc_threshold)}
