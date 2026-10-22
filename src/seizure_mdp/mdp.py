"""The decision layer: 72-state / 3-action MDP.

State  = Level(4) x Trend(3) x Duration(2) x Mode(3) = 72
         base index b = level*6 + trend*2 + duration   (0..23)
         state index s = b*3 + mode                     (0..71)
Mode   = 0 Idle, 1 Soft-refractory, 2 Hard-refractory
Action = 0 No-Alert, 1 Soft alert, 2 Hard alert

This module is the code of record for the semantics that produced Table 3
(it is the verified ``reproduce_table3.py`` logic with every notebook
global made an explicit parameter). It is pure numpy and needs no GPU.
"""
import numpy as np

NB, NM, NA = 24, 3, 3          # base states, modes, actions
NS = NB * NM                   # 72 states
N_LEVELS, N_TRENDS, N_DURATION = 4, 3, 2

IDLE, SOFT_REFR, HARD_REFR = 0, 1, 2
NO_ALERT, SOFT, HARD = 0, 1, 2
RISING, STABLE, FALLING = 0, 1, 2

# MODE_T[current_mode][action] -> next mode (deterministic, used when solving).
# Refractory is involuntary: No-Alert cannot leave it (the environment timer
# does that at rollout time). Escalation Soft->Hard is allowed, Hard->Soft is not.
MODE_T = np.array([[IDLE,      SOFT_REFR, HARD_REFR],
                   [SOFT_REFR, SOFT_REFR, HARD_REFR],
                   [HARD_REFR, HARD_REFR, HARD_REFR]], dtype=int)

LEVEL_NAMES = ['L0', 'L1', 'L2', 'L3']
TREND_NAMES = ['Rising', 'Stable', 'Falling']
DUR_NAMES = ['Short', 'Long']
MODE_NAMES = ['Idle', 'SoftRefr', 'HardRefr']
ACTION_NAMES = ['No Alert', 'Soft Alert', 'Hard Alert']


# ---------------------------------------------------------------------------
# Features from the smoothed probability stream
# ---------------------------------------------------------------------------
def level(p, bins):
    """Ordinal level of a smoothed probability under 3 bin edges."""
    return 0 if p < bins[0] else 1 if p < bins[1] else 2 if p < bins[2] else 3


def features(ps, bins, tau=0.05, dur_thr=3):
    """Level / Trend / Duration sequences for one stream.

    trend   : 0 rising (dP > tau), 1 stable, 2 falling (dP < -tau); first window stable
    duration: 0 short (run length in current level <= dur_thr), 1 long
    """
    ps = np.asarray(ps, float)
    n = len(ps)
    lv = np.array([level(p, bins) for p in ps], dtype=int)
    tr = np.ones(n, int)
    if n > 1:
        d = np.diff(ps)
        tr[1:] = np.where(d > tau, RISING, np.where(d < -tau, FALLING, STABLE))
    du = np.zeros(n, int)
    run = 1
    for t in range(n):
        run = 1 if t == 0 else (run + 1 if lv[t] == lv[t - 1] else 1)
        du[t] = 0 if run <= dur_thr else 1
    return lv, tr, du


def base_id(l, t, d):
    return l * (N_TRENDS * N_DURATION) + t * N_DURATION + d


def decode_base(b):
    l, r = divmod(b, N_TRENDS * N_DURATION)
    t, d = divmod(r, N_DURATION)
    return l, t, d


def state_id(l, t, d, m):
    return base_id(l, t, d) * NM + m


def decode_state(s):
    b, m = divmod(s, NM)
    l, t, d = decode_base(b)
    return l, t, d, m


def state_name(s):
    l, t, d, m = decode_state(s)
    return f"{LEVEL_NAMES[l]}/{TREND_NAMES[t]}/{DUR_NAMES[d]}/{MODE_NAMES[m]}"


# ---------------------------------------------------------------------------
# Base transition matrix T_base (24 x 24), action independent
# ---------------------------------------------------------------------------
def T_base_from_streams(streams, bins, tau=0.05, dur_thr=3):
    """Maximum-likelihood base transitions counted WITHIN each stream.

    Rows never observed as a source get a uniform distribution.
    """
    cnt = np.zeros((NB, NB))
    for st in streams:
        if len(st) < 2:
            continue
        lv, tr, du = features(st, bins, tau, dur_thr)
        bs = lv * 6 + tr * 2 + du
        for t in range(len(bs) - 1):
            cnt[bs[t], bs[t + 1]] += 1
    T = np.zeros((NB, NB))
    for i in range(NB):
        s = cnt[i].sum()
        T[i] = cnt[i] / s if s > 0 else np.ones(NB) / NB
    return T


def T_base_from_cache(fold_cache):
    """Recover T_base from the cached full matrix.

    The fold caches store ``T`` as (4 actions, 96, 96) built on the historical
    4-mode state space; the base marginal of the No-Alert / Idle rows is the
    training-stream estimate of T_base (made with the bins in force when the
    cache was built, i.e. [0.30, 0.60, 0.85]).
    """
    T = fold_cache['T']
    Tb = np.zeros((NB, NB))
    for b in range(NB):
        row = T[0, b * 4 + 0]
        for b2 in range(NB):
            Tb[b, b2] = row[b2 * 4:(b2 + 1) * 4].sum()
    return Tb / Tb.sum(1, keepdims=True)


def T_full_from_base(Tb):
    """Expand T_base to the (3, 72, 72) action-dependent matrix via MODE_T."""
    T = np.zeros((NA, NS, NS))
    for a in range(NA):
        for s in range(NS):
            b, m = divmod(s, NM)
            m2 = MODE_T[m][a]
            for b2 in range(NB):
                T[a, s, b2 * NM + m2] = Tb[b, b2]
    return T


# ---------------------------------------------------------------------------
# Immediate cost J (72 x 3), ordinal form (the adopted use_prep=False)
# ---------------------------------------------------------------------------
def cost(alpha_fa, base, alpha_md=None):
    """Immediate cost matrix.

    base : dict with 3-element 'w_md', 'w_fa', 'mode_cost' (No-Alert/Soft/Hard,
           Idle/SoftRefr/HardRefr), 'repeat_penalty' and optionally 'alpha_md'.

        tau = 1 - trend  (Rising +1, Stable 0, Falling -1);  d = duration + 1
        A = d * level        (missed-detection driver, ordinal index)
        B = d * (3 - level)  (false-alarm driver)
        J = md_exp[mode] * A * (tau+2) * w_md[a] * alpha_md
          + B * (2-tau) * w_fa[a] * alpha_fa
          + mode_cost[mode]
          + repeat_penalty * w_fa[a] / w_fa[Hard]   if alerting inside a refractory
    The missed-detection term only counts in Idle (md_exp = [1, 0, 0]): once a
    patient-facing alert is standing, not alerting again is not a miss.
    """
    w_md = np.asarray(base['w_md'], float)
    w_fa = np.asarray(base['w_fa'], float)
    mc = np.asarray(base['mode_cost'], float)
    rep = float(base['repeat_penalty'])
    if alpha_md is None:
        alpha_md = float(base.get('alpha_md', 1.0))
    assert len(w_md) == NA and len(w_fa) == NA and len(mc) == NM
    md_exp = np.array([1., 0., 0.])
    J = np.zeros((NS, NA))
    for s in range(NS):
        l, t, d, m = decode_state(s)
        tau, dd = 1 - t, d + 1
        A, B = dd * l, dd * (3 - l)
        for a in range(NA):
            j = md_exp[m] * A * (tau + 2) * w_md[a] * alpha_md \
                + B * (2 - tau) * w_fa[a] * alpha_fa + mc[m]
            if m in (SOFT_REFR, HARD_REFR) and a in (SOFT, HARD):
                j += rep * w_fa[a] / w_fa[HARD]
            J[s, a] = j
    return J


# ---------------------------------------------------------------------------
# Value iteration (cost minimisation)
# ---------------------------------------------------------------------------
def value_iteration(Tb, J, gamma=0.95, eps=1e-6, max_iter=1000):
    """Return (V, policy, n_iter). ``Tb`` is the 24x24 base matrix."""
    T = T_full_from_base(Tb)
    V = np.zeros(NS)
    n_it = 0
    for n_it in range(1, max_iter + 1):
        Q = np.stack([J[:, a] + gamma * (T[a] @ V) for a in range(NA)], 1)
        Vn = Q.min(1)
        if np.max(np.abs(Vn - V)) < eps:
            V = Vn
            break
        V = Vn
    Q = np.stack([J[:, a] + gamma * (T[a] @ V) for a in range(NA)], 1)
    return V, Q.argmin(1), n_it


def solve(Tb, J, gamma=0.95, eps=1e-6, max_iter=1000):
    """Optimal policy (72,) for base dynamics ``Tb`` and cost ``J``."""
    return value_iteration(Tb, J, gamma, eps, max_iter)[1]


# ---------------------------------------------------------------------------
# Rollout with the refractory environment
# ---------------------------------------------------------------------------
def env_step(mode, counter, action, soft_w, hard_w):
    """One environment transition: (next_mode, next_counter, alert_fired)."""
    if action == SOFT:
        return SOFT_REFR, soft_w, True
    if action == HARD:
        return HARD_REFR, hard_w, True
    if mode in (SOFT_REFR, HARD_REFR):
        counter -= 1
        return (IDLE, 0, False) if counter <= 0 else (mode, counter, False)
    return IDLE, 0, False


def rollout_mdp(ps, policy, bins, soft_w=60, hard_w=360, tau=0.05, dur_thr=3,
                return_trace=False):
    """Apply a solved policy to ONE stream with the refractory timer.

    Returns a bool array of patient-facing alerts (Soft or Hard), or, with
    ``return_trace=True``, a dict with actions / alerts / modes / features.
    """
    lv, tr, du = features(ps, bins, tau, dur_thr)
    n = len(ps)
    alerts = np.zeros(n, bool)
    actions = np.zeros(n, int)
    modes = np.zeros(n, int)
    mode = ctr = 0
    for t in range(n):
        modes[t] = mode
        a = int(policy[state_id(lv[t], tr[t], du[t], mode)])
        actions[t] = a
        mode, ctr, fired = env_step(mode, ctr, a, soft_w, hard_w)
        alerts[t] = fired
    if return_trace:
        return {'actions': actions, 'alerts': alerts, 'modes': modes,
                'levels': lv, 'trends': tr, 'durations': du}
    return alerts


def solve_fold(fold_cache, bins, base, alpha_fa, T_source='cache'):
    """Build T_base for one fold, solve the MDP and return (Tb, policy).

    T_source 'cache'   : the training-stream estimate stored in the cache
             'streams' : re-estimate under ``bins`` from TRAIN_STREAMS if the
                         fold has them, else from the fold's FAR stream (the
                         notebook's fallback).
    """
    if T_source == 'cache':
        Tb = T_base_from_cache(fold_cache)
    elif T_source == 'streams':
        streams = (fold_cache['TRAIN_STREAMS'] if 'TRAIN_STREAMS' in fold_cache
                   else [fold_cache['P_smooth_FAR']])
        Tb = T_base_from_streams(streams, bins, base['trend_threshold'],
                                 base['duration_threshold'])
    else:
        raise ValueError(f"unknown T_source {T_source!r}")
    J = cost(alpha_fa, base)
    pol = solve(Tb, J, base['gamma'], base.get('vi_epsilon', 1e-6),
                base.get('vi_max_iter', 1000))
    return Tb, pol
