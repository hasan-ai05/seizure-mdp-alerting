"""Non-MDP decision layers evaluated in the SAME refractory environment.

Threshold, k-of-n and 2-level (soft/hard threshold) rules. A firing rule
issues one alert and is then blocked for the Soft (or Hard, if p > hard_theta)
refractory length, exactly like the MDP rollout.
"""
import numpy as np


def rollout_threshold(p, theta, k=1, n=1, hard_theta=None, soft_w=60, hard_w=360):
    """Alerts (bool array) of a threshold / k-of-n / 2-level rule on ONE stream.

    fire(t)  : at least k of the last n windows have p > theta
    blocked  : through t0 + W inclusive after an alert at t0 (W = hard_w if
               p[t0] > hard_theta else soft_w), matching env_step whose
               counter reaches 0 after the step at t0 + W.
    """
    p = np.asarray(p, float)
    above = (p > theta).astype(int)
    cs = np.concatenate([[0], np.cumsum(above)])
    i = np.arange(len(p))
    fire = (cs[i + 1] - cs[np.maximum(0, i - n + 1)]) >= k
    al = np.zeros(len(p), bool)
    blocked_until = -1
    for t in np.nonzero(fire)[0]:
        if t > blocked_until:
            hard = hard_theta is not None and p[t] > hard_theta
            al[t] = True
            blocked_until = t + (hard_w if hard else soft_w)
    return al


def make_baselines(thresholds=(.30, .40, .50, .60, .70, .85),
                   k_of_n=((.5, 3, 5), (.6, 3, 5), (.6, 5, 10), (.7, 5, 10)),
                   two_level=((.5, .85), (.6, .85))):
    """List of (name, kwargs for rollout_threshold)."""
    arms = [(f'thr {t:.2f}', dict(theta=t)) for t in thresholds]
    arms += [(f'k-of-n {t:.2f} ({k}/{n})', dict(theta=t, k=k, n=n)) for t, k, n in k_of_n]
    arms += [(f'2-level {a:.2f}/{b:.2f}', dict(theta=a, hard_theta=b)) for a, b in two_level]
    return arms


# The 12 arms of the paper
BASELINES = make_baselines()
