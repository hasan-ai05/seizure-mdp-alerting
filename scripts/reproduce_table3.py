#!/usr/bin/env python
"""Reproduce Table 3 (decision layer only) from the cached classifier outputs.

Usage:
    python scripts/reproduce_table3.py /path/to/mdp_seizure_cache [--config configs/final.yaml]

The folder must contain MASTER_complete_session.pkl (see README "Data").
No GPU, no EEG data; runs in about a minute.

Per patient: build T_base from the recorded source, solve the 72-state MDP,
roll the policy out with the refractory timer, score sensitivity / FAR, and
compare with the best of 12 baseline arms at the same sensitivity.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from seizure_mdp.config import load_yaml, mdp_base, DEFAULT_CONFIG_PATH   # noqa: E402
from seizure_mdp.baselines import make_baselines                            # noqa: E402
from seizure_mdp.evaluate import evaluate_mdp, best_baseline_at_sensitivity  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cache_dir', help='folder holding MASTER_complete_session.pkl')
    ap.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    base = mdp_base(cfg)
    SOP, SPH, WIN = cfg['clinical']['SOP'], cfg['clinical']['SPH'], cfg['windowing']['window_sec']
    bl = cfg['baselines']
    baselines = make_baselines(bl['thresholds'], [tuple(x) for x in bl['k_of_n']],
                               [tuple(x) for x in bl['two_level']])

    with open(os.path.join(args.cache_dir, 'MASTER_complete_session.pkl'), 'rb') as f:
        D = pickle.load(f)

    print(f"{'patient':7} {'bins':>18} {'T src':>8} {'a_fa':>5} | {'MDP sens/FAR':>16} "
          f"{'recorded':>9} | best baseline at same sensitivity")
    results = {}
    for pid, p in cfg['patients'].items():
        pc = D[p['cache_key']]
        bins = list(p['level_bins'])
        m = evaluate_mdp(pc, bins, base, float(p['alpha_fa']), p['T_source'], SOP, SPH, WIN)
        bf, bname, bh, bn = best_baseline_at_sensitivity(
            pc, m['n_hit'], baselines, base['soft_arp'], base['hard_arp'], SOP, SPH, WIN)
        gap = 100 * (m['far_mean'] - bf) / bf if bf else 0.0
        results[pid] = (m, bf, bname)
        print(f"{pid:7} {str(bins):>18} {p['T_source']:>8} {p['alpha_fa']:>5g} | "
              f"{m['n_hit']}/{m['n_folds']} {m['far_mean']:>6.3f} {p['recorded_far']:>14.3f} | "
              f"{bname} {bf:.3f} ({bh}/{bn})  gap {gap:+.1f}%")
    return results


if __name__ == '__main__':
    main()
