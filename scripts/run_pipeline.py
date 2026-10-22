#!/usr/bin/env python
"""End-to-end pipeline: download -> load -> segment -> LOSO folds -> fold cache
-> screening -> MDP vs baselines.

    python scripts/run_pipeline.py --patients chb01 chb03 --data-dir data/chbmit
                                   --cache-dir cache --config configs/final.yaml

Needs CHB-MIT (downloaded from PhysioNet on demand, ~40 GB for the full
set) and a GPU for the classifier training (~70 min per patient on an A100).
The decision-layer evaluation at the end is instant and can be re-run from
the cache alone with --skip-training.

Per-patient MDP settings (bins, T source, alpha_fa) come from the
``patients`` section of the config when the patient is listed there;
otherwise the uniform adopted bins and T re-estimated from TRAIN_STREAMS are
used with --alpha-fa.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from seizure_mdp.config import load_config, load_yaml, mdp_base, DEFAULT_CONFIG_PATH  # noqa: E402
from seizure_mdp import data as D                                                     # noqa: E402
from seizure_mdp.baselines import make_baselines                                      # noqa: E402
from seizure_mdp.evaluate import evaluate_mdp, best_baseline_at_sensitivity, screen_patient  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--patients', nargs='+', default=['chb01', 'chb03', 'chb10', 'chb20'],
                    help='CHB-MIT patient ids')
    ap.add_argument('--data-dir', default='data/chbmit', help='where EDF files are / go')
    ap.add_argument('--cache-dir', default='cache', help='where per-patient fold caches are written')
    ap.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    ap.add_argument('--alpha-fa', type=float, default=64.0,
                    help='false-alarm weight for patients without a per-patient entry')
    ap.add_argument('--no-download', action='store_true', help='do not fetch missing files')
    ap.add_argument('--skip-training', action='store_true',
                    help='only evaluate existing caches in --cache-dir')
    ap.add_argument('--force', action='store_true', help='rebuild caches even if present')
    return ap.parse_args(argv)


def build_patient_cache(pid, cfg, args):
    from seizure_mdp.cache import build_fold_cache
    cache_path = os.path.join(args.cache_dir, f'loso_{pid}_with_streams.pkl')
    if os.path.exists(cache_path) and not args.force:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    if args.skip_training:
        print(f"  {pid}: no cache at {cache_path} and --skip-training set")
        return None

    summary = D.download_summary(pid, args.data_dir) if not args.no_download \
        else os.path.join(args.data_dir, pid, f'{pid}-summary.txt')
    if summary is None or not os.path.exists(summary):
        print(f"  {pid}: summary not available")
        return None
    info = D.patient_info_from_summary(summary)
    if info['total_seizures'] < cfg['data']['min_seizures']:
        print(f"  {pid}: only {info['total_seizures']} seizures (< {cfg['data']['min_seizures']}), skipped")
        return None
    if not args.no_download:
        D.download_patient(pid, info, args.data_dir)

    print(f"  {pid}: loading + preprocessing ...", flush=True)
    patient = D.load_patient(pid, info, cfg)
    seg_train = D.segment_patient(patient, cfg, overlap=cfg['train_overlap'])
    seg_eval = D.segment_patient(patient, cfg, overlap=cfg['eval_overlap'])
    folds, finfo = D.build_loso_folds(seg_eval, cfg, pid, cfg['seed'])
    print(f"  {pid}: {finfo['n_folds']} LOSO folds, {finfo['n_excluded']} seizures excluded, "
          f"{len(finfo['interictal_train'])} train / {len(finfo['interictal_FAR'])} FAR interictal recordings")
    return build_fold_cache(folds, seg_train, seg_eval, cfg, cache_path, force=args.force)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, data_dir=args.data_dir)
    nested = load_yaml(args.config)
    base = mdp_base(cfg)
    SOP, SPH, WIN = cfg['SOP'], cfg['SPH'], cfg['window_sec']
    bl = nested['baselines']
    baselines = make_baselines(bl['thresholds'], [tuple(x) for x in bl['k_of_n']],
                               [tuple(x) for x in bl['two_level']])
    scr = nested['screening']
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"{'patient':7} {'screen':>22} | {'bins':>18} {'T src':>8} {'a_fa':>5} | "
          f"{'MDP sens/FAR':>14} | best baseline at same sensitivity")
    for pid in args.patients:
        pc = build_patient_cache(pid, cfg, args)
        if pc is None:
            continue
        s = screen_patient(pc, scr['peak_threshold'], scr['catch_fraction'], scr['auc_threshold'], SOP, SPH)
        screen = f"AUC {s['auc']:.3f} {s['catch']}/{s['total']} {'ok' if s['valid'] else 'OUT'}"
        p = nested['patients'].get(pid, {})
        bins = list(p.get('level_bins', base['level_bins']))
        tsrc = p.get('T_source', 'streams')
        a_fa = float(p.get('alpha_fa', args.alpha_fa))
        m = evaluate_mdp(pc, bins, base, a_fa, tsrc, SOP, SPH, WIN)
        bf, bname, bh, bn = best_baseline_at_sensitivity(pc, m['n_hit'], baselines, base['soft_arp'],
                                                         base['hard_arp'], SOP, SPH, WIN)
        gap = 100 * (m['far_mean'] - bf) / bf if bf else 0.0
        print(f"{pid:7} {screen:>22} | {str(bins):>18} {tsrc:>8} {a_fa:>5g} | "
              f"{m['n_hit']}/{m['n_folds']} {m['far_mean']:>6.3f} | {bname} {bf:.3f} ({bh}/{bn})  "
              f"gap {gap:+.1f}%  p_chance {m['p_chance']:.4f}")


if __name__ == '__main__':
    main()
