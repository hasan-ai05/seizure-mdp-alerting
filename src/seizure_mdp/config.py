"""Configuration loading.

The YAML file (configs/final.yaml) is organised in sections. The pipeline
functions lifted from the notebook expect a *flat* dict with the original
notebook key names (``config['fs']``, ``config['SOP']``, ``config['epochs']``,
...), so :func:`flatten` merges the observation-layer sections into one
dict. The ``mdp``, ``patients``, ``baselines``, ``screening`` and ``data``
sections are kept as nested dicts under their own keys.
"""
import os
import copy

import yaml

# Sections whose keys are merged into the top level of the flat config.
_FLAT_SECTIONS = ('signal', 'windowing', 'clinical', 'loso', 'training')

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'configs', 'final.yaml')


def load_yaml(path=DEFAULT_CONFIG_PATH):
    """Load the YAML file as a nested dict (no flattening)."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def flatten(nested):
    """Merge the observation-layer sections into one flat dict.

    Returns a new dict; nested sections not in ``_FLAT_SECTIONS`` are kept
    as-is under their own key (``cfg['mdp']``, ``cfg['patients']``, ...).
    """
    flat = {}
    for key, val in nested.items():
        if key in _FLAT_SECTIONS and isinstance(val, dict):
            flat.update(copy.deepcopy(val))
        else:
            flat[key] = copy.deepcopy(val)
    # Decision-layer values the notebook also kept in CONFIG
    mdp = flat.get('mdp', {})
    for k in ('level_bins', 'trend_threshold', 'duration_threshold', 'gamma',
              'vi_epsilon', 'vi_max_iter'):
        if k in mdp and k not in flat:
            flat[k] = copy.deepcopy(mdp[k])
    if 'soft_arp' in mdp:
        flat.setdefault('soft_refractory_windows', mdp['soft_arp'])
    if 'hard_arp' in mdp:
        flat.setdefault('hard_refractory_windows', mdp['hard_arp'])
    return flat


def load_config(path=DEFAULT_CONFIG_PATH, data_dir=None):
    """Load and flatten the configuration.

    Parameters
    ----------
    path : str
        YAML file. Defaults to ``configs/final.yaml`` in this repository.
    data_dir : str or None
        Where CHB-MIT EDF files live (``<data_dir>/<patient>/<file>.edf``).
        Stored under ``cfg['data_dir']``.
    """
    cfg = flatten(load_yaml(path))
    if data_dir is not None:
        cfg['data_dir'] = data_dir
    cfg.setdefault('data_dir', 'data/chbmit')
    cfg.setdefault('seed', 42)
    return cfg


def mdp_base(cfg):
    """Return the frozen MDP cost parameters (the notebook's reduced BASE)."""
    m = cfg['mdp'] if 'mdp' in cfg else cfg
    return {
        'w_md': list(m['w_md']),
        'w_fa': list(m['w_fa']),
        'mode_cost': list(m['mode_cost']),
        'repeat_penalty': float(m['repeat_penalty']),
        'alpha_md': float(m.get('alpha_md', 1.0)),
        'gamma': float(m['gamma']),
        'vi_epsilon': float(m.get('vi_epsilon', 1e-6)),
        'vi_max_iter': int(m.get('vi_max_iter', 1000)),
        'soft_arp': int(m['soft_arp']),
        'hard_arp': int(m['hard_arp']),
        'trend_threshold': float(m['trend_threshold']),
        'duration_threshold': int(m['duration_threshold']),
        'level_bins': list(m['level_bins']),
    }
