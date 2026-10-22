"""CHB-MIT acquisition, preprocessing, segmentation and LOSO fold construction.

Lifted from the development notebook (cells 2-7 of the final run order).
Changes relative to the notebook:
  * ``wget`` via os.system replaced by urllib;
  * ``data_dir`` is a function parameter / ``config['data_dir']`` instead of a
    hard-coded Colab path (files live at ``<data_dir>/<patient>/<file>``);
  * the interictal split seed uses ``zlib.crc32`` instead of ``hash()`` (see
    :func:`split_interictal`).
"""
import os
import urllib.request
import zlib
from dataclasses import dataclass, field

import numpy as np

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"


# ---------------------------------------------------------------------------
# Download / summary parsing
# ---------------------------------------------------------------------------
def download_file(patient, filename, data_dir, base_url=BASE_URL):
    """Download ``<base_url>/<patient>/<filename>`` to ``<data_dir>/<patient>/``.

    Returns True if the file is present (already there or downloaded and
    larger than 1 kB); a failed / empty download is removed.
    """
    url = f"{base_url}/{patient}/{filename}"
    out_dir = os.path.join(data_dir, patient)
    os.makedirs(out_dir, exist_ok=True)
    output = os.path.join(out_dir, filename)
    if not os.path.exists(output):
        try:
            urllib.request.urlretrieve(url, output)
        except Exception:
            pass
        if os.path.exists(output) and os.path.getsize(output) > 1000:
            return True
        if os.path.exists(output):
            os.remove(output)
        return False
    return True


def download_summary(patient, data_dir, base_url=BASE_URL):
    """Download ``<patient>-summary.txt``; returns its path or None."""
    filename = f"{patient}-summary.txt"
    ok = download_file(patient, filename, data_dir, base_url)
    return os.path.join(data_dir, patient, filename) if ok else None


def parse_summary(summary_path):
    """Parse a CHB-MIT summary file.

    Returns {recording_filename: [{'start': int, 'end': int}, ...]};
    recordings without seizures map to an empty list.
    """
    seizures_by_file = {}
    current_file = None
    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('File Name:'):
                current_file = line.split(':')[1].strip()
                seizures_by_file[current_file] = []
            elif 'Seizure' in line and 'Start Time:' in line:
                digits = ''.join(c for c in line.split(':')[1] if c.isdigit())
                if digits and current_file:
                    seizures_by_file[current_file].append({'start': int(digits)})
            elif 'Seizure' in line and 'End Time:' in line:
                digits = ''.join(c for c in line.split(':')[1] if c.isdigit())
                if digits and current_file and seizures_by_file[current_file]:
                    seizures_by_file[current_file][-1]['end'] = int(digits)
    return seizures_by_file


def patient_info_from_summary(summary_path):
    """The per-patient dict the loaders expect (seizures_by_file, counts)."""
    seizures_by_file = parse_summary(summary_path)
    total = sum(len(s) for s in seizures_by_file.values())
    return {'seizures_by_file': seizures_by_file,
            'total_seizures': total,
            'seizure_files': [f for f, s in seizures_by_file.items() if len(s) > 0]}


def download_patient(patient, patient_info, data_dir, base_url=BASE_URL, verbose=True):
    """Download every EDF listed in the patient's summary (seizure + interictal)."""
    n_ok = 0
    for fname in patient_info['seizures_by_file']:
        if download_file(patient, fname, data_dir, base_url):
            n_ok += 1
        elif verbose:
            print(f"  failed: {patient}/{fname}")
    if verbose:
        print(f"  {patient}: {n_ok} EDF files present")
    return n_ok


# ---------------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------------
@dataclass
class PatientData:
    """All data for a single patient. Isolated container."""
    patient_id: str
    seizures_by_file: dict          # {filename: [{'start','end'}]}
    total_seizures: int
    seizure_files: list             # recordings containing seizures
    interictal_files: list          # recordings without seizures
    recordings: dict = field(default_factory=dict)   # loaded data
    results: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"PatientData({self.patient_id}: {self.total_seizures} seizures, "
                f"{len(self.seizure_files)} seizure recs, "
                f"{len(self.interictal_files)} interictal recs)")


def preprocess_recording(edf_path, config):
    """Load one EDF: standard channels, bandpass 0.5-40 Hz (FIR, zero phase),
    notch 60 Hz. Returns (n_channels, n_timepoints) or None if unreadable or
    more than two standard channels are missing.
    """
    import mne
    mne.set_log_level('ERROR')
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception:
        return None
    available = [ch for ch in config['channels'] if ch in raw.ch_names]
    if len(available) < len(config['channels']) - 2:
        return None
    raw.pick(available)                 # notebook used the (now deprecated) pick_channels
    raw.reorder_channels(available)
    raw.filter(l_freq=config['bandpass_low'], h_freq=config['bandpass_high'],
               method='fir', phase='zero', verbose=False)
    raw.notch_filter(freqs=config['notch_freq'], verbose=False)
    return raw.get_data()


def discover_patient_files(patient_id, patient_info, config):
    """All EDF files of a patient split into (seizure_files, interictal_files, all_edf)."""
    patient_dir = os.path.join(config['data_dir'], patient_id)
    all_edf = sorted(f for f in os.listdir(patient_dir)
                     if f.endswith('.edf') and not f.endswith('.seizures'))
    sbf = patient_info['seizures_by_file']
    seizure_files = [f for f in all_edf if len(sbf.get(f, [])) > 0]
    interictal_files = [f for f in all_edf if len(sbf.get(f, [])) == 0]
    return seizure_files, interictal_files, all_edf


def load_patient(patient_id, patient_info, config, max_interictal=None):
    """Load and preprocess ALL seizure recordings and up to ``max_interictal``
    interictal recordings (None = config['max_interictal'] = all)."""
    seizure_files, interictal_files, _ = discover_patient_files(patient_id, patient_info, config)
    if max_interictal is None:
        max_interictal = config.get('max_interictal', None)
    selected = interictal_files if max_interictal is None else interictal_files[:max_interictal]

    pd_obj = PatientData(patient_id=patient_id,
                         seizures_by_file=patient_info['seizures_by_file'],
                         total_seizures=patient_info['total_seizures'],
                         seizure_files=seizure_files,
                         interictal_files=selected)
    for fname in seizure_files + selected:
        edf_path = os.path.join(config['data_dir'], patient_id, fname)
        if not os.path.exists(edf_path):
            continue
        data = preprocess_recording(edf_path, config)
        if data is not None:
            pd_obj.recordings[fname] = {'data': data,
                                        'seizures': pd_obj.seizures_by_file.get(fname, [])}
    return pd_obj


# ---------------------------------------------------------------------------
# Segmentation and labelling
# ---------------------------------------------------------------------------
def label_window(t, seizures, config):
    """Label of the window starting at ``t`` seconds:
    1 preictal [onset-SOP-SPH, onset-SPH), 2 ictal, 0 interictal, None excluded
    (closer than ``interictal_margin`` to a seizure without being preictal)."""
    SOP, SPH, margin = config['SOP'], config['SPH'], config['interictal_margin']
    for s in seizures:
        start = s['start']
        end = s.get('end', start + 40)
        if start <= t <= end:
            return 2
        if (start - SOP - SPH) <= t < (start - SPH):
            return 1
        if abs(t - start) < margin or abs(t - end) < margin:
            return None
    return 0


def segment_recording(data, seizures, config, overlap):
    """Cut one recording into labelled windows -> (windows (n,C,T), labels (n,), times (n,))."""
    fs = config['fs']
    win_size = int(config['window_sec'] * fs)
    step = int(win_size * (1 - overlap))
    n_samples = data.shape[1]
    windows, labels, times = [], [], []
    t_idx = 0
    while t_idx + win_size <= n_samples:
        t_sec = t_idx / fs
        lbl = label_window(t_sec, seizures, config)
        if lbl is not None:
            windows.append(data[:, t_idx:t_idx + win_size])
            labels.append(lbl)
            times.append(t_sec)
        t_idx += step
    if not windows:
        return np.empty((0, data.shape[0], win_size)), np.array([]), np.array([])
    return np.array(windows), np.array(labels), np.array(times)


def segment_patient(patient, config, overlap):
    """Segment all recordings; dict {fname: {'windows','labels','times','seizures'}}.
    Keeping windows grouped by recording is what enables leakage-free LOSO."""
    segmented = {}
    for fname, rec in patient.recordings.items():
        w, l, t = segment_recording(rec['data'], rec['seizures'], config, overlap)
        if len(w) > 0:
            segmented[fname] = {'windows': w, 'labels': l, 'times': t,
                                'seizures': rec['seizures']}
    return segmented


# ---------------------------------------------------------------------------
# LOSO folds
# ---------------------------------------------------------------------------
def split_interictal(segmented, config, patient_id, seed=42):
    """Shuffle interictal recordings with a patient-specific seed and split
    them into train / FAR groups (config['interictal_train_ratio']).

    NOTE: the original notebook seeded with ``SEED + hash(patient_id) % 10000``.
    Python's str ``hash()`` is randomised per interpreter session
    (PYTHONHASHSEED), so that split was NOT reproducible across sessions and
    the exact train/FAR partition behind the cached results cannot be
    regenerated bit-exactly. The cache preserves the partition that was
    used. Here the seed is ``seed + zlib.crc32(patient_id) % 10000``, which
    is deterministic everywhere.
    """
    interictal_recs = sorted(f for f, s in segmented.items() if len(s['seizures']) == 0)
    rng = np.random.default_rng(seed + zlib.crc32(patient_id.encode()) % 10000)
    interictal_recs = list(interictal_recs)
    rng.shuffle(interictal_recs)
    n_total = len(interictal_recs)
    n_train = int(round(n_total * config['interictal_train_ratio']))
    n_train = max(1, min(n_train, n_total - 1))          # >= 1 in each group
    return interictal_recs[:n_train], interictal_recs[n_train:]


def filter_valid_seizures(segmented, config):
    """Seizure recordings with >= config['min_preictal_windows'] preictal windows."""
    min_pre = config['min_preictal_windows']
    valid, excluded = [], []
    for fname, s in segmented.items():
        if len(s['seizures']) == 0:
            continue
        n_pre = int(np.sum(s['labels'] == 1))
        if n_pre >= min_pre:
            valid.append(fname)
        else:
            excluded.append((fname, n_pre))
    return valid, excluded


def build_loso_folds(segmented, config, patient_id, seed=42):
    """One fold per valid seizure recording; interictal_FAR is never trained on."""
    inter_train, inter_FAR = split_interictal(segmented, config, patient_id, seed)
    valid_seizures, excluded = filter_valid_seizures(segmented, config)
    folds = []
    for i, test_sz in enumerate(valid_seizures):
        folds.append({'fold_id': i + 1, 'patient_id': patient_id,
                      'test_seizure': test_sz,
                      'train_seizures': [f for f in valid_seizures if f != test_sz],
                      'interictal_train': inter_train,
                      'interictal_FAR': inter_FAR})
    info = {'patient_id': patient_id, 'n_valid_seizures': len(valid_seizures),
            'n_excluded': len(excluded), 'excluded': excluded,
            'interictal_train': inter_train, 'interictal_FAR': inter_FAR,
            'n_folds': len(folds)}
    return folds, info


# ---------------------------------------------------------------------------
# Fold datasets
# ---------------------------------------------------------------------------
def collect_windows(fnames, seg_dict, include_labels=(0, 1, 2)):
    """Concatenate windows of the listed recordings -> (W (N,C,T), L (N,), T (N,))."""
    W, L, T = [], [], []
    for fname in fnames:
        if fname not in seg_dict:
            continue
        s = seg_dict[fname]
        mask = np.isin(s['labels'], include_labels)
        if mask.sum() > 0:
            W.append(s['windows'][mask])
            L.append(s['labels'][mask])
            T.append(s['times'][mask])
    if not W:
        return np.empty((0,)), np.array([]), np.array([])
    return np.concatenate(W, 0), np.concatenate(L, 0), np.concatenate(T, 0)


def balance_training(windows, labels, config, seed=42):
    """Undersample to interictal:preictal = config['train_ratio']; drop ictal.
    Returns (windows, binary labels)."""
    ratio = config['train_ratio']
    pre_idx = np.where(labels == 1)[0]
    inter_idx = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)
    n_keep_inter = int(len(pre_idx) * ratio)
    if n_keep_inter <= len(inter_idx):
        keep_inter = rng.choice(inter_idx, n_keep_inter, replace=False)
        keep_pre = pre_idx
    else:
        keep_pre = rng.choice(pre_idx, int(len(inter_idx) / ratio), replace=False)
        keep_inter = inter_idx
    keep = np.sort(np.concatenate([keep_inter, keep_pre]))
    return windows[keep], (labels[keep] == 1).astype(int)


def build_fold_data(fold, seg_train, seg_eval, config, seed=42):
    """Training (50 % overlap, balanced), test (held-out seizure, 0 % overlap)
    and FAR (interictal_FAR, 0 % overlap) sets of one LOSO fold."""
    train_files = fold['train_seizures'] + fold['interictal_train']
    W_tr, L_tr, _ = collect_windows(train_files, seg_train, include_labels=(0, 1))
    X_train, y_train = balance_training(W_tr, L_tr, config, seed)

    test_file = fold['test_seizure']
    X_test, y_test_raw, t_test = collect_windows([test_file], seg_eval, include_labels=(0, 1, 2))
    y_test = (y_test_raw == 1).astype(int)
    test_seizures = seg_eval[test_file]['seizures']

    X_FAR, _, t_FAR = collect_windows(fold['interictal_FAR'], seg_eval, include_labels=(0,))
    return {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test, 't_test': t_test, 'y_test_raw': y_test_raw,
            'X_FAR': X_FAR, 't_FAR': t_FAR, 'test_seizures': test_seizures}
