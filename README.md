# Beyond the Threshold: An Adaptive Markov Decision Layer for Graded, Personalized Seizure Alerting

Code for the paper of the same title (H. M. Akhras, A. Nasir, 2026, under review).

**Paper draft (PDF):** https://drive.google.com/file/d/1-RhfrJpov4A2nKV58WP66_Pn_jIvqFqH/view?usp=sharing

## Idea

Seizure-prediction classifiers emit a probability stream; almost every system
then turns it into alarms with a fixed threshold. We replace the threshold by a
small Markov decision process that sits *on top of* an unchanged prediction
layer (CNN-TCN + LSTM + random forest, AUC-weighted fusion, 5-window causal
moving average). The state is the discretised probability **level** (4), its
**trend** (rising / stable / falling), the **duration** in the current level
(short / long) and the alerting **mode** (idle, soft-refractory,
hard-refractory): 72 states. The actions are *No alert*, *Soft alert* and
*Hard alert*, each with its own refractory period (5 min / 30 min). A
per-patient transition matrix is estimated from the training recordings only,
a missed-detection / false-alarm cost with one clinical knob (alpha_fa) is
minimised by value iteration, and the resulting policy is rolled out window by
window. The decision layer is evaluated leave-one-seizure-out on CHB-MIT
against twelve threshold-type rules in the same refractory environment.

## Results (Table 3 of the paper)

False-alarm rate per hour of the MDP against the best non-MDP rule that
reaches the same event sensitivity:

| patient | sensitivity | best baseline FAR/h | MDP FAR/h | change |
|---------|-------------|--------------------:|----------:|-------:|
| chb01   | 6/6 (100 %) | 1.223 (2-level 0.60/0.85) | **0.506** | -58.6 % |
| chb03   | 3/5 (60 %)  | 0.000 (k-of-n 0.60, 5/10) | **0.000** | tie |
| chb10   | 6/7 (86 %)  | 0.696 (threshold 0.70)    | **0.232** | -66.7 % |
| chb20   | 4/4 (100 %) | 0.250 (threshold 0.85)    | **0.208** | -16.8 % |

(The chb20 change is -16.8 % on the rounded values reported in the paper;
the script below prints -16.7 % because it uses the unrounded FARs.)

## Reproduce in one minute

The decision layer is deterministic and needs no GPU. Download the cache
release (`MASTER_complete_session.pkl`, 17 MB, see *Data*) into a folder and
run

```bash
pip install numpy scipy scikit-learn pyyaml
python scripts/reproduce_table3.py /path/to/mdp_seizure_cache
```

Output of that command against the released cache:

```
patient               bins    T src  a_fa |     MDP sens/FAR  recorded | best baseline at same sensitivity
chb01     [0.3, 0.6, 0.85]    cache    64 | 6/6  0.506          0.506 | 2-level 0.60/0.85 1.223 (6/6)  gap -58.6%
chb03     [0.3, 0.6, 0.75]  streams    64 | 3/5  0.000          0.000 | k-of-n 0.60 (5/10) 0.000 (3/5)  gap +0.0%
chb10     [0.3, 0.6, 0.75]  streams    96 | 6/7  0.232          0.232 | thr 0.70 0.696 (6/7)  gap -66.7%
chb20     [0.3, 0.6, 0.85]    cache    96 | 4/4  0.208          0.208 | thr 0.85 0.250 (4/4)  gap -16.7%
```

`SEIZURE_CACHE=/path/to/mdp_seizure_cache pytest tests/` asserts the four
FAR values to three decimals.

## Full pipeline

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --patients chb01 chb03 chb10 chb20 \
       --data-dir data/chbmit --cache-dir cache --config configs/final.yaml
```

This downloads the patients' EDF files from PhysioNet (CHB-MIT 1.0.0),
preprocesses them with MNE (18 bipolar channels, 0.5-40 Hz FIR, 60 Hz
notch), cuts 5-s windows (preictal = [onset-35 min, onset-5 min), interictal
= more than 1 h from any seizure), builds leave-one-seizure-out folds, trains
the three classifiers per fold, stores the smoothed probability streams and
the training-stream transition estimate in a per-patient cache, and finally
runs the screening gate, the MDP and the baselines. Training takes roughly
70 minutes per patient on an A100; everything after the cache is instant.
`--skip-training` evaluates existing caches only.

## Repository map

```
configs/final.yaml          every adopted parameter (signal, windows, LOSO, training,
                            frozen MDP cost "BASE_v2", per-patient Table 3 settings)
src/seizure_mdp/
  config.py                 YAML loading, flattening to the notebook's key names
  data.py                   PhysioNet download, summary parsing, MNE preprocessing,
                            segmentation / labelling, LOSO folds, balancing
  models.py                 CNN-TCN, LSTM, RF (PSD), training with early stopping,
                            AUC-weighted fusion, history buffer (torch imported lazily)
  mdp.py                    72-state model: features, state encoding, T_base,
                            ordinal cost, value iteration, rollout with refractory
  baselines.py              threshold / k-of-n / 2-level rules, the 12 arms
  evaluate.py               event sensitivity, FAR/h per recording, chance test,
                            best-baseline-at-matched-sensitivity, patient screening
  cache.py                  per-fold cache builder (the only GPU step)
scripts/reproduce_table3.py decision layer from the cache (Table 3)
scripts/run_pipeline.py     end-to-end
tests/test_reproduce.py     unit tests + Table 3 regression (needs SEIZURE_CACHE)
archive/                    the original development notebook, unchanged
```

## Configuration notes

Please read this before comparing numbers with the notebook.

* **Frozen cost (BASE_v2).** `w_md = [4, 2, 1]`, `w_fa = [0, 2, 2]`,
  `mode_cost = [0, 0.3, 0.6]`, `repeat_penalty = 2`, `alpha_md = 1`,
  `gamma = 0.95`, value-iteration tolerance 1e-6, soft/hard refractory 60/360
  windows, trend threshold 0.05, duration threshold 3, cost in the ordinal
  form (`A = d * level`, `B = d * (3 - level)`). The only per-patient knob of
  the cost is `alpha_fa` (64 for chb01 and chb03, 96 for chb10 and chb20).
* **Per-patient level bins and transition source are not uniform.** The
  numbers in Table 3 were recorded with the settings listed under
  `patients:` in `configs/final.yaml`:
  * chb01 and chb20: bins `[0.30, 0.60, 0.85]` and the transition matrix
    stored in the fold cache (the training-stream estimate made with those
    bins);
  * chb10: bins `[0.30, 0.60, 0.75]` and the transition matrix re-estimated
    under those bins from the stored training streams (`TRAIN_STREAMS`);
  * chb03: bins `[0.30, 0.60, 0.75]`; its cache has no training streams, so
    the transition matrix was re-estimated from the fold's FAR stream (the
    notebook's fallback in `evaluate_with_bins`). This means that for chb03
    the transition statistics were taken from the same interictal recordings
    on which the FAR is measured, which is a leak into the transition model
    (not into the classifier). With the cached, training-stream transition
    matrix instead, chb03 gives FAR 0.025 at 3/5 with either bin setting
    (notebook cell 66); the best baseline at 3/5 is 0.000 in both cases, so
    the "tie" becomes a small loss. We flag this as the honest reading of the
    chb03 row.
* **The uniform rule gives a different chb20 number.** Applying the adopted
  uniform bins `[0.30, 0.60, 0.75]` with a re-estimated transition matrix
  yields chb20 FAR 0.333 at 4/4 (notebook cell 87), not 0.208. The 0.208
  reported in the paper is the 0.85-bins / cached-T setting.
* **Training-stream transition estimates under new bins exist for chb10
  only.** Only `cache_chb10_with_streams` stores `TRAIN_STREAMS`; the other
  patients' caches carry just the transition matrix built with 0.85 bins.
  Re-verifying the 0.75 bins cleanly on chb01/chb03/chb20 requires retraining
  them with `scripts/run_pipeline.py` (which now always stores the streams).

## Known limitations

* **The original interictal train/FAR split was not reproducible.** The
  notebook seeded the shuffle with Python's `hash(patient_id)`, which is
  randomised per interpreter session. The exact partition behind the cached
  results therefore cannot be regenerated bit-exactly from the EDF files; the
  released cache preserves it. `src/seizure_mdp/data.py` uses
  `zlib.crc32(patient_id)` instead, so new runs are deterministic but will not
  match the cached partition.
* Four patients (chb01, chb03, chb10, chb20) passed the classifier-quality
  screen (AUC >= 0.70 and >= 60 % of seizures reaching P >= 0.60 in the
  preictal window); chb05 and chb18 did not. Results on four patients are a
  proof of concept, not a population claim.
* The interictal FAR recordings are shared by all folds of a patient (only
  the held-out seizure changes), so per-fold FARs are not independent
  samples.
* Absolute FARs depend on the classifier build; the paper's claim is the
  relative gap to threshold rules on identical probability streams.

## Data

No data is included. EEG: CHB-MIT Scalp EEG Database v1.0.0 on PhysioNet
(https://physionet.org/content/chbmit/1.0.0/), downloaded by
`scripts/run_pipeline.py` or with any PhysioNet mirror. Cache: the session
pickle `MASTER_complete_session.pkl` (classifier outputs, transition
matrices and per-patient caches, 17 MB) will be attached as a GitHub Release
asset and deposited on Zenodo; it is what `scripts/reproduce_table3.py`
reads.

## Status

Paper under review at the *Journal of Undergraduate Research International* (KFUPM). The repository is public for transparency; the submitted manuscript is linked at the top.

## Citation

```bibtex
@unpublished{akhras2026beyond,
  title  = {Beyond the Threshold: An Adaptive Markov Decision Layer for Graded, Personalized Seizure Alerting},
  author = {Akhras, Hasan Mohammed and Nasir, Ali},
  year   = {2026},
  note   = {under review}
}
```

See also `CITATION.cff`. Licensed under the MIT License.
