"""seizure_mdp -- an adaptive Markov decision layer for graded seizure alerting.

Modules
-------
config     load configs/final.yaml
data       CHB-MIT download, parsing, preprocessing, segmentation, LOSO folds
models     CNN-TCN / LSTM / RF classifiers, AUC-weighted fusion, history buffer
mdp        72-state MDP: features, transitions, cost, value iteration, rollout
baselines  threshold / k-of-n / 2-level decision rules
evaluate   event sensitivity, FAR/h, chance test, patient screening
cache      per-fold cache builder (the only GPU-heavy step)
"""
__version__ = '1.0.0'
