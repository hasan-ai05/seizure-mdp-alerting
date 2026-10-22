"""Regression test of the decision layer against the recorded Table 3 FARs.

Set SEIZURE_CACHE to the folder holding MASTER_complete_session.pkl, e.g.

    SEIZURE_CACHE=/path/to/mdp_seizure_cache pytest tests/

Without the variable the test is skipped. A few structural unit tests run
regardless.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from seizure_mdp import mdp  # noqa: E402

RECORDED = {'chb01': (6, 0.506), 'chb03': (3, 0.000), 'chb10': (6, 0.232), 'chb20': (4, 0.208)}


def test_state_encoding_bijective():
    assert all(mdp.state_id(*mdp.decode_state(s)) == s for s in range(mdp.NS))
    assert len({mdp.state_id(l, t, d, m) for l in range(4) for t in range(3)
                for d in range(2) for m in range(3)}) == 72


def test_cost_shape_and_md_gate():
    base = {'w_md': [4, 2, 1], 'w_fa': [0, 2, 2], 'mode_cost': [0, .3, .6], 'repeat_penalty': 2}
    J = mdp.cost(64.0, base)
    assert J.shape == (72, 3)
    # inside a refractory the missed-detection term is off: No-Alert costs only the mode cost
    s = mdp.state_id(3, 0, 1, mdp.SOFT_REFR)
    assert J[s, mdp.NO_ALERT] == pytest.approx(0.3)


def test_rollout_refractory_timer():
    pol = np.full(72, mdp.SOFT)                      # always alert when allowed? no: Soft in every state
    al = mdp.rollout_mdp(np.full(10, 0.9), pol, [.3, .6, .75], soft_w=3, hard_w=5)
    # Soft in every state means an alert every window (repeat alerts restart the timer)
    assert al.all()
    pol = np.zeros(72, int)
    pol[[mdp.state_id(3, t, d, mdp.IDLE) for t in range(3) for d in range(2)]] = mdp.SOFT
    al = mdp.rollout_mdp(np.full(10, 0.9), pol, [.3, .6, .75], soft_w=3, hard_w=5)
    assert al.tolist() == [True, False, False, False, True, False, False, False, True, False]


@pytest.mark.skipif('SEIZURE_CACHE' not in os.environ, reason='SEIZURE_CACHE not set')
def test_table3_far():
    import reproduce_table3
    results = reproduce_table3.main([os.environ['SEIZURE_CACHE']])
    for pid, (hits, far) in RECORDED.items():
        m = results[pid][0]
        assert m['n_hit'] == hits, pid
        assert round(m['far_mean'], 3) == pytest.approx(far, abs=5e-4), pid
