import runpy
from pathlib import Path

import numpy as np
import pytest


compute_all_metrics = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / 'openood' / 'evaluators' /
        'metrics.py'))['compute_all_metrics']


@pytest.mark.unit
def test_compute_all_metrics_uses_ood_positive_fpr95():
    rng = np.random.default_rng(308)
    id_conf = rng.normal(0.9, 0.05, 1000)
    ood_conf = rng.normal(0.5, 0.2, 1000)
    label = np.r_[np.zeros(1000, dtype=int), -np.ones(1000, dtype=int)]
    conf = np.r_[id_conf, ood_conf]
    pred = label.copy()

    fpr = compute_all_metrics(conf, label, pred)[0]

    assert fpr == pytest.approx(0.093)
