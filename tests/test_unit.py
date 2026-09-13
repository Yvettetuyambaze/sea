import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sea.audit import delta_m, pmin_delta
from sea.data.labels import codes_to_multihot, scp_dict_to_snomed
from sea.metrics import brier_score, chance_level_test, macro_auroc


def test_scp_mapping_and_multihot():
    codes = scp_dict_to_snomed({"AFIB": 100, "SR": 80, "UNKNOWN": 10})
    assert "164889003" in codes
    assert "426783006" in codes
    rows = [
        {"name": "AF", "equivalents": ["164889003"]},
        {"name": "NSR", "equivalents": ["426783006"]},
        {"name": "SB", "equivalents": ["426177001"]},
    ]
    vec = codes_to_multihot(codes, rows)
    assert vec == [1, 1, 0]


def test_metrics_and_audit():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.3], [0.1, 0.7]])
    assert macro_auroc(y, scores) > 0.9
    assert brier_score(y, scores) < 0.1
    chance = chance_level_test(np.array([0.8, 0.82, 0.79, 0.81]))
    assert chance["ci_excludes_chance"]
    assert abs(delta_m(0.84, 0.78, "macro_auroc") - 0.06) < 1e-9
    assert pmin_delta([5, 10, 20, 30, 50], [0.80, 0.805, 0.82, 0.821, 0.822], 0.01) == 5


if __name__ == "__main__":
    test_scp_mapping_and_multihot()
    test_metrics_and_audit()
    print("unit tests ok")
