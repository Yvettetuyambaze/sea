import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sea.audit import delta_m, pmin_delta
from sea.config import apply_quick, load_config
from sea.train import make_loader
from sea.data.cache import ArrayECGDataset
from sea.data.datasets import _safe_rdheader, load_challenge_dir
from sea.data.discover import discover_roots, find_ptbxl_root
from sea.data.labels import codes_to_multihot, report_text_to_snomed, scp_dict_to_snomed
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


def test_machine_report_phrases_are_local_and_specific():
    assert "427084000" in report_text_to_snomed("Sinus tachycardia")
    assert "426177001" in report_text_to_snomed("Sinus bradycardia")
    assert "426783006" in report_text_to_snomed("Normal sinus rhythm")
    assert "164889003" in report_text_to_snomed("Atrial fibrillation")
    tach = report_text_to_snomed("Sinus tachycardia")
    assert "426783006" not in tach
    assert report_text_to_snomed("unrelated statement") == []


def test_metrics_and_audit():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.3], [0.1, 0.7]])
    assert macro_auroc(y, scores) > 0.9
    assert brier_score(y, scores) < 0.1
    chance = chance_level_test(np.array([0.8, 0.82, 0.79, 0.81]))
    assert chance["ci_excludes_chance"]
    assert abs(delta_m(0.84, 0.78, "macro_auroc") - 0.06) < 1e-9
    assert pmin_delta([5, 10, 20, 30, 50], [0.80, 0.805, 0.82, 0.821, 0.822], 0.01) == 5


def test_discover_nested_ptbxl(tmp_path: Path | None = None):
    root = tmp_path or Path("_tmp_discover")
    nested = root / "upload" / "ptb-xl-1.0.3"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "ptbxl_database.csv").write_text("ecg_id\n1\n", encoding="utf-8")
    (nested / "scp_statements.csv").write_text("x\n", encoding="utf-8")
    assert find_ptbxl_root(root / "upload") == nested
    found = discover_roots(root / "upload")
    assert found["ptbxl"] == nested


def test_challenge_loader_skips_bad_headers(tmp_path: Path):
    good = tmp_path / "JS00001.hea"
    good.write_text(
        "JS00001 1 500 500\n"
        "JS00001.mat 16 1000.0(0)/mV 16 0 0 0 0 I\n"
        "# Dx: 164889003\n",
        encoding="ascii",
    )
    (tmp_path / "JS00001.mat").write_bytes(b"x")
    bad = tmp_path / "JS00002.hea"
    bad.write_text("<html>not a wfdb header</html>\n", encoding="ascii")
    (tmp_path / "JS00002.mat").write_bytes(b"x")
    assert _safe_rdheader(str(good.with_suffix(""))) is not None
    assert _safe_rdheader(str(bad.with_suffix(""))) is None
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml").raw
    bundle = load_challenge_dir("chapman", tmp_path, cfg, "snomed_shared")
    assert len(bundle.records) == 1
    assert Path(bundle.records[0]).name == "JS00001"


def test_make_loader_drops_incomplete_batch():
    x = np.zeros((9, 12, 16), dtype=np.float32)
    y = np.zeros((9, 2), dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    cfg = {"train": {"batch_size": 8, "num_workers": 0}, "device": "cpu"}
    kept = sum(batch[0].shape[0] for batch in make_loader(ds, np.arange(9), cfg, drop_last=True))
    assert kept == 8


def test_array_dataset_and_quick_config():
    x = np.zeros((4, 12, 1000), dtype=np.float32)
    y = np.zeros((4, 5), dtype=np.float32)
    ds = ArrayECGDataset(x, y)
    signal, label = ds[0]
    assert tuple(signal.shape) == (12, 1000)
    assert tuple(label.shape) == (5,)
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml").raw
    assert "ptbxl" in cfg["datasets"]
    quick = apply_quick(cfg)
    assert quick["train"]["epochs"] <= 2
    assert quick["adaptation"]["monte_carlo_reps"] == 1
    assert quick["datasets"]["mimic"]["max_records"] <= 128


if __name__ == "__main__":
    test_scp_mapping_and_multihot()
    test_machine_report_phrases_are_local_and_specific()
    test_metrics_and_audit()
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_discover_nested_ptbxl(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_challenge_loader_skips_bad_headers(Path(tmp))
    test_array_dataset_and_quick_config()
    test_make_loader_drops_incomplete_batch()
    print("unit tests ok")
