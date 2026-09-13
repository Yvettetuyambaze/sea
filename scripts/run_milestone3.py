#!/usr/bin/env python
"""Run Fall 2026 milestones M2 and M3 on public datasets.

MIMIC-IV-ECG is not required. Georgia (US 12-lead, PhysioNet/CinC 2021) is the
second target so PTB-XL -> {Chapman, Georgia} is a full source-to-both-targets
rotation, matching the proposal contingency in Section VII.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("\n==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-m2", action="store_true")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--targets", nargs="+", default=["chapman", "georgia"])
    args = parser.parse_args()

    if not args.skip_download:
        run([PY, "scripts/download_data.py", "--config", args.config, "--datasets", "ptbxl", *args.targets])

    if not args.skip_m2:
        run([PY, "scripts/train_source.py", "--config", args.config, "--dataset", "ptbxl", "--task", "superclass"])

    run([PY, "scripts/train_source.py", "--config", args.config, "--dataset", "ptbxl", "--task", "snomed_shared"])
    ckpt = ROOT / "results/checkpoints/ptbxl_snomed_shared.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    for target in args.targets:
        run(
            [
                PY,
                "scripts/eval_ood.py",
                "--config",
                args.config,
                "--checkpoint",
                str(ckpt),
                "--target",
                target,
                "--task",
                "snomed_shared",
            ]
        )
        run(
            [
                PY,
                "scripts/run_adaptation.py",
                "--config",
                args.config,
                "--checkpoint",
                str(ckpt),
                "--source",
                "ptbxl",
                "--target",
                target,
                "--task",
                "snomed_shared",
                "--reps",
                str(args.reps),
            ]
        )

    status = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt),
        "targets": args.targets,
        "reps": args.reps,
        "mimic_status": "pending_credentialed_access",
        "stand_in_target": "georgia",
    }
    out = ROOT / "results/m3_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nMilestone 3 pipeline finished. Status written to {out}")


if __name__ == "__main__":
    main()
