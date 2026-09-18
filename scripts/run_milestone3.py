#!/usr/bin/env python
"""Run Fall 2026 milestones M2 and M3 on already-prepared local datasets."""

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
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-download", action="store_true", default=True)
    parser.add_argument("--download", action="store_true", help="Also fetch missing public files")
    parser.add_argument("--skip-m2", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--targets", nargs="+", default=["chapman", "mimic"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    extra = []
    if args.config:
        extra += ["--config", args.config]
    if args.data_root:
        extra += ["--data-root", args.data_root]
    if args.output_dir:
        extra += ["--output-dir", args.output_dir]
    if args.epochs:
        extra += ["--epochs", str(args.epochs)]
    if args.batch_size:
        extra += ["--batch-size", str(args.batch_size)]
    if args.quick:
        extra.append("--quick")

    if args.download:
        datasets = ["ptbxl", *args.targets]
        run([PY, "scripts/download_data.py", "--datasets", *datasets])

    if not args.skip_m2:
        run([PY, "scripts/run_milestone2.py", *extra])

    run([PY, "scripts/train_source.py", "--dataset", "ptbxl", "--task", "snomed_shared", *extra])
    ckpt = Path(args.output_dir or "results") / "checkpoints" / "ptbxl_snomed_shared.pt"
    if args.output_dir is None:
        ckpt = ROOT / "results/checkpoints/ptbxl_snomed_shared.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    for target in args.targets:
        run([PY, "scripts/eval_ood.py", "--checkpoint", str(ckpt), "--target", target, "--task", "snomed_shared", *extra])
        adapt = [
            PY,
            "scripts/run_adaptation.py",
            "--checkpoint",
            str(ckpt),
            "--source",
            "ptbxl",
            "--target",
            target,
            "--task",
            "snomed_shared",
            *extra,
        ]
        if args.reps is not None:
            adapt += ["--reps", str(args.reps)]
        run(adapt)

    status = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt),
        "targets": args.targets,
        "reps": args.reps,
        "quick": args.quick,
    }
    out = Path(args.output_dir or (ROOT / "results")) / "m3_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nMilestone 2/3 pipeline finished. Status written to {out}")


if __name__ == "__main__":
    main()
