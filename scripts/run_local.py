#!/usr/bin/env python
"""Local M2 then M3 driver: resume downloads, train, write step-by-step figures."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable

from sea.config import load_config
from sea.data.discover import find_mimic_root
from sea.viz import plot_pipeline_board, write_html_index


def write_status(payload: dict) -> None:
    path = ROOT / "results" / "local_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        plot_pipeline_board(payload.get("steps", {}), ROOT / "results")
        write_html_index(ROOT / "results")
    except Exception as exc:
        print(f"status figure skipped: {exc}", flush=True)


def run(cmd: list[str]) -> None:
    print("\n==>", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"COMMAND FAILED exit={exc.returncode}: {' '.join(cmd)}", flush=True)
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-m2", action="store_true", help="PTB-XL superclass already trained")
    parser.add_argument("--skip-source", action="store_true", help="SNOMED source already trained; start at OOD")
    parser.add_argument("--skip-ood", action="store_true", help="OOD eval already saved; start at adaptation")
    args = parser.parse_args()
    if args.skip_ood:
        args.skip_source = True
    if args.skip_source:
        args.skip_m2 = True

    cfg_path = str(ROOT / "configs/local.yaml")
    cfg = load_config(cfg_path).raw
    targets = ["chapman"]
    mimic_root = Path(cfg["datasets"]["mimic"]["root"])
    if find_mimic_root(mimic_root) is not None:
        targets.append("mimic")

    steps = {
        "1": "running",
        "2": "pending",
        "3": "pending",
        "4": "pending",
        "5": "pending",
        "6": "pending",
        "7": "pending",
        "8": "pending",
    }
    payload = {
        "stage": "download_ptbxl",
        "targets": targets,
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "steps": steps,
    }
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("GPU driver sees RTX 2050 but this PyTorch build is CPU-only. Training will be slower locally.", flush=True)
    write_status(payload)

    extra = ["--config", cfg_path]
    ckpt = ROOT / "results/checkpoints/ptbxl_snomed_shared.pt"
    if args.skip_source:
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing SNOMED checkpoint {ckpt}")
        for key in ("1", "2", "3", "4", "5", "6"):
            steps[key] = "done"
        steps["7"] = "running"
        payload.update(stage="ood_and_adaptation", steps=steps)
        write_status(payload)
        for target in targets:
            if not args.skip_ood:
                run([PY, "scripts/eval_ood.py", "--checkpoint", str(ckpt), "--target", target, "--task", "snomed_shared", *extra])
            run(
                [
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
            )
        steps["7"] = "done"
        steps["8"] = "done"
        payload.update(stage="finished", steps=steps)
        write_status(payload)
        run([PY, "scripts/summarize_results.py", "--output-dir", str(ROOT / "results")])
        print("Local M2/M3 run finished. Figures: results/figures/index.html", flush=True)
        return

    if not args.skip_m2:
        run([PY, "scripts/download_data.py", "--config", cfg_path, "--datasets", "ptbxl"])
        steps["1"] = "done"
        payload.update(stage="milestone2", steps=steps)
        write_status(payload)

        steps["2"] = "running"
        steps["3"] = "running"
        steps["4"] = "running"
        write_status(payload)
        run([PY, "scripts/run_milestone2.py", *extra])
        steps["2"] = "done"
        steps["3"] = "done"
        steps["4"] = "done"
        steps["5"] = "done"
        payload.update(stage="download_targets", steps=steps)
        write_status(payload)
    else:
        steps["1"] = "done"
        steps["2"] = "done"
        steps["3"] = "done"
        steps["4"] = "done"
        steps["5"] = "done"
        payload.update(stage="download_targets", steps=steps)
        write_status(payload)

    run([PY, "scripts/download_data.py", "--config", cfg_path, "--datasets", *targets])
    steps["6"] = "running"
    payload.update(stage="snomed_source", steps=steps)
    write_status(payload)
    run([PY, "scripts/train_source.py", "--dataset", "ptbxl", "--task", "snomed_shared", *extra])
    steps["6"] = "done"
    steps["7"] = "running"
    payload.update(stage="ood_and_adaptation", steps=steps)
    write_status(payload)

    ckpt = ROOT / "results/checkpoints/ptbxl_snomed_shared.pt"
    for target in targets:
        run([PY, "scripts/eval_ood.py", "--checkpoint", str(ckpt), "--target", target, "--task", "snomed_shared", *extra])
        run(
            [
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
        )
    steps["7"] = "done"
    steps["8"] = "done"
    payload.update(stage="finished", steps=steps)
    write_status(payload)
    run([PY, "scripts/summarize_results.py", "--output-dir", str(ROOT / "results")])
    print("Local M2/M3 run finished. Figures: results/figures/index.html", flush=True)


if __name__ == "__main__":
    main()
