# SEA

Train a 12-lead ECG model on one dataset, test it on another, then adapt it with a small slice of labelled target data.

The backbone is `resnet1d_wang`. Adaptation is either temperature scaling (calibration only) or linear probing (freeze the backbone, retrain the head).

## Setup

Python 3.10+. A GPU helps; CPU works, just slower.

```bash
pip install -e .
```

Or `pip install -r requirements.txt` and keep `src` on `PYTHONPATH`.

## Data

Waveforms live in `data/` and are not committed.

```bash
python scripts/download_data.py --config configs/local.yaml --datasets ptbxl chapman
```

MIMIC-IV-ECG is credentialed. Set `PHYSIONET_USER` and `PHYSIONET_PASSWORD` in the environment if you need the tables. The default is CSV tables plus streamed waveforms — not the full zip.

If the files are already on disk (Drive, a zip, an extracted PhysioNet folder):

```bash
python scripts/prepare_uploads.py --src /path/to/uploads --dst data
```

That writes `configs/runtime.yaml` with the paths it found.

## Configs

| File | Use |
|---|---|
| `configs/default.yaml` | Shared defaults |
| `configs/local.yaml` | Smaller batches for a local machine |
| `configs/colab.yaml` | GPU / Drive paths |
| `configs/runtime.yaml` | Generated locally, gitignored |

## Train

PTB-XL diagnostic superclasses (official folds 1–8 / 9 / 10):

```bash
python scripts/train_source.py --dataset ptbxl --task superclass --config configs/local.yaml
```

Same backbone, 10 shared SNOMED-CT labels (needed before OOD / adaptation):

```bash
python scripts/train_source.py --dataset ptbxl --task snomed_shared --config configs/local.yaml
```

Checkpoints go to `results/checkpoints/` (gitignored).

## Evaluate and adapt

Unadapted transfer:

```bash
python scripts/eval_ood.py --checkpoint results/checkpoints/ptbxl_snomed_shared.pt --target chapman --task snomed_shared --config configs/local.yaml
```

Temperature scaling vs linear probing at P = 5, 10, 20, 30, 50%:

```bash
python scripts/run_adaptation.py --checkpoint results/checkpoints/ptbxl_snomed_shared.pt --source ptbxl --target chapman --task snomed_shared --config configs/local.yaml
```

## Drivers

End-to-end locally (`configs/local.yaml`):

```bash
python scripts/run_local.py
```

Skip steps you already finished:

```bash
python scripts/run_local.py --skip-m2
python scripts/run_local.py --skip-source
python scripts/run_local.py --skip-ood
```

Colab: open `notebooks/colab_milestones.ipynb` with a GPU runtime.

```bash
python scripts/run_milestone2.py --config configs/colab.yaml
python scripts/run_milestone3.py --config configs/colab.yaml --skip-m2 --targets chapman
python scripts/summarize_results.py
```

## Tests

```bash
python tests/test_unit.py
```

## Layout

```text
configs/               YAML configs
notebooks/             Colab driver
scripts/               download, train, eval, adapt
src/sea/models/        1D-ResNet
src/sea/data/          loaders, labels, preprocess, download
src/sea/adapt/         temperature scaling, linear probing
src/sea/               train, metrics, eval
tests/
```
