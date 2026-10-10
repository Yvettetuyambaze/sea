# SEA - Sample-efficient adaptation of 12-lead ECG classifiers

Sample-efficient adaptation of 12-lead ECG classifiers.

I keep one published 1D-ResNet (`resnet1d_wang`) and ask a practical question: after I train on one public ECG archive, how much labelled data from a *new* archive do I need before cheap adaptation stops helping?

This repository is the code and the public scores for that audit. I do not invent a new network. I do not commit waveforms.

**What I have run so far**

1. Train on **PTB-XL** (Germany).
2. Test unadapted on **Chapman-Shaoxing** (China) and **Georgia / G12EC** (United States).
3. Adapt with temperature scaling (one scalar on the logits) and linear probing (freeze the backbone, retrain the last layer) at 5, 10, 20, 30 and 50% of each target’s adaptation pool.

Georgia is the second target in this repo. I am not using a hospital dump or a credentialed archive in the current numbers.

---

## Two label tasks

They are not the same thing, and ten SNOMED names are not ten superclasses.

| Task | When I use it | Labels |
|---|---|---|
| **A — diagnostic superclasses** | Milestone 2 only, PTB-XL | NORM, MI, STTC, CD, HYP |
| **B — shared SNOMED-CT** | Every source → target pair | AF, IAVB, LBBB, RBBB, NSR, PAC, SB, STach, TAb, LAD |

Task A follows the official PTB-XL `diagnostic_class` map (Wagner 2020; Strodthoff 2021 Table 2). Task B follows the CinC 2021 equivalent codes. Unmapped diagnoses are dropped. I never fill NORM or AF because a code is missing.

How many recordings I actually used: PTB-XL **21,799**; Chapman **10,237** (10 bad headers skipped); Georgia **10,334** (10 skipped).

---

## Waveforms

Every usable ECG is forced to the same tensor: **12 leads × 10 seconds × 100 Hz**.

1. Read the WFDB (or CSV) file.
2. Skip a truncated header. Do not repair it.
3. Put leads in I, II, III, aVR, aVL, aVF, V1–V6 order.
4. Resample to 100 Hz.
5. Pad or centre-crop to 1,000 samples.
6. Replace NaN / Inf with 0.
7. Z-score each lead.

If a waveform later fails to load, the cache can write an all-zero window. On the caches I trained on, that counter is **0 / 21,799**, **0 / 10,237**, and **0 / 10,334**. I do not invent a diagnosis for a missing label.

---

## Results in this repo

| Evaluation | Macro-AUROC | 95% CI |
|---|---:|---|
| PTB-XL superclasses (test fold 10, *n* = 2,198) | 0.912 | 0.905 – 0.920 |
| PTB-XL, 10 SNOMED heads (*n* = 2,198; TAb and LAD excluded, no positives) | 0.954 | 0.943 – 0.963 |
| Unadapted → Chapman (*n* = 10,237) | 0.858 | 0.851 – 0.864 |
| Unadapted → Georgia (*n* = 10,334) | 0.849 | 0.846 – 0.853 |

Both target intervals exclude chance (0.5). The drop from 0.954 is about 0.10 AUROC on each archive.

I hold out 30% of each target for testing. The other 70% is the adaptation pool. *P* is a share of that pool, not of the whole archive. Linear probing raises ranking from 5% labelled target data. Under τ = 0.01 on consecutive AUROC steps, **P_min = 20%** on both Chapman and Georgia. Temperature scaling leaves AUROC where it is (as it should) and only slightly improves Brier; ECE got worse.

JSON scores: `results/eval/`. Figures: `results/figures/`. Checkpoints and caches stay on my machine.

---

## Setup

Python 3.10 or newer. A GPU is faster. CPU is what I used for the numbers above.

```bash
pip install -e .
```

Or `pip install -r requirements.txt` and put `src` on `PYTHONPATH`.

---

## Data (local only)

Put archives under `data/`. That folder is gitignored.

```bash
python scripts/download_data.py --config configs/local.yaml --datasets ptbxl chapman georgia
```

If I already have a zip or an extracted PhysioNet folder:

```bash
python scripts/prepare_uploads.py --src /path/to/uploads --dst data
```

That writes `configs/runtime.yaml` (also gitignored).

---

## Train, evaluate, adapt

PTB-XL superclasses (official folds 1–8 / 9 / 10):

```bash
python scripts/train_source.py --dataset ptbxl --task superclass --config configs/local.yaml
```

Same backbone, ten SNOMED labels:

```bash
python scripts/train_source.py --dataset ptbxl --task snomed_shared --config configs/local.yaml
```

Unadapted transfer (Chapman or `georgia`):

```bash
python scripts/eval_ood.py --checkpoint results/checkpoints/ptbxl_snomed_shared.pt --target chapman --task snomed_shared --config configs/local.yaml
```

Temperature scaling vs linear probing, *P* = 5, 10, 20, 30, 50%:

```bash
python scripts/run_adaptation.py --checkpoint results/checkpoints/ptbxl_snomed_shared.pt --source ptbxl --target chapman --task snomed_shared --config configs/local.yaml
```

Label counts, prevalence tests, and missing-value audit:

```bash
python scripts/dataset_label_audit.py
python scripts/make_preprocessing_figures.py
```

End-to-end on a laptop:

```bash
python scripts/run_local.py
```

GPU machines can pass `--config configs/colab.yaml` to the same scripts. There is no notebook in this repo.

---

## Tests

```bash
python tests/test_unit.py
```

---

## Layout

```text
configs/          default.yaml, local.yaml, colab.yaml
scripts/          download, train, eval, adapt, label audit
src/sea/models/   resnet1d_wang
src/sea/data/     loaders, labels, preprocess
src/sea/adapt/    temperature scaling, linear probing
src/sea/          train, metrics, eval
results/eval/     published JSON scores
results/figures/  plots
tests/
```

Write-ups stay in `docs/` on my machine. They are not on GitHub.

---

## What I have not claimed

- A new architecture.
- Reverse arrows (Chapman or Georgia as the source).
- That 20% is a hospital sample-size rule. It is the first grid point where the next AUROC step falls under τ = 0.01, on this backbone and these two public targets.
