# SEA - Sample-efficient adaptation of 12-lead ECG classifiers

I keep one published 1D-ResNet (`resnet1d_wang`) and ask a practical question:

After I train on one public 12-lead ECG archive, how much labelled data from a *new* archive do I need before cheap adaptation stops helping?

This repository is the code and the public scores for that audit. I do not invent a new network. I do not commit waveforms.

The current numbers use three **public** archives only:

| Role | Archive | Country | Usable records |
|---|---|---|---:|
| Source | PTB-XL | Germany | 21,799 |
| Target 1 | Chapman-Shaoxing | China | 10,237 |
| Target 2 | Georgia / G12EC | United States | 10,334 |

Chapman skipped 10 truncated headers (10,247 on disk). Georgia skipped 10 (10,344 on disk). I do not repair a broken header.

I am not reporting a credentialed hospital archive in this repository. The second target here is Georgia.

---

## What I have run so far

1. Train on PTB-XL.
2. Test the same model, unadapted, on Chapman and on Georgia.
3. Adapt with temperature scaling (one scalar on the logits) and linear probing (freeze the backbone, retrain the last layer) at 5, 10, 20, 30 and 50% of each target’s adaptation pool, 10 random repeats.

LoRA is Spring 2027. It is not in these scripts as a Fall method.

---

## Two label tasks

They are not the same thing. Ten SNOMED names are not ten superclasses.

| Task | When I use it | Labels | Where the names come from |
|---|---|---|---|
| **A — diagnostic superclasses** | Milestone 2 only | NORM, MI, STTC, CD, HYP | Official PTB-XL `diagnostic_class` (Wagner 2020; Strodthoff 2021 Table 2) |
| **B — shared SNOMED-CT** | Every source → target pair | AF, IAVB, LBBB, RBBB, NSR, PAC, SB, STach, TAb, LAD | CinC 2021 equivalent codes (Alday; Reyna) |

Task A exists only on PTB-XL. Chapman and Georgia do not have NORM / MI names, so I never score them that way.

Task B keeps a diagnosis only if I can read it on PTB-XL **and** Chapman **and** Georgia. Incomplete RBBB is not counted as RBBB. Everything else is dropped. I never fill NORM or AF because a code is missing.

Forty-four diagnostic SCP codes feed the five superclasses. On the 21,799 PTB-XL records I trained on (multi-label, so the bars do not sum to *n*):

| Superclass | Records | Prevalence |
|---|---:|---:|
| NORM | 9,514 | 43.6% |
| MI | 5,469 | 25.1% |
| STTC | 5,235 | 24.0% |
| CD | 4,898 | 22.5% |
| HYP | 2,649 | 12.2% |

Ten shared SNOMED heads on the same three extracts (again multi-label):

| Label | PTB-XL | Chapman | Georgia |
|---|---:|---:|---:|
| AF | 1,514 | 1,778 | 570 |
| IAVB | 793 | 247 | 768 |
| LBBB | 536 | 205 | 231 |
| RBBB | 541 | 454 | 556 |
| NSR | 16,748 | 1,822 | 1,750 |
| PAC | 398 | 258 | 639 |
| SB | 637 | 3,887 | 1,676 |
| STach | 826 | 1,567 | 1,261 |
| TAb | 0 | 1,875 | 2,303 |
| LAD | 0 | 382 | 939 |

TAb and LAD have **zero** positives on PTB-XL under this map, so those two heads cannot be learned at source. The 0.954 source macro-AUROC is therefore over the eight heads that have both classes on the test fold. NSR is 76.8% on PTB-XL and about 17% on both targets. Chapman is rhythm-heavy (sinus bradycardia 38.0%). Georgia has 21.9% of records with none of the ten labels (PTB-XL 8.6%, Chapman 8.2%).

Jensen–Shannon distance of the ten-label prevalence vectors: PTB-XL vs Chapman **0.267**, PTB-XL vs Georgia **0.257**, Chapman vs Georgia **0.048**.

---

## How I split the data

**Source (PTB-XL).** Official recommended folds. I do not reshuffle patients.

| Split | Folds | *n* | Share of 21,799 |
|---|---|---:|---:|
| Train | 1–8 | 17,418 | 79.9% |
| Validation | 9 | 2,183 | 10.0% |
| Test | 10 | 2,198 | 10.1% |

**Targets (Chapman, Georgia).** I hold out **30%** of each archive for testing. The other **70%** is the adaptation pool. *P* is a share of that pool, not of the whole archive.

| Target | Whole usable *n* | Eval (30%) | Adapt pool (70%) | Labels at 5 / 10 / 20 / 30 / 50% |
|---|---:|---:|---:|---|
| Chapman | 10,237 | 3,071 | 7,166 | 358 / 717 / 1,433 / 2,150 / 3,583 |
| Georgia | 10,334 | 3,100 | 7,234 | 362 / 723 / 1,447 / 2,170 / 3,617 |

Each *P* is drawn 10 times. I report the mean.

**P_min.** First consecutive grid step where the absolute change in the metric falls under τ. I use τ_AUROC = 0.01, τ_Brier = 0.01, τ_ECE = 0.005. Recovering the source 0.954 is never a pass/fail rule.

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

If a waveform later fails to load, the cache can write an all-zero window. On the caches I trained on, that counter is **0 / 21,799**, **0 / 10,237**, and **0 / 10,334**. I do not invent a diagnosis for a missing label. Age coded 300 on PTB-XL is the 90+ privacy flag; age is not a model input, so I do not impute it.

---

## Results in this repo

In-distribution and unadapted transfer (macro-AUROC, 1,000 record-level bootstrap CIs):

| Evaluation | Macro-AUROC | 95% CI |
|---|---:|---|
| PTB-XL superclasses (test fold 10, *n* = 2,198) | 0.912 | 0.905 – 0.920 |
| PTB-XL, shared SNOMED heads (*n* = 2,198; TAb and LAD excluded) | 0.954 | 0.943 – 0.963 |
| Unadapted → Chapman (*n* = 10,237) | 0.858 | 0.851 – 0.864 |
| Unadapted → Georgia (*n* = 10,334) | 0.849 | 0.846 – 0.853 |

Both target intervals exclude chance (0.5). The drop from 0.954 is about **0.10 AUROC** on each archive. Published Table 2 for this backbone on superclasses is about 0.92–0.93. My CPU run sits just under that point estimate; the upper CI (0.920) reaches the lower edge of the band.

Mean linear-probing macro-AUROC over 10 repeats (temperature scaling stays at the unadapted number, as it should):

| *P* | Chapman LP | Georgia LP |
|---|---:|---:|
| 0% (unadapted) | 0.860 | 0.848 |
| 5% | 0.907 | 0.889 |
| 10% | 0.927 | 0.907 |
| 20% | 0.942 | 0.917 |
| 30% | 0.947 | 0.920 |
| 50% | 0.951 | 0.924 |

**P_min (AUROC) = 20%** on both targets. Temperature scaling only slightly improves Brier. ECE got worse.

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

Unadapted transfer (Chapman or Georgia):

```bash
python scripts/eval_ood.py --checkpoint results/checkpoints/ptbxl_snomed_shared.pt --target chapman --task snomed_shared --config configs/local.yaml
```

Temperature scaling vs linear probing, *P* = 5, 10, 20, 30, 50%:

```bash
python scripts/run_adaptation.py --checkpoint results/checkpoints/ptbxl_snomed_shared.pt --source ptbxl --target chapman --task snomed_shared --config configs/local.yaml
```

Same commands with `--target georgia` for the second pair.

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
- A credentialed hospital target. The US numbers here are Georgia.
- That 20% is a hospital sample-size rule. It is the first grid point where the next AUROC step falls under τ = 0.01, on this backbone and these two public targets.
