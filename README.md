# Sample-efficient ECG adaptation (SEA)

Fall 2026 Engineering Research Project (04-990). The question is how much
labelled *target* data, as a percentage of the adaptation pool, is needed
before calibration and parameter adaptation stop helping a 12-lead ECG
classifier that was trained on a different public dataset.

Proposal: `Proposal_Yvette (1).pdf`.

## Milestones implemented

| ID | Deliverable | Status |
|---|---|---|
| M1 | Literature review (≥20 papers) | [docs/literature_review.md](docs/literature_review.md) — 32 papers |
| M2 | PTB-XL 1D-ResNet + three-dataset pipeline | Code ready; train with `scripts/train_source.py` |
| M3 | Unadapted OOD + Temperature Scaling + Linear Probing | Code ready; `scripts/run_milestone3.py` |

MIMIC-IV-ECG credentialing is in progress. Until it arrives, **Georgia**
(PhysioNet/CinC 2021, US 12-lead) is the second public target so
PTB-XL → {Chapman, Georgia} is a full source-to-both-targets rotation
(proposal Section VII contingency).

## Quick start

```text
pip install -r requirements.txt
python scripts/run_milestone3.py
```

That downloads PTB-XL, Chapman-Shaoxing, and Georgia, trains the PTB-XL
superclass baseline (Table 2 check) and a shared-SNOMED source model,
then runs unadapted OOD evaluation and adaptation at
`P ∈ {5, 10, 20, 30, 50}%` with 10 Monte Carlo repetitions.

Individual steps:

```text
python scripts/download_data.py --datasets ptbxl chapman georgia
python scripts/train_source.py --dataset ptbxl --task superclass
python scripts/train_source.py --dataset ptbxl --task snomed_shared
python scripts/eval_ood.py --checkpoint results/checkpoints/ptbxl_snomed_shared.pt --target chapman
python scripts/run_adaptation.py --checkpoint results/checkpoints/ptbxl_snomed_shared.pt --target chapman
```

When MIMIC files are on disk, see [docs/mimic_setup.md](docs/mimic_setup.md).

## Design choices

- Backbone: `resnet1d_wang` (Strodthoff et al., 2021; Wang et al., 2017).
- M2 task: PTB-XL diagnostic superclasses, official folds 1–8 / 9 / 10.
- M3 task: 10 shared SNOMED-CT labels present in PTB-XL, Chapman, and Georgia.
  Unmapped labels are dropped ([docs/label_harmonization.md](docs/label_harmonization.md)).
- Temperature scaling changes Brier/ECE only; AUROC is a manipulation check.
- Linear probing can change discrimination. LoRA is specified for Spring 2027.
- Restricted recordings are never committed.

## Repository layout

```text
configs/default.yaml
docs/literature_review.md
src/sea/data/          preprocess, labels, downloaders
src/sea/models/        1D-ResNet
src/sea/adapt/         temperature scaling, linear probing
scripts/               download, train, OOD eval, adaptation, M3 driver
```
