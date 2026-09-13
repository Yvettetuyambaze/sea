# Label Harmonization

Unmapped statements are dropped. They are never treated as equivalent to a
mapped class (proposal, Section V.A).

## Task A — PTB-XL diagnostic superclass (Milestone 2)

Used only for the in-distribution PTB-XL baseline against Table 2.

| Superclass | Meaning | Source |
|---|---|---|
| NORM | Normal ECG | Wagner et al. 2020; official `scp_statements.csv` |
| MI | Myocardial infarction | same |
| STTC | ST/T change | same |
| CD | Conduction disturbance | same |
| HYP | Hypertrophy | same |

Aggregation follows the official PhysioNet example: diagnostic SCP statements
are mapped through `diagnostic_class`. Folds 1–8 / 9 / 10 are train / val / test
as in Strodthoff et al. (2021).

## Task B — Shared SNOMED-CT set (Milestones 3+)

Used for every source→target pair so ranking metrics are comparable.

| Name | Canonical SNOMED-CT | Scored equivalents | Present in PTB-XL / Chapman / Georgia |
|---|---|---|---|
| AF | 164889003 | — | yes / yes / yes |
| IAVB | 270492004 | — | yes / yes / yes |
| LBBB | 164909002 | 733534002 | yes / yes / yes |
| RBBB | 59118001 | 713427006 | yes (as CRBBB) / yes / yes |
| NSR | 426783006 | — | yes / yes / yes |
| PAC | 284470004 | 63593006 | yes / yes / yes |
| SB | 426177001 | — | yes / yes / yes |
| STach | 427084000 | — | yes / yes / yes |
| TAb | 164934002 | — | yes / yes / yes |
| LAD | 39732003 | — | yes / yes / yes |

Equivalence rules follow the PhysioNet/CinC 2021 scoring notes (Reyna et al.).
PTB-XL SCP codes are converted with `src/sea/data/labels.py` (`SCP_TO_SNOMED`).
Chapman, Georgia, and CPSC labels are read from WFDB `Dx:` header fields.

## MIMIC-IV-ECG (pending credentialed access)

The loader expects `data/mimic-iv-ecg/manifest.json` with per-record SNOMED
codes. Until that file exists, Georgia (US hospital 12-lead, Challenge 2021)
is the public stand-in for the second OOD target. See `docs/mimic_setup.md`.
