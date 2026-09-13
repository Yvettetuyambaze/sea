# MIMIC-IV-ECG setup (credentialed)

Credentialing is already requested. This repository does not redistribute
restricted recordings.

## When access is approved

1. Download MIMIC-IV-ECG from PhysioNet into `data/mimic-iv-ecg/`.
2. Write `data/mimic-iv-ecg/manifest.json`:

```json
{
  "records": [
    {"path": "data/mimic-iv-ecg/records/p1000/p10000032/s40000000/40000000", "snomed": ["164889003", "426783006"]}
  ]
}
```

3. Re-run the M3 pipeline with Georgia replaced by MIMIC:

```text
python scripts/run_milestone3.py --skip-download --skip-m2 --targets chapman mimic
```

The same SNOMED shared label table is used. Unmapped machine-generated
statements stay unmapped.
