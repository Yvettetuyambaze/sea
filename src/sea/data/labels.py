"""Documented label harmonization.

PTB-XL diagnostic superclasses are used only for the in-distribution M2
benchmark (Table 2 of the proposal). Cross-dataset experiments use a shared
SNOMED-CT subset that is present with usable counts in PTB-XL,
Chapman-Shaoxing, and MIMIC-IV-ECG. Unmapped statements are dropped, never treated as equivalent.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

SUPERCLASS_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

# PTB-XL SCP statement -> SNOMED-CT, following the PhysioNet/CinC 2020–2021
# conversion used by Alday et al. and Reyna et al. Only codes that participate
# in the shared scored set are listed; all other SCP statements remain unmapped.
SCP_TO_SNOMED: dict[str, str] = {
    "AFIB": "164889003",
    "1AVB": "270492004",
    "CLBBB": "164909002",
    "LAFB": "445118002",
    "LPFB": "445211001",
    "IRBBB": "713426002",
    "CRBBB": "713427006",
    "_RBBB": "59118001",
    "IVCD": "698252002",
    "SR": "426783006",
    "PAC": "284470004",
    "SVPB": "63593006",
    "SBRAD": "426177001",
    "STACH": "427084000",
    "SARRH": "427393009",
    "TAB": "164934002",
    "INVT": "59931005",
    "LAD": "39732003",
    "RAD": "47665007",
    "LVOLT": "251146004",
    "LNGQT": "111975006",
    "QWAVE": "164917005",
    "PACE": "10370003",
    "AFLT": "164890007",
    "PVC": "427172004",
    "VPB": "17338001",
}

# Chapman original rhythm names -> SNOMED (Zheng et al. 2020).
CHAPMAN_RHYTHM_TO_SNOMED: dict[str, str] = {
    "SR": "426783006",
    "SB": "426177001",
    "AFIB": "164889003",
    "AF": "164890007",
    "ST": "427084000",
    "SVT": "426761007",
    "AT": "270492004",
}

EQUIVALENT_SNOMED: dict[str, str] = {
    "733534002": "164909002",  # CLBBB scored as LBBB
    "713427006": "59118001",  # CRBBB scored as RBBB
    "63593006": "284470004",  # SVPB scored as PAC
    "17338001": "427172004",  # VPB scored as PVC
}


def canonical_snomed(code: str) -> str:
    code = str(code).strip()
    return EQUIVALENT_SNOMED.get(code, code)


def shared_snomed_table(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cfg["snomed_shared"])


def shared_label_names(cfg: dict[str, Any]) -> list[str]:
    return [row["name"] for row in shared_snomed_table(cfg)]


def codes_to_multihot(codes: Iterable[str], label_rows: list[dict[str, Any]]) -> list[int]:
    canon = {canonical_snomed(c) for c in codes}
    vec = []
    for row in label_rows:
        equivalents = {canonical_snomed(c) for c in row["equivalents"]}
        vec.append(int(bool(canon & equivalents)))
    return vec


def scp_dict_to_snomed(scp_codes: dict[str, Any]) -> list[str]:
    out = []
    for key in scp_codes:
        if key in SCP_TO_SNOMED:
            out.append(SCP_TO_SNOMED[key])
    return out


def scp_dict_to_superclasses(scp_codes: dict[str, Any], agg_map: dict[str, str]) -> list[str]:
    classes = []
    for key in scp_codes:
        if key in agg_map:
            classes.append(agg_map[key])
    return sorted(set(classes))


# Machine-generated ECG statements -> shared SNOMED. Applied locally.
# More specific sinus patterns must precede generic sinus rhythm.
REPORT_PHRASES: list[tuple[str, str]] = [
    (r"sinus\s+tachycardia", "427084000"),
    (r"sinus\s+bradycardia", "426177001"),
    (r"atrial\s+fibrillation|\ba\s*[\-/]?\s*fib(?:rillation)?\b|\bafib\b", "164889003"),
    (r"left\s+bundle\s+branch\s+block|\bclbbb\b|\blbbb\b", "164909002"),
    (r"right\s+bundle\s+branch\s+block|\bcrbbb\b|\brbbb\b", "59118001"),
    (r"(?:first|1st|i)\s*(?:degree|deg\.?)?\s*(?:a\s*[\-/]?\s*v|av)\s*block", "270492004"),
    (r"premature\s+atrial|atrial\s+premature|\bpacs?\b|\bapcs?\b", "284470004"),
    (r"left\s+axis\s+deviation|\blad\b", "39732003"),
    (r"t[\s\-]*wave\s+(?:abnormal|invers|change)|nonspecific\s+t[\s\-]*wave", "164934002"),
    (r"(?:normal\s+)?sinus\s+rhythm|\bnsr\b", "426783006"),
]


def report_text_to_snomed(text: str) -> list[str]:
    """Map concatenated machine-report lines to shared SNOMED codes.

    Unmapped phrases are dropped. Input is never logged.
    """
    if not text:
        return []
    lowered = text.lower()
    codes = []
    for pattern, code in REPORT_PHRASES:
        if re.search(pattern, lowered):
            codes.append(code)
    return codes
