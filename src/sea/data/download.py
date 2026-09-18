"""Download ECG datasets.

Public sets (PTB-XL, Chapman, Georgia, CPSC) need no login. MIMIC-IV-ECG is
credentialed: credentials come from PHYSIONET_USER / PHYSIONET_PASSWORD and
are never written to disk or logs. Waveforms stay under data/ (gitignored).
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

USER_AGENT = "sea-ecg-research/0.1 (academic; CMU-Africa 04-990)"
PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3"
CHALLENGE_BASE = "https://physionet.org/files/challenge-2021/1.0.3/training"
MIMIC_BASE = "https://physionet.org/files/mimic-iv-ecg/1.0"


def physionet_credentials() -> tuple[str, str]:
    user = os.environ.get("PHYSIONET_USER", "").strip()
    password = os.environ.get("PHYSIONET_PASSWORD", "").strip()
    if not user or not password:
        raise RuntimeError(
            "MIMIC-IV-ECG download needs PHYSIONET_USER and PHYSIONET_PASSWORD "
            "in the environment (do not paste the password into chat). "
            "Also sign the project DUA at https://physionet.org/content/mimic-iv-ecg/1.0/"
        )
    return user, password


def _auth_opener(user: str | None = None, password: str | None = None):
    if not user:
        return urllib.request.build_opener()
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, "https://physionet.org", user, password or "")
    return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(manager))


def _open(url: str, user: str | None = None, password: str | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return _auth_opener(user, password).open(req, timeout=180)


def download_file(
    url: str,
    dest: Path,
    skip_existing: bool = True,
    user: str | None = None,
    password: str | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return dest
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        return _curl_download(curl, url, dest, user, password)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _open(url, user, password) as src, tmp.open("wb") as out:
        shutil.copyfileobj(src, out)
    tmp.replace(dest)
    return dest


def _read_text(url: str) -> str:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        result = subprocess.run(
            [curl, "-L", "--max-time", "60", "-A", USER_AGENT, url],
            check=True,
            capture_output=True,
        )
        return result.stdout.decode("utf-8", errors="replace")
    with _open(url) as handle:
        return handle.read().decode("utf-8", errors="replace")


def download_ptbxl(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        download_file(f"{PTBXL_BASE}/{name}", root / name)
    records = _read_text(f"{PTBXL_BASE}/RECORDS").splitlines()
    lr_records = [r.strip() for r in records if r.strip().startswith("records100/")]
    jobs = []
    for rec in lr_records:
        for ext in (".hea", ".dat"):
            rel = f"{rec}{ext}"
            jobs.append((f"{PTBXL_BASE}/{rel}", root / rel))
    _download_many(jobs, desc="PTB-XL 100 Hz")
    return root


def _download_many(
    jobs: list[tuple[str, Path]],
    desc: str,
    workers: int = 12,
    user: str | None = None,
    password: str | None = None,
) -> None:
    pending = [(url, dest) for url, dest in jobs if not (dest.exists() and dest.stat().st_size > 0)]
    if not pending:
        print(f"{desc}: already present ({len(jobs)} files)")
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, url, dest, True, user, password): dest for url, dest in pending
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            fut.result()


def maybe_gunzip(path: Path) -> Path:
    if path.suffix != ".gz":
        return path
    out = path.with_suffix("")
    if not out.exists():
        with gzip.open(path, "rb") as src, out.open("wb") as dest:
            shutil.copyfileobj(src, dest)
    return out


DATASET_SOURCES = {
    "ptbxl": ("official", PTBXL_BASE),
    "chapman": ("challenge-2021", "chapman_shaoxing"),
    "georgia": ("challenge-2021", "georgia"),
    "cpsc": ("challenge-2021", "cpsc_2018"),
}

PTBXL_ZIP = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)


def extract_ptbxl_from_zip(zip_path: Path, root: Path) -> Path:
    """Keep official tables + 100 Hz records only (skip 500 Hz to save disk)."""
    import zipfile

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = Path(zip_path)
    print(f"Extracting PTB-XL 100 Hz + tables from {zip_path.name} -> {root}")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        top = names[0].split("/")[0] + "/" if names and "/" in names[0] else ""
        if top and not all(n.startswith(top) or n.rstrip("/") == top.rstrip("/") for n in names[:30]):
            top = ""
        keep = []
        for name in names:
            rel = name[len(top) :] if top and name.startswith(top) else name
            rel_norm = rel.replace("\\", "/")
            if rel_norm.endswith("ptbxl_database.csv") or rel_norm.endswith("scp_statements.csv"):
                keep.append((name, root / Path(rel_norm).name))
            elif "records100/" in rel_norm:
                keep.append((name, root / rel_norm))
        for src, dest in tqdm(keep, desc="extract PTB-XL"):
            if src.endswith("/"):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            with archive.open(src) as incoming, dest.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    return root


def _ptbxl_extract_complete(root: Path) -> bool:
    csv_path = root / "ptbxl_database.csv"
    if not csv_path.exists() or not (root / "scp_statements.csv").exists():
        return False
    n_csv = max(0, sum(1 for _ in csv_path.open("r", encoding="utf-8", errors="replace")) - 1)
    n_dat = sum(1 for _ in root.glob("records100/**/*.dat"))
    if n_dat < max(100, int(0.98 * n_csv)):
        print(f"PTB-XL extract incomplete: {n_dat} waveforms vs {n_csv} table rows")
        return False
    return True


def download_ptbxl_zip(root: Path) -> Path:
    """Faster than 40k small-file GETs: one official zip, then keep 100 Hz + tables."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if _ptbxl_extract_complete(root):
        print(f"PTB-XL already present at {root}")
        return root
    zip_path = root.parent / "ptb-xl-1.0.3.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        print(f"Downloading PTB-XL zip -> {zip_path}")
        _download_with_progress(PTBXL_ZIP, zip_path)
    return extract_ptbxl_from_zip(zip_path, root)


def _download_with_progress(
    url: str,
    dest: Path,
    user: str | None = None,
    password: str | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        return _curl_download(curl, url, dest, user, password)
    tmp = dest.with_suffix(dest.suffix + ".part")
    already = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if already:
        req.add_header("Range", f"bytes={already}-")
    with _auth_opener(user, password).open(req, timeout=180) as src, tmp.open("ab" if already else "wb") as out:
        total = int(src.headers.get("Content-Length") or 0) + already
        bar = tqdm(total=total, initial=already, unit="B", unit_scale=True, desc=dest.name)
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            bar.update(len(chunk))
        bar.close()
    tmp.replace(dest)
    return dest


def _curl_download(
    curl: str,
    url: str,
    dest: Path,
    user: str | None,
    password: str | None,
) -> Path:
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        curl,
        "-C",
        "-",
        "-L",
        "--retry",
        "8",
        "--retry-all-errors",
        "-A",
        USER_AGENT,
        "-o",
        str(tmp),
        url,
    ]
    if user:
        cmd[1:1] = ["-u", f"{user}:{password or ''}"]
    print(f"curl resume download -> {dest.name}")
    subprocess.run(cmd, check=True)
    tmp.replace(dest)
    return dest


def download_challenge_subset(name: str, dest: Path) -> Path:
    """Challenge-2021 subsets live in g1, g2, ... folders, not a flat RECORDS file."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    base = f"{CHALLENGE_BASE}/{name}"
    groups = _list_groups(base)
    jobs = []
    for group in groups:
        records = _read_text(f"{base}/{group}/RECORDS").splitlines()
        for rec in records:
            rec = rec.strip()
            if not rec:
                continue
            rec_name = Path(rec).name
            for ext in (".hea", ".mat"):
                jobs.append((f"{base}/{group}/{rec_name}{ext}", dest / f"{rec_name}{ext}"))
    _download_many(jobs, desc=name)
    return dest


def _list_groups(base: str) -> list[str]:
    try:
        html = _read_text(base + "/")
    except Exception:
        html = _read_text(base)
    groups = []
    for token in html.replace('"', " ").replace("'", " ").split():
        token = token.strip("/")
        if token.startswith("g") and token[1:].isdigit():
            groups.append(token)
    groups = sorted(set(groups), key=lambda g: int(g[1:]))
    if not groups:
        # some subsets (cpsc) may be flat
        try:
            _read_text(f"{base}/RECORDS")
            return [""]
        except Exception as exc:
            raise FileNotFoundError(f"Could not list groups under {base}") from exc
    return groups


def download_dataset(name: str, dest: Path, cfg: dict | None = None, tables_only: bool = True) -> Path:
    name = name.lower()
    if name == "ptbxl":
        try:
            return download_ptbxl_zip(dest)
        except Exception as exc:
            print(f"Zip download failed ({exc}); falling back to per-file GET")
            return download_ptbxl(dest)
    if name == "mimic":
        from .mimic_io import download_mimic_iv_ecg

        return download_mimic_iv_ecg(dest, cfg or {}, tables_only=tables_only)
    if name not in DATASET_SOURCES:
        raise ValueError(f"Unknown dataset {name}. Options: {list(DATASET_SOURCES) + ['mimic']}")
    _, folder = DATASET_SOURCES[name]
    return download_challenge_subset(folder, dest)
