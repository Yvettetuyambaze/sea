"""Find PTB-XL / Chapman / MIMIC roots inside uploaded Drive folders."""

from __future__ import annotations

import zipfile
from pathlib import Path


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


SKIP_DIR_NAMES = {
    "files",
    "records100",
    "records500",
    "records1000",
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
}


def walk_dirs(root: Path, max_depth: int = 4) -> list[Path]:
    root = Path(root)
    found = [root]
    if not _is_dir(root):
        return found
    pending = [(root, 0)]
    while pending:
        current, depth = pending.pop()
        if depth >= max_depth:
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if _is_dir(child) and child.name.lower() not in SKIP_DIR_NAMES:
                found.append(child)
                pending.append((child, depth + 1))
    return found


def find_file(root: Path, name: str, max_depth: int = 4) -> Path | None:
    target = name.lower()
    for folder in walk_dirs(root, max_depth=max_depth):
        try:
            for child in folder.iterdir():
                if child.is_file() and child.name.lower() == target:
                    return child
        except OSError:
            continue
    return None


def find_ptbxl_root(root: Path) -> Path | None:
    csv = find_file(root, "ptbxl_database.csv")
    if csv is None:
        return None
    return csv.parent


def find_mimic_root(root: Path) -> Path | None:
    csv = find_file(root, "record_list.csv")
    if csv is None:
        csv = find_file(root, "machine_measurements.csv")
    if csv is None:
        return None
    return csv.parent


def find_chapman_root(root: Path) -> Path | None:
    root = Path(root)
    named = []
    for folder in walk_dirs(root, max_depth=5):
        lower = folder.name.lower().replace("-", "_")
        if "chapman" in lower or "shaoxing" in lower:
            named.append(folder)
        for diag in ("Diagnostics.xlsx", "Diagnostics.csv", "diagnostics.csv"):
            if (folder / diag).exists():
                return folder
    for folder in named + walk_dirs(root, max_depth=5):
        try:
            if next(folder.glob("JS*.hea"), None) is not None:
                return folder
            if next(folder.glob("g*/JS*.hea"), None) is not None:
                return folder
        except OSError:
            continue
        try:
            hit = next(folder.rglob("JS*.hea"), None)
        except OSError:
            hit = None
        if hit is not None:
            if hit.parent.name.lower().startswith("g") and hit.parent.name[1:].isdigit():
                return hit.parent.parent
            return hit.parent
    return named[0] if named else None


def find_zip(root: Path, needles: tuple[str, ...]) -> Path | None:
    root = Path(root)
    matches: list[Path] = []
    for folder in walk_dirs(root, max_depth=3):
        try:
            children = list(folder.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_file():
                continue
            name = child.name.lower()
            if not (name.endswith(".zip") or name.endswith(".tar.gz") or name.endswith(".tgz")):
                continue
            if any(n in name for n in needles):
                matches.append(child)
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return matches[0]


def zip_member_root(names: list[str]) -> str:
    if not names:
        return ""
    first = names[0].replace("\\", "/")
    top = first.split("/")[0] + "/" if "/" in first else ""
    if top and all(n.replace("\\", "/").startswith(top) or n.replace("\\", "/").rstrip("/") == top.rstrip("/") for n in names[:50]):
        return top
    return ""


def list_zip_names(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.namelist()


def discover_roots(src: Path) -> dict[str, Path]:
    """Return dataset name -> existing directory. Missing datasets are omitted."""
    src = Path(src)
    found: dict[str, Path] = {}
    ptbxl = find_ptbxl_root(src)
    if ptbxl is not None:
        found["ptbxl"] = ptbxl
    chapman = find_chapman_root(src)
    if chapman is not None:
        found["chapman"] = chapman
    mimic = find_mimic_root(src)
    if mimic is not None:
        found["mimic"] = mimic
    return found
