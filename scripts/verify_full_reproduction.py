from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.image as mpimg


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "expected"
REPRODUCED = ROOT / "reproduced"

RTOL = 1e-6
ATOL = 1e-10

CHECK_DIRS = [
    "stage8_curves",
    "final_closure",
    "legacy_components",
    "figures",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file())


def compare_csv(exp: Path, rep: Path) -> tuple[bool, str]:
    a = pd.read_csv(exp)
    b = pd.read_csv(rep)
    if list(a.columns) != list(b.columns):
        return False, "column mismatch"
    if len(a) != len(b):
        return False, f"row mismatch expected={len(a)} reproduced={len(b)}"
    numeric_cells = 0
    max_abs = 0.0
    max_rel = 0.0
    for col in a.columns:
        a_num = pd.to_numeric(a[col], errors="coerce")
        b_num = pd.to_numeric(b[col], errors="coerce")
        numeric_col = (
            (a[col].isna() | a_num.notna()).all()
            and (b[col].isna() | b_num.notna()).all()
        )
        if numeric_col:
            av = a_num.to_numpy(dtype=float)
            bv = b_num.to_numpy(dtype=float)
            if not np.allclose(av, bv, rtol=RTOL, atol=ATOL, equal_nan=True):
                diff = np.abs(av - bv)
                denom = np.maximum(np.abs(av), ATOL)
                rel = diff / denom
                return (
                    False,
                    f"numeric mismatch column={col} max_abs={np.nanmax(diff):.3g} "
                    f"max_rel={np.nanmax(rel):.3g}",
                )
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any():
                diff = np.abs(av[finite] - bv[finite])
                denom = np.maximum(np.abs(av[finite]), ATOL)
                rel = diff / denom
                max_abs = max(max_abs, float(np.max(diff)))
                max_rel = max(max_rel, float(np.max(rel)))
            numeric_cells += len(av)
        else:
            left = a[col].astype("string").fillna("<NA>").tolist()
            right = b[col].astype("string").fillna("<NA>").tolist()
            if left != right:
                return False, f"categorical mismatch column={col}"
    return True, f"csv ok numeric_cells={numeric_cells} max_abs={max_abs:.3g} max_rel={max_rel:.3g}"


def compare_json_value(a, b, path: str = "$") -> tuple[bool, str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, f"json key mismatch at {path}"
        for key in sorted(a):
            ok, msg = compare_json_value(a[key], b[key], f"{path}.{key}")
            if not ok:
                return ok, msg
        return True, "json ok"
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f"json length mismatch at {path}"
        for i, (av, bv) in enumerate(zip(a, b)):
            ok, msg = compare_json_value(av, bv, f"{path}[{i}]")
            if not ok:
                return ok, msg
        return True, "json ok"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(float(a)) and math.isnan(float(b)):
            return True, "json ok"
        if math.isclose(float(a), float(b), rel_tol=RTOL, abs_tol=ATOL):
            return True, "json ok"
        return False, f"json numeric mismatch at {path}: {a} != {b}"
    if a == b:
        return True, "json ok"
    return False, f"json mismatch at {path}: {a!r} != {b!r}"


def compare_json(exp: Path, rep: Path) -> tuple[bool, str]:
    a = json.loads(exp.read_text(encoding="utf-8"))
    b = json.loads(rep.read_text(encoding="utf-8"))
    return compare_json_value(a, b)


def compare_png(exp: Path, rep: Path) -> tuple[bool, str]:
    if rep.stat().st_size <= 0:
        return False, "empty reproduced PNG"
    a = mpimg.imread(exp)
    b = mpimg.imread(rep)
    if a.shape != b.shape:
        return False, f"PNG shape mismatch expected={a.shape} reproduced={b.shape}"
    return True, f"png ok shape={a.shape}"


def compare_text_normalized(exp: Path, rep: Path) -> tuple[bool, str]:
    a = exp.read_text(encoding="utf-8").replace("\r\n", "\n")
    b = rep.read_text(encoding="utf-8").replace("\r\n", "\n")
    if a != b:
        return False, "normalized text mismatch"
    return True, "text ok"


def compare_file(exp: Path, rep: Path) -> tuple[bool, str]:
    suffix = exp.suffix.lower()
    if suffix == ".csv":
        return compare_csv(exp, rep)
    if suffix == ".json":
        return compare_json(exp, rep)
    if suffix == ".png":
        return compare_png(exp, rep)
    if suffix in {".txt", ".md", ".tsv"}:
        return compare_text_normalized(exp, rep)
    return (sha256(exp) == sha256(rep), "byte hash ok")


def main() -> None:
    failures = []
    checked = 0
    summaries = []
    for d in CHECK_DIRS:
        exp_dir = EXPECTED / d
        rep_dir = REPRODUCED / d
        if not exp_dir.exists():
            failures.append(f"missing expected directory: {exp_dir}")
            continue
        for exp in iter_files(exp_dir):
            rel = exp.relative_to(exp_dir)
            rep = rep_dir / rel
            if not rep.exists():
                failures.append(f"missing reproduced file: {d}/{rel.as_posix()}")
                continue
            checked += 1
            ok, msg = compare_file(exp, rep)
            summaries.append((d, rel.as_posix(), msg))
            if not ok:
                failures.append(f"{d}/{rel.as_posix()}: {msg}")
    if failures:
        print("FULL_REPRODUCTION: FAIL")
        for failure in failures:
            print(" -", failure)
        sys.exit(1)
    print(
        f"FULL_REPRODUCTION: PASS ({checked} files; CSV numeric tolerance rtol={RTOL}, "
        f"JSON semantic tolerance, PNG dimensions verified)"
    )


if __name__ == "__main__":
    main()
