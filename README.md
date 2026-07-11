# Paper B Full Reproduction Package v1.0.9

This package is the manuscript-facing reproducibility release for Paper B.
It is designed to regenerate the packaged Stage 8 curve-bootstrap intermediates,
fitted-saturation closure tables,
utility summaries, legacy supporting JSON outputs, and five figure PNGs from the
bundled canonical inputs.

## Reproduction Contract

The archive itself is protected by byte-level SHA-256 checksums in
`checksums_sha256.txt`.

Regenerated analysis outputs are checked by `scripts/verify_full_reproduction.py`
using a cross-platform reproducibility contract:

- CSV files: identical columns, identical row counts, categorical fields exact,
  numeric fields within `rtol=1e-6` and `atol=1e-10`.
- JSON files: semantic comparison with the same numeric tolerance.
- PNG files: generated and dimension-checked; scientific figure validation is based
  on the underlying `figure_data/` CSV files rather than platform-dependent PNG bytes.

This avoids false failures from Windows/Linux line endings, floating-point last-bit
differences, and Matplotlib raster metadata/rendering differences.

The contract starts from bundled canonical standardized inputs in `data/processed/`.
It does not claim to redownload or relicense third-party raw source data. The raw
acquisition/standardization step is outside this public release; the standardized
canonical inputs are included so the manuscript-facing results can be reproduced
exactly.

## Contents

- `data/processed/`: canonical standardized item-level inputs for CIFAR-10H,
  ChaosNLI, and Snapshot Serengeti. Large CSV inputs may be stored as `.csv.gz`
  so repository hosts with per-file upload limits can accept the package.
- `data/legacy_components/`: explicit legacy inputs for Bingol, Snow, and Nitti.
- `scripts/`: deterministic reproduction scripts.
- `expected/`: expected manuscript-facing outputs.
- `manuscript/`: IEEE Access manuscript DOCX and PDF.
- `checksums_sha256.txt`: package file checksums.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/run_full_reproduction.py
```

Successful output ends with:

```text
FULL_REPRODUCTION: PASS
FULL_MODE_REPRODUCTION: PASS
```

For a faster environment check that reuses the packaged
`expected/stage8_curves/` intermediates:

```bash
python scripts/run_full_reproduction.py --mode quick
```

The official reproducibility claim is the full mode:

```bash
python scripts/run_full_reproduction.py --mode full
```

Full mode recomputes the Stage 8 bootstrap curves before rerunning the final closure,
legacy supporting analyses, figures, and verifier.

## Scope

Fully regenerated:

- Stage 8 bootstrap curves and curve summaries for CIFAR-10H, ChaosNLI, Snapshot Serengeti
- final saturation summaries, utility summaries, bootstrap ratio intervals, model fit comparison,
  budget sensitivity, dataset status report, figure data, and manuscript table CSVs
- Bingol, Snow, and Nitti legacy supporting JSON outputs
- five manuscript-facing figure PNGs generated from the regenerated CSV/JSON outputs

## Third-Party Data and Standardized Inputs

This archive does not redistribute full third-party raw datasets. It includes
standardized canonical inputs and compact legacy numeric inputs that are sufficient
to reproduce the manuscript-facing outputs exactly.

- `data/processed/` contains standardized item-level inputs derived from the admitted
  datasets for reproducibility of the Paper B analyses. Files ending in `.csv.gz`
  are gzip-compressed CSV files read directly by the reproduction scripts.
- `data/legacy_components/bingol_tables.json` contains the numeric table inputs used
  by the Bingol supporting reanalysis.
- `data/legacy_components/snow_digitized_curves.json` contains the digitized curve
  inputs used by the Snow supporting reanalysis.
- `data/legacy_components/nitti_data.xlsx` is the bundled Nitti legacy input workbook
  used by the boundary-condition reanalysis.

Original datasets, articles, and source materials remain governed by their original
licenses and terms. This package does not grant new rights to those third-party
materials; it provides the standardized analysis inputs needed for exact reproduction
of the submitted results. Users who need the complete raw sources should obtain them
from the original providers under the original terms.

## Determinism

The primary bootstrap seed is fixed at `20260709` with `B=200`.
Legacy scripts have fixed metadata timestamps, package-relative output paths,
and explicit external input files under `data/legacy_components/`.
The pinned package versions in `requirements.txt` are the versions used for the
local full-mode verification. Small cross-platform floating-point and PNG byte
differences are handled by the verifier contract above.

## Citation Metadata

Author and corresponding author: BongKeun Song.
Affiliation: Friedrich-Alexander-Universitaet Erlangen-Nuernberg (FAU), Erlangen, Germany.
Contact: bongkeun.song@fau.de.
