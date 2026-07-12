# Paper B Full Reproduction Package v1.1.0

This package is the manuscript-facing reproducibility release for Paper B.
It is designed to regenerate the packaged Stage 8 curve-bootstrap intermediates,
fitted-saturation closure tables,
utility summaries, legacy supporting JSON outputs, and five figure PNGs from the
bundled canonical inputs.

## Changes in v1.1.0

The supporting analyses are now harmonized onto the single normalized utility
rule used by the primary closure (manuscript rule (1)):

    U(N) = lambda * S(N) - (1 - lambda) * N / N_budget
    S(N) = clip((C(N) - C(1)) / (C(N_ref) - C(1)), 0, 1.5)
    N_ref    = N_peak for retrograde curves, N95 for saturating curves
    N_budget = observed maximum N of that component

In v1.0.7 the three legacy supporting scripts used each source's own
raw-performance rule `U(N) = lambda*C(N) - (1-lambda)*N/N_max` with a
source-specific, partly extrapolated `N_max` (500 for Bingol; 50/20 for Snow).
v1.1.0 replaces that with rule (1), refits the Snow and Nitti curves with the
same three admissible saturating families as the primary closure, and confines
every budget and epsilon search to the observed N range. Effects:

- Bingol retrograde ratio N*/N_peak: 0.279-0.990  ->  0.104-0.846 (still 20/20 early stop),
  lambda grid {0.3,0.5,0.7,0.9} -> {0.25,0.5,0.75,0.9}, N_budget 500 -> 29.
- Snow epsilon-to-lambda bridge holds inside the observed range for 2 of 5 tasks
  (word-sense disambiguation lambda=0.446, word similarity lambda=0.823) rather
  than the 5 of 5 obtained by extrapolation in v1.0.7.
- Nitti: 46 of 63 sheets have an identifiable N95; of those, 31 produce a valid
  epsilon-to-lambda row with lambda in [0.2, 0.99].

The primary CIFAR-10H / ChaosNLI / Snapshot Serengeti closure (Stage 8 bootstrap,
fitted saturation, N95, rho95, budget sensitivity) is byte-for-tolerance identical
to v1.0.7; only the supporting-component numbers and `fig4_bingol_retrograde_ratios.png`
change.

## Reproduction Contract

The archive itself is protected by byte-level SHA-256 checksums in
`checksums_sha256.txt`.

Regenerated analysis outputs are checked by `scripts/verify_full_reproduction.py`
using a cross-platform reproducibility contract:

- CSV files: identical columns, identical row counts, categorical fields exact,
  numeric fields within `rtol=1e-6` and `atol=1e-10`.
- JSON files: semantic comparison with the same numeric tolerance.
- Nitti legacy JSON: manuscript-facing summary fields are verified
  (sheet counts, admissibility counts, N95/epsilon-to-lambda bridge counts,
  early-stop counts, and core sheet classifications). Nonselected fitted-model
  optimizer parameters are not claim-bearing and are excluded from failure
  decisions because they can vary slightly across numerical environments without
  changing any manuscript result.
- PNG files: generated and dimension-checked; scientific figure validation is based
  on the underlying `figure_data/` CSV files rather than platform-dependent PNG bytes.

This avoids false failures from Windows/Linux line endings, floating-point last-bit
differences, and Matplotlib raster metadata/rendering differences.

The contract starts from bundled canonical standardized inputs in `data/processed/`.
Large standardized label tables are stored as `.csv.gz` where needed so that no
single file exceeds common 25 MB upload limits; the reproduction scripts resolve
both `.csv` and `.csv.gz` inputs automatically.
It does not claim to redownload or relicense third-party raw source data. The raw
acquisition/standardization step is outside this public release; the standardized
canonical inputs are included so the manuscript-facing results can be reproduced
exactly.

## Contents

- `data/processed/`: canonical standardized item-level inputs for CIFAR-10H,
  ChaosNLI, and Snapshot Serengeti.
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

For a faster environment check that skips the expensive Stage 8 bootstrap recomputation
and reuses the packaged `expected/stage8_curves/` intermediates:

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

Not included:

- Galaxy Zoo, which is not part of the admitted closure
- raw third-party downloads

## Third-Party Data and Standardized Inputs

This archive does not redistribute full third-party raw datasets. It includes
standardized canonical inputs and compact legacy numeric inputs that are sufficient
to reproduce the manuscript-facing outputs exactly.

- `data/processed/` contains standardized item-level inputs derived from the admitted
  datasets for reproducibility of the Paper B analyses. Large `labels_long.csv`
  files may be gzip-compressed as `labels_long.csv.gz` without changing the
  canonical table contents.
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
