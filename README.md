# Paper B Full Reproduction Package v1.2.0

This package is the manuscript-facing reproducibility release for Paper B.
It regenerates the packaged Stage 8 curve-bootstrap intermediates, the
fitted-saturation closure tables, the utility summaries, the legacy supporting
JSON outputs, the replicate-level fitted constants of the appendix, and the
manuscript-facing figure PNGs from the bundled canonical inputs.

Cite all versions of this package with the concept DOI
`10.5281/zenodo.CONCEPT_DOI_TO_BE_INSERTED`, which always resolves to the latest
version. The accompanying manuscript cites the version DOI of the release it was
submitted with, so that the printed numbers and the archived code stay pinned to
each other.

## Changes in v1.2.0

v1.2.0 adds the material the revised manuscript relies on. No previously reported
number changes.

- `scripts/refit_replicate_constants.py` (new). Refits the Michaelis-Menten family
  to each of the B = 200 bootstrap replicate curves and writes the half-saturation
  constant K with its percentile interval, the derived reference N95 = 20 + 19K,
  the budget-free threshold eta_c = 1 / (380 (K + 1)), the median residual RMSE,
  and the fraction of replicates in which the offset c0 reaches its lower bound.
  These are the quantities in the appendix table of the manuscript, which earlier
  versions of this package did not produce.
- `scripts/make_figures.py` (rewritten). The figures are now authored at the width
  at which they are printed (3.30 in) instead of being authored at 9.6 in and
  reduced by the word processor, so the stated font sizes are the printed font
  sizes. Rendering is at 600 dpi and the palette is colour-blind safe.
- `fig6_eta_collapse.png` (new). The cost-price collapse figure of the revised
  manuscript. It supersedes `fig5_budget_sensitivity.png`, which is retained so that
  the v1.1.0 figure set can still be regenerated.
- The Snapshot Serengeti curve in `fig2_saturation_curves.png` is now drawn solid up
  to its estimation grid bound and dotted beyond it, because its N95 of 30 lies
  outside the grid the curve was estimated on.
- Legacy metadata strings are corrected: the Bingol source now cites the published
  IEEE TSMC article rather than its preprint, and the Snow and Nitti records no
  longer carry the v1.0.7 raw-performance utility string, which has not described
  their computation since v1.1.0.
- `manuscript/` is removed. The manuscript is distributed by the journal, and the
  copy carried in v1.1.0 predated the revision.
- `checksums_sha256.txt` and `MANIFEST.tsv` are regenerated from the shipped tree.
  The v1.1.0 lists contained entries for files that are not present in the archive.

### Replicate count

The manuscript reports a replicate-count check at B = 2000. It is reproduced with

```bash
python scripts/compute_stage8_curves.py data/processed/<dataset> --mode <mode> \
    --B 2000 --seed 20260709 --outdir reproduced/stage8_curves_B2000
python scripts/refit_replicate_constants.py --curves reproduced/stage8_curves_B2000
```

The B = 2000 intermediates are not shipped: they are roughly ten times the size of
the packaged B = 200 curves, and the reported intervals are unchanged by them. The
packaged expected outputs and the manuscript both use B = 200.

## Changes in v1.1.0

The supporting analyses were harmonized onto the single normalized utility rule
used by the primary closure (manuscript rule (1)):

    U(N) = lambda * S(N) - (1 - lambda) * N / N_budget
    S(N) = clip((C(N) - C(1)) / (C(N_ref) - C(1)), 0, 1.5)
    N_ref    = N_peak for retrograde curves, N95 for saturating curves
    N_budget = the upper end of the estimated grid for the item-level datasets,
               and the largest N reported by the source for the re-analyzed
               components

In v1.0.7 the three legacy supporting scripts used each source's own
raw-performance rule `U(N) = lambda*C(N) - (1-lambda)*N/N_max` with a
source-specific, partly extrapolated `N_max` (500 for Bingol; 50/20 for Snow).
v1.1.0 replaced that with rule (1), refit the Snow and Nitti curves with the
same three admissible saturating families as the primary closure, and confined
every budget and epsilon search to the observed N range. Effects:

- Bingol retrograde ratio N*/N_peak: 0.279-0.990  ->  0.104-0.846 (still 20/20 early stop),
  lambda grid {0.3,0.5,0.7,0.9} -> {0.25,0.5,0.75,0.9}, N_budget 500 -> 29.
- Snow epsilon-to-lambda bridge holds inside the observed range for 2 of 5 tasks
  (word-sense disambiguation lambda=0.446, word similarity lambda=0.823) rather
  than the 5 of 5 obtained by extrapolation in v1.0.7.
- Nitti: 46 of 63 sheets have an identifiable N95; of those, 31 produce a valid
  epsilon-to-lambda row with lambda in [0.2, 0.99].

The primary CIFAR-10H / ChaosNLI / Snapshot Serengeti closure (Stage 8 bootstrap,
fitted saturation, N95, rho95, budget sensitivity) is identical within tolerance
across v1.0.7, v1.1.0 and v1.2.0.

## Naming: `N_support` in the manuscript, `n_obs_max` in the outputs

The revised manuscript calls the upper end of the estimation grid `N_support` and
defines it as the largest N supported by at least 75% of items: the 25th percentile
of per-item label counts, capped at 50, and halved in distribution mode because each
item's labels are split into a reference half and a query half. The packaged code
and CSV columns still use the earlier names for this quantity.

| Manuscript | Package |
| --- | --- |
| `N_support` | `n_obs_max` (CSV column), `observed_max` (`n_budget_type` value) |
| grid bound rule | `scripts/compute_stage8_curves.py`, lines 42-44 |

The column names are deliberately not renamed, so that the verifier contract and any
script written against v1.0.7 or v1.1.0 keeps working. The values are unchanged:
50, 50 and 21 for CIFAR-10H, ChaosNLI and Snapshot Serengeti, and 25 for the
CIFAR-10H reference-distribution sensitivity run.

## Reproduction Contract

The archive itself is protected by byte-level SHA-256 checksums in
`checksums_sha256.txt`. `MANIFEST.tsv` is the canonical logical inventory and lists
the uncompressed identity of the standardized label tables, which are shipped
gzip-compressed.

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
- `checksums_sha256.txt`: byte-level checksums of the shipped files.
- `MANIFEST.tsv`: canonical logical inventory.

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

The appendix constants are regenerated separately, because they refit the packaged
replicate curves rather than the closure tables:

```bash
python scripts/refit_replicate_constants.py
```

## Scope

Fully regenerated:

- Stage 8 bootstrap curves and curve summaries for CIFAR-10H, ChaosNLI, Snapshot Serengeti
- final saturation summaries, utility summaries, bootstrap ratio intervals, model fit comparison,
  budget sensitivity, dataset status report, figure data, and manuscript table CSVs
- Bingol, Snow, and Nitti legacy supporting JSON outputs
- replicate-level fitted constants for the appendix table
- the manuscript-facing figure PNGs, generated from the regenerated CSV/JSON outputs

Not included:

- Galaxy Zoo, which is not part of the admitted closure
- raw third-party downloads
- the B = 2000 replicate intermediates, which are regenerated on demand as shown above

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

Author and corresponding author: Bong-Keun Song.
Affiliation: Friedrich-Alexander-Universitaet Erlangen-Nuernberg (FAU), Erlangen, Germany.
Contact: bongkeun.song@fau.de.
