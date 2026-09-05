# Reproduction Package v1.3.0 — Utility-Optimal Stopping Relative to a Saturation Reference in Bounded Collective-Judgment Systems

This package is the reproducibility release accompanying the manuscript "Utility-Optimal Stopping Relative to a Saturation Reference in Bounded Collective-Judgment Systems: A Multi-Domain Characterization" (IEEE Access, Access-2026-34846).
It regenerates the packaged Stage 8 curve-bootstrap intermediates, the
fitted-saturation closure tables, the utility summaries, the legacy supporting
JSON outputs, the replicate-level fitted constants of the appendix, and the
manuscript-facing figure PNGs from the bundled canonical inputs.

Version: v1.3.0

Author: Bongkeun Song
Affiliation: Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Germany
ORCID: 0009-0008-3120-8126

Archived release: https://doi.org/10.5281/zenodo.21320155 — the
concept DOI, which always resolves to the latest version. The version archived with
the manuscript is release tag **v1.3.0**.

## Changes in v1.3.0

v1.3.0 corrects two defects in the curve estimation and one in the utility definition. The
reported numbers change; earlier versions of this package should not be used.

- **The estimation grid starts at N = 1.** Earlier versions fitted from N = 2 upward while
  normalizing performance against C(1), so the baseline was extrapolated from the fit rather
  than observed. C(1) is now measured.
- **Ties in the majority vote are scored by their expected accuracy under a uniform draw**
  (1/m when the gold label is one of m tied labels), replacing a tie-break that took the
  first label in sort order. The old convention made the result depend on how labels are
  named: on Snapshot Serengeti it moved the accuracy at N = 2 from 0.812 to 0.745, away from
  the value C(1) that it must equal in expectation. `common_metrics.majority_score` replaces
  `majority_label` and is used by both the curve estimation and the standalone helper.
- **Performance is normalized by the fitted asymptotic gain, not by the gain at the
  saturation reference.** A reference read at fraction q carries q times the asymptotic gain,
  so normalizing by it rescaled the performance axis by 1/q and made the optimum depend on
  the choice of q. Under the new normalization S(N) = (N - 1)/(K + N) for the Michaelis-Menten
  family, the stopping count is invariant to the reference convention and the reference
  enters only the reported ratio. `revision_quantities.py` records that the stopping count is
  unchanged across the 90%, 95% and 99% conventions, and separately reports the sensitivity of
  K and of the ratio to the estimation grid, which is not an invariance (Appendix A-F).
- **The utility is written as a price and a capacity.** `utility_curve` takes
  eta = (1 - lambda)/(lambda * N_budget) as the marginal price and N_max as the feasible
  capacity, and records both with every stopping count, together with the unconstrained
  optimum and whether the capacity binds.
- Snapshot Serengeti is reported as a boundary case rather than supporting evidence: with the
  corrected estimation its reference lies 2.2 times beyond the estimation grid and its AIC
  margin over the runner-up falls to about 4.
- `LICENSE.txt` no longer places the standardized third-party inputs under CC BY 4.0; their
  upstream licenses are listed in the new `THIRD_PARTY_LICENSES.md`.

## Changes in v1.2.1

v1.2.1 corrects the reproduction runner and extends two scripts. No previously reported
number changes.

- `scripts/run_full_reproduction.py` now runs `refit_replicate_constants.py` and
  `revision_quantities.py` before `make_figures.py`, so `fig6_eta_collapse.png` finds the
  replicate constants it reads and the verifier checks both CSVs. In v1.2.0 a clean run
  stopped at figure generation with a missing-file error.
- `make_figures.py` (`fig6`): the optima are now searched over `1..N_budget`, the same
  semantics as `recompute_final_closure.py`, so the figure shows where the cost-price
  collapse holds and where the capacity cap binds, rather than an unconstrained search.
- `refit_replicate_constants.py` adds `eta_c_discrete`, the threshold under the closure
  code's normalization at the integer reference `M = ceil(N95)`, alongside the continuous
  approximation.
- `revision_quantities.py` adds per-replicate model selection (which family AIC selects on
  each of the B = 200 bootstrap curves, with the bootstrap distribution of the AIC margin)
  and a reference-fraction sensitivity table (N* at λ = 0.90 when the reference is read at
  90, 95 and 99 percent of the fitted gain).
- The evidence-tier strings in the closure status tables follow the manuscript taxonomy:
  `sensitivity_only`, `retrograde_benchmark`, and `LEGACY_REANALYSIS_AVAILABLE_SEPARATE_COMPONENT`.
  The earlier `sensitivity_supporting`, `retrograde_backbone` and `..._SEPARATE_SUPPORT` labels
  described a tier structure the manuscript no longer uses. No reported number changes.
- Console output of the legacy scripts no longer carries authoring notes. In particular
  `paperB_bingol.py` no longer prints an unconditional success line for the epsilon-to-lambda
  map: it now reports how many of the five conditions yield an equivalent weight inside
  [0.30, 0.95], which on the shipped results is none of them, and the figure summary reports
  the same count instead of a fixed check mark. The role descriptions in the legacy scripts
  ("anchor", "backbone", "core claim", "contribution claim") are replaced by neutral
  descriptions of what each script computes. No packaged result changes.
- All scripts now share one parameter bound for the three candidate families (level and
  amplitude at most 1.2, half-saturation at most 1000). The replicate, B = 2000 and derived
  scripts previously used 1e4 for the half-saturation bound, which moved the fitted constants
  in the fourth decimal place relative to the main fit.
- The residual screen in `recompute_final_closure.fit_family` (`residual_rmse < 0.1`) is part
  of the shape-and-parameter condition of Section V-B and is documented in the code.
- Representative weights are evaluated at their exact values under every budget definition,
  including the observed-max budget where a dense grid is also traced; a row labelled 0.25 is
  computed at 0.25.
- `final_saturation_summary.csv` carries `K_point`, the half-saturation constant of the selected
  point-estimate fit, and `fig6_eta_collapse.png` uses it for its curve and threshold so that
  markers, curve and threshold share one estimand.
- `revision_quantities.csv` adds the coefficients of variation, the rank correlation between N
  and the bootstrap mean, grid coverage and the mean increment above N = 20, so the figures the
  manuscript quotes in Section VI-F are regenerated rather than only stated.
- `dataset_status_report.csv` now records the four gate conditions of Section V-B, the tier they
  imply, and whether that tier matches the one declared for the dataset. The tier is derived from
  the gate rather than read from the specification.
- `fig6_eta_collapse.png` plots the rows of `final_utility_summary.csv` directly, so the figure and
  the tables show the same optima; it previously recomputed them on its own grid.
- The three candidate families now share one set of parameter bounds across every script
  (level and amplitude at most 1.2). The inverse-square-root amplitude was previously bounded at
  10 in the main fit and at 1.0 in the replicate scripts; the fitted amplitudes are far below
  either bound on these datasets, so no reported value changes.
- Equation (A.6) of the manuscript is the exact crossing for the integer optimizer and (A.7) is its
  continuous approximation; `eta_c` and `eta_c_continuous` correspond to them in that order.
- `scripts/serengeti_unscoreable.py` and `expected/diagnostics/serengeti_unscoreable.csv` are new.
  They regenerate the expert-label coverage figures quoted in Section IV-A: the count of events
  whose expert label appears among no volunteer label for that event, the fraction of the
  available set they represent at each end of the grid, and the shift in K when that fraction is
  removed. The runner produces this file and the verifier checks it.
- `eta_c_lo` and `eta_c_hi` in `replicate_constants.csv` are now evaluated under the same discrete
  threshold as `eta_c`, each at its own K and the reference that K implies. They previously used a
  scaling that belonged to the earlier reference-based normalization.
- `recompute_final_closure.py` no longer contains the authoring-time venue-selection helper or
  emits `final_gate_decision.md`; that file was an internal artefact and is not part of the
  reproduction. Its wording also carried an evidence-tier description that the manuscript no
  longer uses.
- `compute_stage8_curves.py` no longer writes `utility_optimum_primary_budget.csv`. That helper
  normalized performance by the observed span of C rather than by the fitted asymptotic gain, so
  its optima were not the quantities the manuscript reports; the reported utility is computed in
  `recompute_final_closure.utility_curve` alone. `common_metrics.utility_optimum` is retained and
  documented as an unused pilot helper.
- The `input_mode` field of the closure tables reads `RAW_ITEM_LEVEL_BOOTSTRAP`; the earlier
  label used "hierarchical", which overstates the resampling design (Section V-A of the paper).
- `scripts/verify_full_reproduction.py` now also checks `expected/b2000/`. Because the
  B = 2000 intermediates are not shipped, a standard run reports that summary as not
  regenerated rather than failing; `run_full_reproduction.py` regenerates it when a
  `reproduced/stage8_curves_B2000` directory is present.
- A new script, `scripts/b2000_validation.py`, and a shipped summary,
  `expected/b2000/b2000_validation_summary.csv`, record the replicate-count check reported in
  Section V-A of the manuscript: the median K and the 2.5–97.5 percentile interval of the
  stopping count at the four representative weights, at B = 200 (the shipped curves) and at
  B = 2000 (a rerun with the same seed). The B = 2000 intermediates are not shipped; the
  summary is checksummed, and the commands that regenerate it are given below.

## Changes in v1.2.0

v1.2.0 adds the material the revised manuscript relies on. No previously reported
number changes.

- `scripts/revision_quantities.py` (new). Regenerates the derived quantities that appear only in
  the manuscript prose: the runner-up family's asymptote, the effect of shortening the estimation
  grid on K and on the continuous stopping ratio, the shortfall of a fixed annotation redundancy
  against the achievable normalized gain, and the item coverage of the estimation grid. Earlier
  versions did not produce these, so a reader could not check them.
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
python scripts/b2000_validation.py --b2000 reproduced/stage8_curves_B2000
```

The B = 2000 intermediates are not shipped: they are roughly ten times the size of
the packaged B = 200 curves, and nine of the twelve reported intervals are unchanged by them and three widen by one contributor. The
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
python scripts/revision_quantities.py
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
  datasets for reproducibility of the analyses reported in the manuscript. Large `labels_long.csv`
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
