# Third-party data: sources, licenses, and transformations

The standardized inputs under `data/` are derived from the datasets below. They are not
relicensed by this package (see `LICENSE.txt`); the terms listed here govern their use.

| Path | Source | License | Transformation applied here |
|---|---|---|---|
| `data/processed/CIFAR-10H/` | Peterson et al., ICCV 2019, human labels on the CIFAR-10 test set | CC BY-NC-SA 4.0 | per-label long form (`item_id`, `label`, `rater_id`); gold labels are the independent CIFAR-10 benchmark labels |
| `data/processed/ChaosNLI/` | Nie, Zhou and Bansal, EMNLP 2020 | CC-NC 4.0 | per-rater rows reconstructed from the released per-item label counts; covers the SNLI, MNLI and abductive-NLI subsets, so the label set has five values |
| `data/processed/Snapshot_Serengeti/` | Swanson et al., Sci. Data 2015, expert-verified subset | Community Data License Agreement, permissive variant | per-label long form; gold labels are the expert classifications |
| `data/legacy_components/` | numeric tables and digitized curves from the publications cited in the manuscript | terms of the respective publications | transcription and digitization only; no reanalysis of raw source data |

No complete raw third-party dataset is redistributed. Each transformation is implemented in
the standardization step documented in `README.md`; the resulting files are checksummed in
`checksums_sha256.txt`.
