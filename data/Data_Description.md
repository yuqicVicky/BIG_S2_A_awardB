# Data Description — STAI-X Challenge 2026

## Folder structure

```
data/
├── train/
│   ├── dose_sys_train.csv       # training target
│   ├── covariates.csv           # training covariates
│   └── images/mat_density/      # MAT density PNG sidecar
│       └── {ST}_{PERIOD_ID}.png
├── val/
│   ├── covariates.csv           # validation covariates (target hidden)
│   └── images/mat_density/      # MAT density PNG sidecar (val periods)
│       └── {ST}_{PERIOD_ID}.png
├── sample_submission.csv        # submission template — fill this and write to ../submission.csv
└── Data_Description.md          # this file
```

The **training** package (`train/`) contains the labeled data you train on.
The **validation** package (`val/`) contains covariates only — the target is hidden.
Predict `rate_per_10000_ed_visits` for every row in `sample_submission.csv`.

---

## Files

| File | Location | Description |
| --- | --- | --- |
| `dose_sys_train.csv` | `train/` | Per-(period_id, jurisdiction) nonfatal overdose ED rates. Training target. |
| `covariates.csv` | `train/` and `val/` | Per-(period_id, jurisdiction) covariates panel. Same column schema in both. |
| `sample_submission.csv` | `data/` (root of this folder) | Submission template: 918 rows = 6 validation period_ids × 51 jurisdictions × 3 categories. Fill `rate_per_10000_ed_visits` and write the result to `../submission.csv` (the repo root). |
| `images/mat_density/{ST}_{PERIOD_ID}.png` | `train/images/` and `val/images/` | Per-(jurisdiction, period_id) 256×256 PNG visualizing within-state MAT clinic density. Optional multimodal feature. |

---

## Period/time mapping

The `period_id` field is an opaque identifier for a monthly period. Each
identifier corresponds to the month-end date shown below.

| Period end date | `period_id` |
| --- | --- |
| `2019-01-31` | `uTjgI1Sv` |
| `2019-02-28` | `wf016pk5` |
| `2019-03-31` | `BkTW58Ff` |
| `2019-04-30` | `shDD7wDP` |
| `2019-05-31` | `aZFXT65l` |
| `2019-06-30` | `fizSTkFs` |
| `2019-07-31` | `wQLd1SNL` |
| `2019-08-31` | `kmxcVN2e` |
| `2019-09-30` | `x24Jbzaz` |
| `2019-10-31` | `FuLb1kk4` |
| `2019-11-30` | `1Tl9271R` |
| `2019-12-31` | `Fj7ebbrB` |
| `2020-01-31` | `h9Re4kM3` |
| `2020-02-29` | `gqVDbZc7` |
| `2020-03-31` | `UKjQnuej` |
| `2020-04-30` | `TedmliP4` |
| `2020-05-31` | `PJu8Wb2C` |
| `2020-06-30` | `mZcpe0Ud` |
| `2020-07-31` | `NOtvYKB9` |
| `2020-08-31` | `YBlSTfgc` |
| `2020-09-30` | `rX3aMRGn` |
| `2020-10-31` | `44RA6kMl` |
| `2020-11-30` | `88NtGYTF` |
| `2020-12-31` | `tVb8fHGc` |
| `2021-01-31` | `aHT3VIho` |
| `2021-02-28` | `a239r7U4` |
| `2021-03-31` | `kTaI18at` |
| `2021-04-30` | `FZxIVFvr` |
| `2021-05-31` | `27DK2m8F` |
| `2021-06-30` | `y5ysDDpd` |
| `2021-07-31` | `NWU8bRHI` |
| `2021-08-31` | `3JILuYCd` |
| `2021-09-30` | `56aULHvm` |
| `2021-10-31` | `KDy1VIvO` |
| `2021-11-30` | `j1dZWmlF` |
| `2021-12-31` | `OugqP9RF` |
| `2022-01-31` | `4VCAqmuO` |
| `2022-02-28` | `nICRHvl9` |
| `2022-03-31` | `omhpgEVm` |
| `2022-04-30` | `WB9kCj4E` |
| `2022-05-31` | `iIi2mgES` |
| `2022-06-30` | `CDQGTxV0` |
| `2022-07-31` | `ePA08XXo` |
| `2022-08-31` | `MZ0ENeKD` |
| `2022-09-30` | `4MVfmuye` |
| `2022-10-31` | `N81HwK1a` |
| `2022-11-30` | `QpWgWZqu` |
| `2022-12-31` | `68B5zQl0` |
| `2023-01-31` | `BhtGJhRU` |
| `2023-02-28` | `9Dp3l3qq` |
| `2023-03-31` | `LALpfR23` |
| `2023-04-30` | `wa7tAVQg` |
| `2023-05-31` | `eVeAG5UX` |
| `2023-06-30` | `0Un18Xny` |
| `2023-07-31` | `9FQthr9A` |
| `2023-08-31` | `xtjIUpyk` |
| `2023-09-30` | `yIHgtqjY` |
| `2023-10-31` | `DpR0556d` |
| `2023-11-30` | `lSdEh765` |
| `2023-12-31` | `7cCeqHbf` |
| `2024-01-31` | `OqDkgaDk` |
| `2024-02-29` | `k4mmkR0U` |
| `2024-03-31` | `63zxcdKZ` |
| `2024-04-30` | `S2Qn2n8u` |
| `2024-05-31` | `i9aSkhZb` |
| `2024-06-30` | `UJgFAh3i` |
| `2024-07-31` | `OIpwoBOI` |
| `2024-08-31` | `lfTz14iT` |
| `2024-09-30` | `3CdEQbdr` |
| `2024-10-31` | `Kk6iVNym` |
| `2024-11-30` | `tLoy7Zpr` |
| `2024-12-31` | `S1xSdqr5` |
| `2025-01-31` | `Hy8SBtar` |
| `2025-02-28` | `rle4IZEn` |
| `2025-03-31` | `5Lptd03a` |
| `2025-04-30` | `jtUOZLP4` |
| `2025-05-31` | `dp3VfN8B` |
| `2025-06-30` | `dsZhPyK4` |
| `2025-07-31` | `aL5zkp6g` |
| `2025-08-31` | `yFh3wzPe` |
| `2025-09-30` | `lXSJn8AD` |
| `2025-10-31` | `kbpS9xmS` |
| `2025-11-30` | `DmKNJoJt` |
| `2025-12-31` | `if5b8Sut` |

---

## Columns

### `train/dose_sys_train.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `period_id` | str | Opaque 8-character period identifier. Joins to `covariates.csv` and `mat_density` image filenames. See the period/time mapping above for the corresponding month-end date. |
| `jurisdiction` | str | USPS two-letter state code, or `DC` (51 jurisdictions total) |
| `overdose_category` | str | Drug class. Scoring categories: `all_drugs`, `all_opioids`, `all_stimulants`. Categories are nested and non-exclusive — never sum across them. |
| `rate_per_10000_ed_visits` | float | Suspected nonfatal overdose ED visits per 10,000 total ED visits. `NaN` when suppressed. |

### `train/covariates.csv` and `val/covariates.csv`

Long panel keyed on `(period_id, jurisdiction)`. Identical column schema in both files.

| Column | Type | Meaning |
| --- | --- | --- |
| `period_id` | str | Joins to `dose_sys_train.csv` and `sample_submission.csv`; see the period/time mapping above for the corresponding month-end date. |
| `jurisdiction` | str | USPS two-letter state code or `DC` |
| `unemployment_rate` | float | State unemployment rate, percent |
| `labor_force` | int | Total state labor force, persons |
| `temp_avg_f` | float | State average temperature for the period, °F |
| `precip_in` | float | State total precipitation for the period, inches |
| `gtrends_overdose` | float | Google Trends search interest for "overdose" (0–100) |
| `gtrends_fentanyl` | float | Google Trends search interest for "fentanyl" (0–100) |
| `gtrends_naloxone` | float | Google Trends search interest for "naloxone" (0–100) |
| `gtrends_opioid` | float | Google Trends search interest for "opioid" (0–100) |
| `gtrends_methamphetamine` | float | Google Trends search interest for "methamphetamine" (0–100) |
| `state_doh_release` | str | Raw concatenated state Department of Health press-release text for that (state, period_id), drug-related only, capped at 500 whitespace-delimited tokens. Empty when no qualifying releases were issued. Calendar references have been stripped. |

Suppressed or missing numeric values appear as `NaN`; missing text is the empty string.

### `sample_submission.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `row_id` | int | Stable identifier. Your output must include every `row_id` from this file — no more, no fewer. |
| `period_id` | str | Opaque period identifier for the validation row; see the period/time mapping above for the corresponding month-end date. |
| `jurisdiction` | str | USPS state code or `DC` |
| `overdose_category` | str | One of `all_drugs`, `all_opioids`, `all_stimulants` |
| `rate_per_10000_ed_visits` | float | Replace with your prediction. Template ships with `0.0` placeholders. |

### Image sidecar (`images/mat_density/`)

Filenames: `{JURISDICTION}_{PERIOD_ID}.png` (e.g., `WV_uTjgI1Sv.png`).
Each image is a 256×256 PNG of the state polygon with a viridis heatmap of
MAT prescriber density. Bright yellow = high density; dark purple = sparse.
Feed to a vision encoder for multimodal features, or ignore and use the CSV
columns alone.
