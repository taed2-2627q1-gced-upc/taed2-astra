# Dataset Card

> Kept in sync with `params.yaml` (`dataset` section) and the Great Expectations
> suite in `src/taed2_astra/data/validate.py`. Numbers below were computed from
> the DVC-tracked snapshot named in **Version**.

## Overview

- **Name:** Prediction of Sepsis (PhysioNet/Computing in Cardiology Challenge 2019, flattened)
- **Source / link:** Kaggle re-distribution of the PhysioNet 2019 Challenge training data
  (<https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis>), which mirrors
  <https://physionet.org/content/challenge-2019/1.0.0/>. The archive bundles the original
  per-patient `.psv` files (`training_setA/`, `training_setB/`), a single flattened
  `Dataset.csv` (the file we read), the challenge manuscript and the license.
- **License and terms of use:** Open Data Commons Open Database License (ODbL) — shipped
  as `LICENSE.txt` inside the archive. Attribution is required and derived databases
  must be shared under the same terms. The underlying PhysioNet data are de-identified.
- **Version / snapshot date:** archive `prediction-of-sepsis.zip`, 78.3 MB,
  MD5 `c0bc7d8788c394d48e16d36fccffb7b7` (from `data/raw/prediction-of-sepsis.zip.dvc`),
  pulled September 2026. Any later snapshot must be re-added with `dvc add`.
- **DVC path:** `data/raw/prediction-of-sepsis.zip` (raw, immutable) →
  `data/processed/train.parquet`, `data/processed/test.parquet` (outputs of the `prepare` stage).

## Composition

- **Rows / columns:** 1,552,210 rows × 43 columns in `Dataset.csv`. One row is one hour of
  one ICU stay. 40,336 distinct patients; median stay in the data 38 hours, maximum 336.
- **Unit of analysis vs. unit of prediction:** the model scores a single hour-row; the
  natural grouping unit is the patient (`Patient_ID`). This matters for splitting (below).
- **Feature columns (41):**
  - *Vital signs (8):* `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2`
  - *Laboratory values (26):* `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`,
    `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`,
    `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`,
    `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets`
  - *Demographics and context (7):* `Age`, `Gender` (0 = female, 1 = male), `Unit1` (MICU),
    `Unit2` (SICU), `HospAdmTime` (hours between hospital and ICU admission, negative),
    `ICULOS` (ICU length of stay in hours), `Hour` (row index within the stay)
  - All features are numeric (38 float, 3 integer). No free text.
- **Dropped at load time:** `Unnamed: 0` (a pandas row index leaked by the export; listed
  under `dataset.drop`). `Patient_ID` is kept for splitting but is never a feature.
- **Target column:** `SepsisLabel` ∈ {0, 1}. Following the challenge definition, the label is
  1 for every hour from six hours *before* clinical onset of sepsis onwards (t ≥ t_sepsis − 6),
  so the task is early warning, not diagnosis.
- **Class balance:** 1.80 % of rows are positive (27,916 of 1,552,210); 7.27 % of patients
  (2,932 of 40,336) develop sepsis at some point. The imbalance is why PR-AUC, not accuracy,
  is the primary metric in `metrics/benchmark.json`.
- **Missingness:** 68.4 % of feature cells are empty overall. Demographics and `ICULOS`,
  `Hour`, `HospAdmTime` are complete; vitals are 10–13 % missing (`HR`, `MAP`, `O2Sat`);
  most labs are ≥ 95 % missing (`Bilirubin_direct` 99.8 %, `Fibrinogen` 99.3 %,
  `TroponinI` 99.0 %). Missingness is informative — a lab is ordered when a clinician is
  concerned — so the preprocessor keeps a missing indicator per column rather than
  discarding that signal.
- **Demographics:** age 14–100, median 64; 44.1 % female.

## Collection

- **How was it collected:** retrospective extraction of routinely recorded ICU
  electronic health records, resampled to hourly bins, by the PhysioNet 2019 Challenge
  organisers (Reyna et al., *Critical Care Medicine*, 2020 — manuscript included in the
  archive). Labels were derived algorithmically from the Sepsis-3 criteria
  (suspicion of infection + SOFA increase), not hand-annotated.
- **Population covered:** adult ICU patients from two US hospital systems
  (`training_setA` and `training_setB` in the archive; the flattened CSV merges them
  without keeping the hospital identifier).
- **Known gaps or exclusions:** the challenge's third, hidden hospital (test set C) is not
  included, so external-hospital generalisation cannot be measured here. Patients under 18
  are rare (minimum age 14). Only ICU hours are present: nothing from the ward or the ED.
  Hospital of origin is not recoverable from `Dataset.csv`, so hospital-level bias cannot
  be audited from the processed data alone.

## Preprocessing

- **Cleaning steps** (`src/taed2_astra/data/make_dataset.py`): read `Dataset.csv` directly
  from the zip; drop `Unnamed: 0`. No imputation, scaling or filtering happens here — those
  live inside the model pipeline (`build_preprocessor`) so serving cannot skip them.
- **Splits (train/test) and how they were made:** 80 / 20 with `GroupShuffleSplit` on
  `Patient_ID`, `random_state = 42` (`params.yaml: prepare`). Result: 1,241,213 train rows
  (32,268 patients, 1.83 % positive) and 310,997 test rows (8,068 patients, 1.69 % positive),
  **zero patients in common**. Splitting by row instead would leak each patient's own
  trajectory into the test set and inflate every metric. Within training, model selection
  uses `StratifiedGroupKFold` on the same column (`benchmark` stage), for the same reason.
- **Validation** (`src/taed2_astra/data/validate.py`, rules in `params.yaml: validation`,
  output `reports/data_validation.json`): Great Expectations runs the same suite on **both**
  splits.
  - *Critical* (the pipeline stops): at least 10,000 rows; exactly the expected 43 columns;
    every column `int64`/`float64`; `SepsisLabel` non-null, in `{0, 1}` and with a mean
    between 0.5 % and 5 %; `Patient_ID`, `Hour`, `Age`, `Gender`, `ICULOS` non-null;
    `Gender`, `Unit1`, `Unit2` in `{0, 1}`; **zero patients shared between train and test**.
  - *Warning* (recorded for review): each of the 41 features within a physiologically
    plausible range for at least 99.9 % of its non-null values.
  - **What it caught:** `FiO2` (a fraction, 0.21–1.0) is out of range in 0.30 % of its train
    values: 291 below 0.21 (mostly `0.0`) and 24 above 1 (`2.0`, `10`, `4000`, i.e.
    percentages typed in the wrong unit). It is a warning, not a stop, because the affected
    values are 0.03 % of rows; cleaning them is a candidate preprocessing step.
    `HospAdmTime` has 8 nulls in train, so it is deliberately not in the "complete" list.

## Ethical considerations

- **Personal or sensitive data:** clinical measurements of real ICU patients, de-identified
  at source (no names, dates shifted, ages capped at 100). Still health data: keep it in
  the DVC remote, never in Git, and do not attempt re-identification.
- **Known biases:** two US hospital systems only; label defined algorithmically, so
  label noise follows the Sepsis-3 heuristics rather than clinical adjudication; sicker
  patients are measured more often, so both the presence and the frequency of values
  correlate with the outcome; the model can learn that clinicians' behaviour predicts
  sepsis, which will not transfer to a hospital with different ordering practices.
- **Appropriate uses:** teaching and evaluating MLOps practices (this course); research
  on early-warning models under the ODbL terms; benchmarking tabular classifiers on
  imbalanced, heavily missing clinical data.
- **Out-of-scope uses:** any clinical decision-making, triage or patient-facing deployment.
  The model built on this data is a course artefact, not a validated medical device, and has
  not been evaluated on any population other than the one described here.
