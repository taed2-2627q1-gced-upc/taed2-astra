---
co2_eq_emissions:
  emissions: 0.178
  power_consumption: 0.001022
  source: CodeCarbon 3.3.1 (EmissionsTracker), reports/emissions/emissions.csv
  training_type: pre-training
  geographical_location: catalonia, Spain
  hardware_used: 12 x Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz (CPU), 1 x NVIDIA GeForce GTX 1650 (GPU)
  training_time: 93.2
  optimization_techniques: histogram-based gradient boosting (binned features); ensembles rejected because
    their extra energy bought no meaningful PR-AUC gain
model_info:
  model_file_size: 1110933
  datasets_size: 1241213
  performance_metrics:
  - metric: roc_auc
    value: 0.8089
  - metric: pr_auc
    value: 0.0921
  - metric: recall
    value: 0.6379
  - metric: precision
    value: 0.0593
  - metric: f1
    value: 0.1085
  - metric: brier
    value: 0.1356
---

# Model Card

> Numbers here match `metrics/metrics.json`, `metrics/benchmark.json` and the
> MLflow runs named below. Regenerate with `dvc repro`; if a number changes,
> this file must change with it.

## Model details

- **Owner:** Team Astra (TAED2 2026-27 Q1)
- **Version:** 0.1.0 — first trained model, September 2026. Shipped candidate:
  `hist_gradient_boosting` (`params.yaml: train.model`).
- **Model type:** scikit-learn `Pipeline` = preprocessing + `HistGradientBoostingClassifier`.
  - *Preprocessing* (`build_preprocessor`): every column is numeric, so each goes through
    median imputation **with a missing-value indicator** and standard scaling. 41 raw inputs
    become 78 model features (41 values + 37 indicators for columns that are ever empty).
    The indicators exist because a lab being ordered at all is itself predictive.
  - *Estimator:* histogram-based gradient-boosted trees, `learning_rate=0.1`, `max_iter=300`,
    `max_leaf_nodes=31`, `min_samples_leaf=100`, `class_weight="balanced"`, `random_state=42`.
    Early stopping (`auto`, 10 % validation fraction) never triggered — all 300 rounds were used.
  - *Artefact:* `models/model.pkl`, 1.11 MB, DVC-tracked; also logged to MLflow (cloudpickle).
  - *Registry:* `make promote` registers this file in the MLflow Model Registry as
    `taed2-astra-sepsis` (tagged with its DVC MD5) and points the `@champion` alias at it,
    only after the release gates pass.
- **Training date:** 2026-09-19
- **MLflow run IDs** (experiment `taed2-astra`, <https://dagshub.com/Santi-49/taed2-astra.mlflow>):
  - `train-hist_gradient_boosting`: `fa6bf90f20b243bc80768d0f56e6fb5f`
  - `evaluate-hist_gradient_boosting`: `a79775a4f7764d149effad1508984f93`
  - benchmark parent: `37b5c51070c64add8772351879a11e9d`
    (child for this candidate: `3fa9f6ffe14a48c4b1877f263e6c6db3`)

## Intended use

- **Primary use case:** hourly early-warning score for sepsis in ICU patients — given one
  hour of vitals, labs and demographics, output the probability that the patient is within
  the label window (from six hours before clinical onset onwards, per the PhysioNet 2019
  definition). Served over the FastAPI `/predict` endpoint.
- **Intended users:** the course team and reviewers, as a worked example of an MLOps
  pipeline (reproducible training, experiment tracking, tested API).
- **Out-of-scope uses:** any real clinical use. Not a medical device; not validated
  prospectively, not validated on any hospital outside the two in the training data, and
  precision at the default threshold is ~6 %, i.e. roughly 16 false alarms per true alert.

## Training data

See [dataset_card.md](dataset_card.md). Training split: 1,241,213 hourly rows from
32,268 patients (1.83 % positive), disjoint by patient from the test split.

## Model selection

Five candidates were declared in `params.yaml` and scored on **identical**
`StratifiedGroupKFold(3)` folds (grouped by patient) within the training split.
Ranked by PR-AUC, the honest metric at 1.8 % prevalence (mean ± std over folds):

| Candidate | PR-AUC | ROC-AUC | Brier | Fit time (s) | Latency (ms / 1k rows) | Size (MB) |
|---|---|---|---|---|---|---|
| `ensemble_soft` (HGB + LR, averaged probabilities) | 0.0927 ± 0.0023 | 0.792 | 0.144 | 87 | 33.0 | 1.11 |
| `ensemble_stacking` (HGB + LR → LR meta-learner) | 0.0912 ± 0.0021 | 0.788 | 0.206 | 287 | 23.8 | 1.11 |
| **`hist_gradient_boosting`** (shipped) | 0.0898 ± 0.0025 | 0.782 | 0.121 | 72 | 25.5 | 1.11 |
| `logistic_regression` | 0.0755 ± 0.0028 | 0.755 | 0.189 | 22 | 5.4 | 0.007 |
| `tabpfn` (Hugging Face `Prior-Labs/TabPFN-v2-clf`) | skipped | — | — | — | — | — |

- TabPFN was skipped because the optional `foundation` dependency group (torch) was not
  installed for this run; the run is recorded in MLflow with `status = skipped`.
  Enable with `uv sync --group foundation`.
- **Why not the ensemble?** The team's rule was: adopt an ensemble only if its PR-AUC gain
  exceeds the best single model's fold-to-fold std *and* latency stays ≤ 2×. Soft voting
  passed on CV, but barely (+0.0030 vs std 0.0025, 1.29× latency). Trained and evaluated on
  the held-out test split (MLflow runs `5957a3e5fe1746ce9ce5788f39b2a434`,
  `105bc38c8f6e47f597d819ee75ceafca`) the gain shrank to **+0.0003 PR-AUC** and
  **+0.001 ROC-AUC**, with a *worse* Brier score (0.154 vs 0.136). A second model to maintain,
  29 % more latency and poorer calibration for no measurable gain is not worth it. The
  switch remains one line away: `dvc exp run -S train.model=ensemble_soft`.

## Evaluation

- **Test set:** 310,997 hourly rows from 8,068 patients, 1.69 % positive, zero patient overlap
  with training (`GroupShuffleSplit`, `random_state=42`). Used exactly once, by the
  `evaluate` stage, after model selection was closed.
- **Metrics** (`metrics/metrics.json`):

  | Metric | Value | Reading |
  |---|---|---|
  | ROC-AUC | **0.809** | ranks a random positive hour above a random negative one 81 % of the time |
  | PR-AUC | **0.092** | 5.5× the no-skill baseline of 0.017 (the positive rate) |
  | Brier | 0.136 | probabilities are shifted upward by class weighting; recalibrate before using as risk estimates |
  | Balanced accuracy | 0.732 | |
  | Recall | 0.638 | catches 64 % of label-window hours |
  | Precision | 0.059 | ~1 in 17 alerts is a true positive |
  | F1 | 0.108 | |
  | Predicted positive rate | 0.182 | flags 18 % of all ICU hours at the default threshold |

- **Decision threshold:** 0.5 on the balanced-weight probability (`params.yaml: evaluate.threshold`).
  This is a placeholder operating point, not a tuned one. Because `class_weight="balanced"`
  inflates positive probabilities, 0.5 corresponds to a high-recall / low-precision regime.
  The threshold-free metrics (ROC-AUC, PR-AUC) do not depend on it.

## Fairness

The `fairness` stage audits the shipped model's decisions on the test split with
[AI Fairness 360](https://github.com/Trusted-AI/AIF360) and writes `metrics/fairness.json`.
Definitions live in `params.yaml: fairness`:

- **Favorable outcome = an alert.** For a septic patient, being flagged is what brings a
  clinician to the bedside, so the metrics ask who is *denied* an alert.
- **Groups:** `Gender` (reference: male) and `Age` cut at 65 (reference: under 65). Metrics
  are *other group minus reference*. Race, ethnicity and hospital are not in the data, so
  they cannot be audited.
- **Unit of analysis:** the ICU hour, the same as every other metric in this card.

| Attribute | Recall (other / reference) | Equal opportunity diff. | Average odds diff. | Disparate impact | Sepsis prevalence (other / reference) |
|---|---|---|---|---|---|
| `Gender` (female vs male) | 0.612 / 0.654 | −0.042 | −0.032 | 0.871 | 1.44 % / 1.89 % |
| `Age` (≥ 65 vs < 65) | 0.647 / 0.628 | +0.019 | +0.019 | 1.115 | 1.75 % / 1.63 % |

- **Reading:** septic women are flagged a little less often than septic men (61 % vs 65 % of
  label-window hours). Both gaps are within the release gates (|difference| ≤ 0.10).
- **Why disparate impact is not a gate:** sepsis is less common in women in this data, so an
  equal alert rate would mean *over*-alerting one group. Only error-rate parity (equal
  opportunity, average odds) is enforced.
- **Mitigation available, not shipped.** `fairness.mitigation.method: reweighing` trains with
  AIF360 Reweighing weights on `Gender`. It was compared with no mitigation on the benchmark's
  patient-grouped 3-fold CV over the training split (mean ± std over folds), so the test split
  played no part in the choice and the table above stays an unbiased final audit. Reweighing
  narrowed the `Gender` gaps (equal opportunity −0.016 ± 0.061 → −0.004 ± 0.045, average odds
  −0.019 ± 0.037 → −0.006 ± 0.027, disparate impact 0.859 → 0.938) at no real cost (recall
  0.556 → 0.551, PR-AUC 0.0898 → 0.0897). The unweighted model ships: its gaps already pass the
  gates, and the improvement is smaller than the fold-to-fold spread, the same rule that keeps
  ensembles out. Group-specific thresholds (post-processing) were rejected: they would make the
  served decision depend on the patient's sex.

## Limitations

- **Per-hour, memoryless.** Each row is scored on its own; the model sees no trend
  (e.g. rising lactate over six hours). Sequence features or lag columns are the obvious next
  gain and would be added in `build_preprocessor`, not the API.
- **Learns clinician behaviour.** The missing-value indicators are among the strongest
  signals: sicker patients get more labs. That signal will not transfer to a unit with different
  ordering habits, and it means the model partly predicts "someone is already worried".
- **Two hospitals, US, ICU only.** No evidence of generalisation elsewhere; the hidden
  challenge hospital is not in our data.
- **Undertrained at the margin.** Early stopping never fired in 300 rounds, so more
  iterations or a higher learning rate may still help; not explored because it belongs to
  a hyperparameter-search stage, not this milestone.
- **Stacking's internal CV is not patient-grouped** (scikit-learn limitation), so the
  `ensemble_stacking` row in the benchmark is slightly optimistic; the outer grouped folds
  still score it fairly.
- **Not calibrated, not threshold-tuned.** Probabilities are not clinical risks.

## Sustainability

CodeCarbon 3.3.1 on one laptop (12 × Intel i7-9750H CPU plus a GTX 1650 GPU, Catalonia grid),
for the `dvc repro` of 2026-10-06 logged in `reports/emissions/emissions.csv`. The same numbers
are in the front matter above and in each MLflow run. An earlier, interrupted
`logistic_regression` row at 15:21 that day is not counted.

| Run | Duration | Energy | Emissions |
|---|---|---|---|
| Benchmark: `logistic_regression` (3 folds) | 53 s | 0.57 Wh | 0.10 g CO2eq |
| Benchmark: `hist_gradient_boosting` (3 folds) | 200 s | 2.33 Wh | 0.40 g |
| Benchmark: `ensemble_soft` (3 folds) | 222 s | 2.57 Wh | 0.45 g |
| Benchmark: `ensemble_stacking` (3 folds) | 705 s | 8.12 Wh | 1.41 g |
| **Benchmark total** (4 candidates; TabPFN skipped) | 20 min | 13.58 Wh | **2.36 g** |
| **Shipped model training** (`train-hist_gradient_boosting`) | 93 s | 1.02 Wh | **0.18 g** |
| Inference on the test split (310,997 rows) | 6.2 s | 0.027 Wh | 0.005 g |

What the numbers changed in our decisions:

- **Model selection is where the energy goes.** The benchmark cost 13× the final training
  run. Stacking alone was 60 % of it, because its internal 3-fold CV refits both members
  inside every outer fold, and it already failed the ensemble rule on CV (+0.0014 PR-AUC,
  below the 0.0025 fold std).
  A future benchmark should drop candidates that already failed the selection rule, or run
  `benchmark.max_rows` on a sample first.
- **Soft voting was not the expensive option.** Its benchmark energy was within 10 % of
  gradient boosting alone (the logistic member is cheap), so it was rejected for latency,
  maintenance and calibration, not energy. The card says so rather than claiming a green win
  it did not earn.
- **Serving is negligible next to training.** About 0.09 mWh and 0.015 mg CO2eq per 1,000
  predictions, so retraining frequency, not request volume, dominates the footprint
  of this component.
- **Figures:** `reports/figures/energy_consumed_per_model.png` and
  `reports/figures/duration_vs_energy.png`, redrawn by `dvc repro plots`.
- **Caveat:** on a laptop CodeCarbon estimates CPU power rather than metering it, so the
  values are good for comparing candidates on the same machine, not as absolute measurements.
  The 2026-09-28 run of the same code on an i7-1355U logged about half the energy for
  training and most benchmark candidates (CodeCarbon estimated ~7 W of CPU power there versus
  ~30 W here). Stacking is the exception: it ran 2.2× longer there, so its energy was similar.
  The conclusions held (stacking dominates, soft voting costs about the same as gradient
  boosting), but the ratios moved (benchmark 19× training there, 13× here).
  The GPU is idle (scikit-learn runs on the CPU), but CodeCarbon still counts its ~2 W draw,
  about 5 % of each run's energy.

## Quality gates

A model only ships if `tests/test_model_quality.py` passes on the held-out test split.
Thresholds live in `params.yaml` (`model_quality`) and sit below the current scores so a real
regression fails, not noise:

| Gate | Threshold | Current |
|---|---|---|
| ROC-AUC | ≥ 0.78 | 0.809 |
| PR-AUC lift over the positive rate | ≥ 4.0× | 5.5× |
| Recall at `evaluate.threshold` | ≥ 0.55 | 0.638 |
| ROC-AUC gap between slices (`Gender`, `Unit1`) | ≤ 0.10 | 0.016 (Gender), 0.045 (Unit1) |
| \|Equal opportunity difference\| (`Gender`, `Age`) | ≤ 0.10 | 0.042 (Gender), 0.019 (Age) |
| \|Average odds difference\| (`Gender`, `Age`) | ≤ 0.10 | 0.032 (Gender), 0.019 (Age) |
| Mean risk rises when HR +40, Temp +2 °C, Resp +15, MAP −30, Lactate +4 | > 0 | all rise |
| Identifiers (`Patient_ID`) and the label are not model inputs | — | pass |

The Unit1 gap (0.82 vs 0.77 ROC-AUC between ICU types) is within the gate but is the largest
subgroup difference observed and should be watched if the model is retrained.
