---
co2_eq_emissions:
  emissions: 0.0901
  power_consumption: 0.000518
  source: CodeCarbon 3.3.1 (EmissionsTracker), reports/emissions/emissions.csv
  training_type: pre-training
  geographical_location: catalonia, Spain
  hardware_used: 12 x 13th Gen Intel(R) Core(TM) i7-1355U (CPU)
  training_time: 107.9
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

CodeCarbon on a single laptop CPU (Intel i7-9750H, Spain grid), `reports/emissions/emissions.csv`:

- **Shipped model training** (`train-hist_gradient_boosting`): 109 s, 0.0013 kWh,
  **0.23 g CO2eq** (`emissions_kg_co2 = 0.000226` in the MLflow run).
- **Full benchmark** (4 candidates × 3 folds): ~25 min compute, 0.0173 kWh, **3.0 g CO2eq**
  in total — LR 0.15 g, HGB 0.48 g, soft voting 0.57 g, stacking 1.81 g. Stacking alone
  cost 60 % of the benchmark's emissions for a model that was not adopted; that is part of
  why the ensemble decision above weighs cost, not just score.
- **Inference** (`inference-<model>` rows): scoring the whole test split, reported in
  `metrics/metrics.json` as `inference_duration_s`, `inference_emissions_kg_co2` and `inference_energy_kwh`.
- **Report:** `reports/emissions/emissions.csv` (CodeCarbon's log: one row per tracked run with duration,
  emissions, energy, location and hardware; versioned in Git), the same numbers per run in MLflow, and the
  Hugging Face `co2_eq_emissions` metadata at the top of this file, regenerated by `dvc repro co2_report`.
