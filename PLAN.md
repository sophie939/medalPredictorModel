# Olympic Medal Predictor — Portfolio-Grade Overhaul Plan

> Created 2026-06-01. Goal: make this repo a defensible interview portfolio piece for
> Graduate Quant Developer roles in London, target ready by August 2026. ~3–4 week build.

## Context

The current repo (verified 2026-06-01) has three problems any quant interviewer will flag in minutes:

1. **No software-engineering signal.** Everything lives in 4 Jupyter notebooks. No package
   layout, no tests, no CI, no dependency pinning, no type hints.
2. **Evaluation is not rigorous.** 5-fold random CV across 2008–2020 data lets the model train
   on future Olympics to predict past ones. Reports only AUC-ROC and accuracy — both misleading
   on a 10.8% positive class. No PR-AUC, no calibration check, no confidence intervals, no
   threshold tuning. Recall of 9% is reported but never addressed.
3. **No quant-flavoured framing.** The model outputs probabilities but never treats them as bets.
   No expected-value analysis, no calibration of stated vs realised medal probabilities.

Ancillary cleanup: (a) drop synthetic features (HRV/VO2max/blood oxygen) that won't survive
interview scrutiny; (b) consolidate two parallel feature pipelines (16-feature non-gender vs
20-feature gender-aware) down to gender-aware only — the model currently trains on the wrong set.

Target: polished repo with a results writeup that reads like a junior quant research note.

## Target structure

```
medalPredictorModel/
├── pyproject.toml                # packaging + pinned deps + tool config
├── README.md                     # rewritten — results-focused
├── .github/workflows/ci.yml      # lint + type-check + tests on every push
├── .pre-commit-config.yaml       # ruff + mypy + nbstripout
├── .gitignore                    # ignore .pkl, .DS_Store, __pycache__
├── data/
│   ├── raw/olympic.db            # SQLite (move from root)
│   └── processed/                # gitignored; rebuilt by `make data`
├── src/medal_predictor/
│   ├── __init__.py
│   ├── data.py                   # DB query + gender parsing
│   ├── features.py               # gender-aware z-scores, BMI, GDP/cap
│   ├── leakage.py                # past-only country/sport medal rates
│   ├── cv.py                     # walk-forward splitter by Olympic year
│   ├── model.py                  # train, hyperparam search
│   ├── evaluation.py             # PR-AUC, calibration, bootstrap CIs
│   └── betting.py                # EV framework, threshold-as-bet
├── notebooks/
│   ├── 01_data_exploration.ipynb # thin — calls src/, shows charts
│   ├── 02_feature_analysis.ipynb # gender-aware EDA, correlations
│   ├── 03_model_training.ipynb   # walk-forward CV results
│   ├── 04_evaluation.ipynb       # PR/calibration/bootstrap
│   └── 05_betting_analysis.ipynb # EV framing — the differentiator
├── tests/
│   ├── test_data.py              # gender parsing, filter correctness
│   ├── test_features.py          # z-score is gender-stratified; BMI sane
│   ├── test_leakage.py           # past-only rates don't peek at future
│   ├── test_cv.py                # walk-forward splits, no year overlap
│   └── test_evaluation.py        # bootstrap CI shrinks with more samples
└── scripts/
    ├── build_features.py         # CLI: rebuild data/processed/
    └── train.py                  # CLI: train + save model + metrics
```

## Phased plan

### Phase 1 — Foundation (~3 days)
Goal: project skeleton in place; no behaviour change yet.

- Create `pyproject.toml` with pinned `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`,
  `joblib`, `pytest`, `ruff`, `mypy`.
- Move `olympic.db` to `data/raw/`; add `.gitignore` to exclude `*.pkl`, `data/processed/`,
  `.DS_Store`, `__pycache__/`.
- Create empty `src/medal_predictor/` package with `__init__.py` and module stubs.
- Create `tests/` directory with one passing smoke test (`test_import.py`).
- Set up `.pre-commit-config.yaml`: ruff format/check + nbstripout.
- Set up `.github/workflows/ci.yml` running `ruff check`, `mypy src/`, `pytest`.

**Critical files:** `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`,
`.pre-commit-config.yaml`, `src/medal_predictor/__init__.py`, `tests/test_import.py`.

### Phase 2 — Data + features migration (~4 days)
Goal: move all logic out of notebooks 1 & 2 into `src/`. Drop synthetic features. Gender-aware only.

- `src/medal_predictor/data.py`: `load_raw(db_path) -> pd.DataFrame` (existing SQL join) plus
  `parse_gender(df) -> pd.DataFrame` (port event-name parser from notebook 1, drop 483 YOG events).
- `src/medal_predictor/features.py`:
  - Drop `heartRateVariability`, `vo2Max`, `bloodOxygen`, `bodyFat`, `injurySeverityScore`,
    `fitness_score`, `risk_score`.
  - Keep gender-stratified `height_zscore`, `weight_zscore`, `bmi`, `gdp_per_capita`,
    `estimated_age`, gender one-hot.
  - Final feature count: ~12 (down from 20).
- `src/medal_predictor/leakage.py`: **new** — `compute_past_only_rates(df, year)` returns country
  and sport medal rates using *only* rows where `participation_year < year`. Kills the target leak.
- Delete `olympic_data_raw.pkl`, `olympic_data_with_gender.pkl`, `olympic_data_processed.pkl`,
  `features_X.pkl`, `feature_names.pkl` from git (rebuilt now).
- Tests: `test_data.py` (parse_gender drops YOG; 54%/43% split holds), `test_features.py`
  (z-score mean ≈ 0 within each gender), `test_leakage.py` (2016 rates can't see 2020 rows).

**Critical files:** `src/medal_predictor/data.py`, `features.py`, `leakage.py`, `tests/test_*.py`.

### Phase 3 — Honest evaluation (~5 days)
Goal: replace random 5-fold CV with walk-forward; report metrics that respect class imbalance.

- `src/medal_predictor/cv.py`: `WalkForwardOlympicsSplit` — train years strictly precede test year.
  Three splits: 2008→2012, 2008+2012→2016, 2008+2012+2016→2020.
- `src/medal_predictor/model.py`: `train(X, y, params)` returns sklearn Pipeline
  (StandardScaler + GradientBoostingClassifier). Hyperparam search uses `WalkForwardOlympicsSplit`.
- `src/medal_predictor/evaluation.py`:
  - `precision_recall_summary(y_true, y_score)` → PR-AUC, average precision, max F1 + its threshold.
  - `calibration_report(y_true, y_score, n_bins=10)` → reliability diagram data + Brier score.
  - `bootstrap_metric(y_true, y_score, metric_fn, n=1000)` → 95% CI via bootstrap resampling.
- Notebook 3: load features, run walk-forward CV, comparison table (old random-5-fold vs new
  walk-forward) to show the leakage was real.
- Notebook 4: PR curve, calibration plot, bootstrap CIs, threshold tuning chart.

**Critical files:** `cv.py`, `model.py`, `evaluation.py`, `notebooks/03_model_training.ipynb`,
`notebooks/04_evaluation.ipynb`.

### Phase 4 — Quant framing (~3 days)
Goal: the differentiator. Treat predictions as bets.

- `src/medal_predictor/betting.py`:
  - `implied_odds(p) -> decimal_odds` — model probability to fair odds.
  - `expected_value(p_model, p_market, stake=1)` — EV of taking a market line.
  - `kelly_fraction(p_model, decimal_odds)` — Kelly criterion bet sizing.
  - `simulate_betting(predictions, y_true, threshold)` — running PnL betting 1 unit on every
    athlete rated above the threshold; report Sharpe-style ratio.
- Notebook 5: synthetic "market odds" = baseline 10.8% medal rate; edge curve vs model-market gap.

**Critical files:** `betting.py`, `notebooks/05_betting_analysis.ipynb`.

### Phase 5 — Polish (~2 days)
Goal: the repo looks like a junior quant has owned it for a year.

- Rewrite `README.md`: problem → method → headline result with 95% CI → what was learned.
  ~400 words, no spelling errors. Hero image: PR curve + calibration plot side-by-side.
- Add `scripts/build_features.py` and `scripts/train.py` — whole pipeline reproduces in two CLI calls.
- Add `make` targets: `make data`, `make train`, `make test`.
- `pytest` green, `ruff check .` clean, `mypy src/` clean — all enforced by CI.
- Bump test coverage to ≥80% on `src/medal_predictor/`.

## Reusable code already in the repo

- Event-name gender parser logic in `notebook_1_setup.ipynb` → port to `data.py:parse_gender`.
- Gender-aware z-score logic in `notebook_2_eda_features.ipynb` → port to `features.py`.
- Country/sport medal rate logic in `notebook_2_eda_features.ipynb` → **rewrite** in `leakage.py`
  to fix the target-leakage bug.
- SQL join query in `notebook_1_setup.ipynb` → port to `data.py:load_raw`.

## Verification

- **After Phase 1:** `pre-commit run --all-files` passes; CI runs green on a dummy commit.
- **After Phase 2:** `pytest tests/test_data.py tests/test_features.py tests/test_leakage.py` green;
  `python scripts/build_features.py` regenerates `data/processed/features.parquet` from scratch.
- **After Phase 3:** Notebook 3 runs top-to-bottom; leakage-comparison table shows walk-forward AUC
  noticeably lower than random-5-fold (the proof). PR-AUC, max-F1 threshold, Brier with 95% CIs in nb 4.
- **After Phase 4:** Notebook 5 produces a PnL curve over the four Olympics; edge vs 10.8% baseline.
- **After Phase 5:** `make test` green; `make train` produces `models/medal_predictor.pkl` + metrics
  JSON; README headline result reproducible via the README "Quickstart" block.

## Key risks / decisions deferred

- **Walk-forward will probably hurt headline AUC** (~0.76 random-5-fold → maybe 0.65–0.70 honest).
  Lean into this — the comparison itself is the interview talking point.
- **PR-AUC will be low** (~0.25 with a 10.8% positive class may be the truth). Report it with a
  calibration plot and EV analysis rather than hiding behind 89% accuracy.
- **Dropping synthetic features may reduce headline accuracy.** Accepted — model honesty matters
  more than the headline number for this audience.
