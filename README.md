# Olympic Medal Predictor

Predicting whether a Summer-Olympics athlete wins a medal (gold/silver/bronze) from
their physical attributes, country economics and historical performance — built as a
small but **methodologically honest** ML project. The emphasis is on evaluation rigour
over a flashy headline number: time-respecting cross-validation, metrics that survive a
10.8%-positive class, calibrated probabilities, and a quant-style "treat predictions as
bets" framing.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # installs the package + dev tools
pytest                         # run the test suite
# Then open the notebooks (Run All) to reproduce the figures:
#   notebook_1_setup.ipynb        – load data, parse gender
#   notebook_2_eda_features.ipynb – gender-aware features, leakage-safe rates
#   notebook_3_results.ipynb      – walk-forward CV, PR/calibration/bootstrap, betting edge
```

All modelling logic lives in `src/medal_predictor/`; the notebooks are thin and just
call it and plot.

## The problem

The data is a SQLite database I built for a university course from the
[KeithGalli Olympics dataset](https://github.com/KeithGalli/Olympics-Dataset), covering
the 2008, 2012, 2016 and 2020 Summer Games. After cleaning: **17,118 athlete-event rows**
across **176 countries** and **48 sports**. Only **10.8%** of rows are medals, so the
class is heavily imbalanced — which drives every modelling decision below.

## Method

| Stage | Module | What it does |
|-------|--------|--------------|
| Data | `data.py` | SQL join over the DB; parses gender from event names, drops mixed/youth events |
| Features | `features.py` | **Gender-stratified** height/weight z-scores, BMI, GDP-per-capita, real age, gender one-hot. Synthetic columns (HRV, VO2max, blood-oxygen, etc.) are **dropped** |
| Leakage | `leakage.py` | Country/sport medal rates computed from **earlier Olympics only** — a 2016 athlete never sees 2020 |
| CV | `cv.py` | `WalkForwardOlympicsSplit`: train years strictly precede the test year (2008→2012→2016→2020) |
| Model | `model.py` | `StandardScaler` + `GradientBoostingClassifier`; hyper-parameter search scored on **PR-AUC**, not accuracy |
| Evaluation | `evaluation.py` | PR-AUC / max-F1 threshold, calibration + Brier score, bootstrap 95% CIs |
| Betting | `betting.py` | Implied odds, expected value vs the base-rate "market", Kelly sizing, PnL simulation |

Three design choices an interviewer can probe:

- **Gender stratification.** The average male athlete is ~12 cm taller and ~16 kg heavier
  than the average female. Treating them as one population makes physical features
  meaningless, so height/weight are standardised *within gender* (54% of events parsed as
  men, 43% as women, the rest dropped).
- **No target leakage.** The original notebooks computed country/sport medal rates over the
  *whole* dataset, so the feature for a 2016 athlete already baked in 2020 results.
  `leakage.py` fixes this; it's the main reason the honest score is lower than the old one.
- **No random k-fold.** Random folds let the model train on future Olympics to predict past
  ones. Walk-forward CV mirrors how the model would actually be used.

## Results

> Numbers below are from the 2020 walk-forward hold-out — paste them in after a
> `Run All` of `notebook_3_results.ipynb`.

| Metric | Walk-forward (honest) | Old random 5-fold (leaky) |
|--------|-----------------------|---------------------------|
| AUC-ROC | _fill_ | 0.751 |
| PR-AUC / Avg precision | _fill_ (95% CI [_lo_, _hi_]) | — |
| Max-F1 (and threshold) | _fill_ | — |
| Brier score | _fill_ | — |

The story is deliberately *not* "accuracy went up". The old pipeline reported 89.6%
accuracy and AUC 0.751 — both inflated by leakage and the easy true-negatives. Walk-forward
evaluation on a 10.8% class gives a lower but **trustworthy** number, reported with a
confidence interval and a calibration check. The betting section then asks the only
question that matters for a probability model: *does the edge over the base rate translate
into positive expected value?*

## Limitations / next steps

- Medal rates are by **sport**, not **event** — rowing/swimming hand out many more medals,
  which biases those sports upward. Event-level rates would be the next feature.
- Only four Olympics, so walk-forward has just three folds; CIs are correspondingly wide.
- The "market" line is a flat base rate, not real bookmaker odds — the betting analysis is
  illustrative of the framework, not a live edge.

## Repo layout

```
src/medal_predictor/   # data, features, leakage, cv, model, evaluation, betting
tests/                 # pytest suite for every module
notebook_1..3          # thin notebooks that call src/ and plot
pyproject.toml         # pinned deps + ruff/mypy/pytest config
.github/workflows/     # CI: ruff + mypy + pytest on every push
```
