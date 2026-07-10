# Project summary — Olympic Medal Predictor

A one-page reference for *me*: what this project is, how it fits together, and how to talk
about it in an interview. (README is the public-facing version; this is the cheat-sheet.)

## In one sentence

A gradient-boosting model that predicts whether an Olympic athlete medals, rebuilt from a
pile of notebooks into a tested Python package — with the emphasis on **honest evaluation**
(no leakage, time-based CV, imbalance-aware metrics) and a **quant "predictions as bets"**
framing.

## The data

- Self-built SQLite DB (uni course), KeithGalli Olympics source, Summer Games 2008–2020.
- After cleaning: **17,118 athlete-event rows**, 176 countries, 48 sports.
- Target: `has_medal` (top-3 finish). Only **10.8%** positive → imbalanced, the central
  problem.

## Pipeline at a glance (`src/medal_predictor/`)

```
data.py      → load_raw (SQL join) + parse_gender (from event names, drop mixed/youth)
features.py  → gender-stratified z-scores, BMI, GDP/capita, real age, gender one-hot
leakage.py   → country/sport medal rates from PAST Olympics only (the leakage fix)
cv.py        → WalkForwardOlympicsSplit (train years < test year)
model.py     → StandardScaler + GradientBoostingClassifier; grid search scored on PR-AUC
evaluation.py→ PR-AUC, max-F1 threshold, calibration + Brier, bootstrap 95% CIs
betting.py   → implied_odds, expected_value, kelly_fraction, simulate_betting
```
Every module has tests in `tests/`. Notebooks 1–3 are thin: they call `src/` and plot.

## The three decisions to lead with in interviews

1. **Fixed target leakage.** Old code computed country/sport medal rates over the *whole*
   dataset, so a 2016 athlete's "country rate" already contained 2020. `leakage.py` rebuilds
   them from earlier years only. *This is why the honest AUC is lower than the old 0.751 —
   and that drop is the point, not a regression.*
2. **Walk-forward CV instead of random k-fold.** Random folds train on future Olympics to
   predict past ones — invalid for time-ordered data. Walk-forward (2008→2012→2016→2020)
   mirrors real use.
3. **Imbalance-aware metrics + calibration.** Accuracy is 89% only because 89% of athletes
   don't medal. I report PR-AUC, the max-F1 threshold, a calibration curve + Brier score,
   and bootstrap CIs — so a probability of 0.3 actually means ~30%.

## The quant angle (the differentiator)

`betting.py` treats each predicted probability as a bet against a naive "market" = the
10.8% base rate. `expected_value` is positive exactly when the model is more confident than
the market (`p_model > p_market`); `kelly_fraction` sizes the bet; `simulate_betting` runs a
1-unit-per-athlete PnL across thresholds. The interview line: *"a classifier is only useful
where it disagrees with the market — so I measured the edge in EV terms, not just AUC."*

## What I dropped and why

- **Synthetic features** (HRV, VO2max, blood oxygen, body fat, injury/fitness/risk scores)
  were generated with `random` — they'd be flagged instantly as fake signal. Removed.
- **The non-gender pipeline** and the **pickle artifacts** — the package rebuilds everything
  from the DB, so cached `.pkl` files are gitignored.

## Known limitations (say these before they ask)

- Medal rates are by sport, not event (rowing/swimming inflate their athletes).
- Only 4 Olympics → 3 walk-forward folds → wide CIs.
- "Market" is a flat base rate, not real odds — the betting layer demos the framework.

## Engineering signals

`pyproject.toml` (pinned deps, src-layout), `pytest` suite, `ruff` + `mypy` clean, CI on
every push (`.github/workflows/`), pre-commit with `nbstripout`. Reproducible from the DB via
the three notebooks.

## TODO before sending to recruiters

- [x] `Run All` on `notebook_3_results.ipynb` so figures/outputs are saved in the file.
- [x] Paste the walk-forward numbers into the README results table.
