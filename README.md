# Football Match Outcome Prediction

A supervised machine-learning project for Premier League match outcome classification: home win, draw or away win.

## Methods

- SQLite data extraction and pandas feature engineering.
- Recent team form using lagged rolling statistics, rest days, starting-lineup ratings and head-to-head history.
- Dummy baseline, KNN, Logistic Regression, Random Forest and Gradient Boosting.
- Chronological season splits, time-series cross-validation, randomized hyperparameter search, accuracy and macro F1.

## Notebooks

- [Revised evaluation](football_prediction_revised.ipynb): recommended version, with train-only imputation inside model pipelines, date-grouped CV, validation-based model selection, and one final historical holdout evaluation. Outputs are intentionally cleared because the source database is not in this repository.
- [Original exploratory experiment](football_prediction.ipynb): retained with its historical outputs for transparency. Its lineup imputation used all seasons and its test season was used repeatedly for comparison; those results should not be treated as independent or validated metrics.

The historical split covers training seasons 2008/09–2013/14, validation 2014/15, and test 2015/16. Since the original experiment already examined the test season, independent evaluation needs a new unseen season. Lineup-based predictions assume lineups are known at prediction time.

## Run locally

Use Python and the dependencies in `requirements.txt`. Obtain the original SQLite dataset through an authorized source; place `database.sqlite` next to the notebook. It must contain the Match, League, Team and Player_Attributes tables used in the original project.

```bash
python -m venv .venv
# Activate the virtual environment.
python -m pip install -r requirements.txt
jupyter notebook football_prediction_revised.ipynb
```

The dataset is not redistributed here. The revised notebook has not yet been run against it. This is an educational ML experiment, not a production forecasting service.
