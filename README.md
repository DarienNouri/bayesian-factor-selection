# Bayesian lag selection for NYC home prices

- [Full Report](report.pdf)
- [Code](bayesian_analysis.ipynb)
- [Data](nyc_housing_data.csv)

## Overview

Bayesian spike-and-slab regression in PyMC that selects which lagged NYC indicators predict the city's average monthly property transaction price. Of 54 candidates (six indicators at lags of 0 to 8 months), 311 calls at a 2-month lag and evictions at 6 and 8 months have the highest posterior inclusion probabilities (0.923, 0.839, 0.848). Adding the lags raises in-sample R² from 0.774 to 0.864.

## Results

Numbers below come from the [report](report.pdf) (Table 2 and the model-fit table).

**Candidates.** 54 predictors: six monthly NYC indicators (Citi Bike rides, restaurant health scores, new business applications, evictions, 311 calls and property transactions), each at lags of 0 to 8 months. The data has 102 monthly observations, late 2014 to early 2023; the model is fit on the last 94 after building the lags.

**Selection.** Four predictors stand out by posterior inclusion probability (PIP). The remaining 50 have a PIP of 0.10 or less.

| Predictor (lag) | PIP | Coef. mean (standardized) |
| --- | --- | --- |
| 311 calls (2 months) | 0.923 | 0.285 |
| Evictions (8 months) | 0.848 | -0.312 |
| Evictions (6 months) | 0.839 | -0.328 |
| Transactions (6 months) | 0.482 | 0.082 |

**Fit.** In-sample R² is 0.864 with the lagged features, against 0.774 for the same model on current-month features only (no lags).

The report notes that the COVID-19 eviction pause likely inflates the importance of the eviction indicators.

![Posterior mean coefficient (line) and inclusion probability (bars) at lags 0 to 8 months for each indicator, from Figure 2 of the report](images/07-lag-structures-lagged-model.png)

This figure comes from a separate MCMC run from Table 2, so its bar heights differ from the table.
