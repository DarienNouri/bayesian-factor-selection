# A Bayesian Framework for Temporal Variable Selection in Autocorrelated Systems

- [Full Report](report.pdf)
- [Code](bayesian_analysis.ipynb)
- [Data](nyc_housing_data.csv)

## Overview

This project presents a Bayesian spike-and-slab framework for identifying predictive temporal structures in high-dimensional, autocorrelated datasets. The model employs sparsity-inducing priors to probabilistically select informative predictors from a large set of lagged variables, focusing on inference rather than data specificity.

Applied to NYC housing prices, the method identifies key lagged indicators with high posterior inclusion probabilities while offering interpretable posterior distributions over coefficients and outcomes. These posterior insights—alongside posterior predictive checks—enable evaluation of model fit and predictive reliability not through single metrics, but via full distributional behavior.

The analysis demonstrates how Bayesian modeling provides not just selection, but structured uncertainty quantification and probabilistic reasoning across interchangeable time-dependent datasets.