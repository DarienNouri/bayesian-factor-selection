# %%  [markdown]
# # ISYE Final Project Code
# **Author:** Darien Nouri \
# **Email:** dnouri3@gatech.edu | nouri.darien@gmail.com \
# **gtID:** 904081737
#

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from datetime import datetime
import os

os.makedirs('images', exist_ok=True)

GTID = 904081737

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)

np.random.seed(GTID)

az.style.use('arviz-darkgrid')

# %% load data

data = pd.read_csv('nyc_housing_data.csv', date_format="%Y-%m-%d", parse_dates=['yr-month'])
print("Data head:")
data.head()

# %%

data['yr-month'] = pd.to_datetime(data['yr-month'])
data = data.set_index('yr-month')
data['citi_bike_ride_volume'] = data['citi_bike_ride_volume'].diff(12).fillna(0)

print("\nBasic stats of data:")
data.describe()

# %%  [markdown]
# # 3. EDA
# %%

fig = plt.figure(figsize=(18, 15))

normalized_data = data.copy()
for column in normalized_data.columns:
    min_val = normalized_data[column].min()
    max_val = normalized_data[column].max()
    normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)

plt.subplot(421)
plt.plot(data.index, normalized_data['average_transaction'], 'g:', linewidth=2, label='Avg Transaction Price')
plt.title('Average Real Estate Transaction Price')
plt.ylabel('Normalized Value')
plt.grid(True)
plt.legend()

plt.subplot(422)
plt.plot(data.index, normalized_data['average_transaction'], 'g:', linewidth=2, label='Avg Transaction Price')
plt.plot(data.index, normalized_data['citi_bike_ride_volume'], 'b-', linewidth=2, label='Citi Bike Rides', color='red')
plt.title('Citi Bike Ride Volume')
plt.grid(True)
plt.legend()

plt.subplot(423)
plt.plot(data.index, normalized_data['average_transaction'], 'g:', linewidth=2, label='Avg Transaction Price')
plt.plot(data.index, normalized_data['average_restaurant_heath_inspection'], 'r-', linewidth=2, label='Health Inspection', color='red', linestyle="--")
plt.title('Avg Restaurant Health Inspection Score')
plt.grid(True)
plt.legend()

plt.subplot(424)
plt.plot(data.index, normalized_data['average_transaction'], 'g:', linewidth=2, label='Avg Transaction Price')
plt.plot(data.index, normalized_data['new_business_applications'], 'm-', linewidth=2, label='Business Applications', color='red', linestyle="--")
plt.title('New Business Applications')
plt.grid(True)
plt.legend()

plt.subplot(425)
plt.plot(data.index, normalized_data['average_transaction'], 'g:', linewidth=2, label='Avg Transaction Price')
plt.plot(data.index, normalized_data['average_evictions'], 'c-', linewidth=2, label='Evictions', color='red', linestyle="--")
plt.title('Average Evictions')
plt.grid(True)
plt.legend()

plt.subplot(426)
plt.plot(data.index, normalized_data['average_transaction'], 'g:', linewidth=2, label='Avg Transaction Price')
plt.plot(data.index, normalized_data['311_calls'], 'y-', linewidth=2, label='311 Calls', color='red', linestyle="--")
plt.title('311 Calls')
plt.grid(True)
plt.legend()

plt.subplot(427)
plt.plot(data.index, normalized_data['average_transaction'], 'g:', linewidth=2, label='Avg Transaction Price')
plt.plot(data.index, normalized_data['n_transactions'], 'k-', linewidth=2, label='Number of Transactions', color='red', linestyle="--")
plt.title('Number of Transactions')
plt.grid(True)
plt.legend()

plt.suptitle('NYC Alt Real Estate Data Indicators Over Time (Normalized)', fontsize=16)

plt.tight_layout()
plt.savefig('images/01-alt-data-indicators-over-time-normalized.png')
plt.show()

# %%

g = sns.pairplot(data[[
    'average_transaction',
    'citi_bike_ride_volume',
    'average_restaurant_heath_inspection',
    'new_business_applications',
]])

plt.suptitle('Relationships Between Transaction Price and Selected Indicators', y=1.02, fontsize=16)

for ax in g.axes.flat:
    ax.set_xlabel(ax.get_xlabel(), rotation=45, ha='right', fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), rotation=45, ha='right', fontsize=10)

    ax.tick_params(axis='both', which='major', labelsize=8)

plt.subplots_adjust(bottom=0.15, left=0.15)

plt.tight_layout()
plt.savefig('images/03-pairplot-selected-indicators.png', )
plt.show()

# %%  [markdown]
# # 4. Creating Lagged Features

# %%

max_lag = 8

predictors = [
    'citi_bike_ride_volume',
    'average_restaurant_heath_inspection',
    'new_business_applications',
    'average_evictions',
    '311_calls',
    'n_transactions',
]

data_with_lags = data.copy()

for predictor in predictors:
    for lag in range(1, max_lag + 1):
        data_with_lags[f'{predictor}_lag{lag}'] = data_with_lags[predictor].shift(lag)

data_with_lags = data_with_lags.dropna()

print(f"Original data shape: {data.shape}")
print(f"Data with lags shape: {data_with_lags.shape}")
print("\nColumns in the dataset with lags:")
print(data_with_lags.columns.tolist())

# %%  [markdown]
# # 5. Preparing Data for Bayesian Modeling

# %%

features = data_with_lags.columns.drop('average_transaction')
target = 'average_transaction'

target_mean = data_with_lags[target].mean()
target_std = data_with_lags[target].std()

X_scaled = (data_with_lags[features] - data_with_lags[features].mean()) / data_with_lags[features].std()
y_scaled = (data_with_lags[target] - target_mean) / target_std

data_scaled = pd.DataFrame(X_scaled, index=data_with_lags.index)
data_scaled[target] = y_scaled

print("Summary of standardized features:")
data_scaled.describe().T

# %%  [markdown]
# # 6. Bayesian Variable Selection Model
# %% # 6. Bayesian Variable Selection Model
x_cols_base = ['citi_bike_ride_volume', 'average_restaurant_heath_inspection',
       'new_business_applications', 'average_evictions', '311_calls',
       'n_transactions']

# Use only the selected features
X_data = data_scaled[x_cols_base].values
y_data = data_scaled[target].values
feature_names = x_cols_base
n_features = X_data.shape[1]
n_samples = X_data.shape[0]

print(f"Number of samples: {n_samples}")
print(f"Number of features: {n_features}")



with pm.Model() as m_base:
    sigma = pm.HalfCauchy("sigma_squared", beta=1)

    spike_scale = 0.001
    slab_scale = 1.0

    tau = pm.Beta("tau", alpha=1, beta=3)
    gamma = pm.Bernoulli("gamma", p=tau, shape=n_features)
    beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, shape=n_features)
    beta = pm.Deterministic("beta", beta_raw * (spike_scale * (1 - gamma) + slab_scale * gamma))
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    mu = alpha + pm.math.dot(X_data, beta)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=pm.math.sqrt(sigma), observed=y_data)

    trace_base = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        return_inferencedata=True,
        target_accept=0.97,
        random_seed=GTID,
        idata_kwargs={'log_likelihood': True},
    )

#%%
with m_base:
    posterior_predictive_base = pm.sample_posterior_predictive(trace_base, random_seed=GTID)

# %%
posterior_predictive_idata = posterior_predictive_base['posterior_predictive']
az.plot_ppc(posterior_predictive_base)
plt.title('Posterior Predictive Check - Bayesian Variable Selection Model')
plt.tight_layout()
plt.savefig('images/08b-posterior-predictive-check-base.png')
plt.show()

# %%
post_samples_base = trace_base.posterior.stack(sample=("chain", "draw"))

def get_expected_value_base(sample):
    return sample["alpha"].values + np.dot(X_data, sample["beta"].values)

y_pred_base = np.array([get_expected_value_base(post_samples_base.isel(sample=i)) for i in range(len(post_samples_base.sample))])
y_pred_mean_base = y_pred_base.mean(axis=0)
y_pred_mean_original_base = y_pred_mean_base * target_std + target_mean
y_actual_original = y_data * target_std + target_mean

plt.figure(figsize=(12, 8))
plt.plot(data_with_lags.index, y_actual_original, 'b-', label='Actual')
plt.plot(data_with_lags.index, y_pred_mean_original_base, 'r--', label='Predicted')
plt.fill_between(
    data_with_lags.index,
    np.percentile(y_pred_base, 2.5, axis=0) * target_std + target_mean,
    np.percentile(y_pred_base, 97.5, axis=0) * target_std + target_mean,
    alpha=0.3,
    color='r',
)
plt.title('Actual vs. Predicted Average Transaction Prices - Bayesian Variable Selection Model')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/09b-actual-vs-predicted-prices-base.png')
plt.show()

# %%

ss_res_base = np.sum((y_actual_original - y_pred_mean_original_base)**2)
ss_tot_base = np.sum((y_actual_original - np.mean(y_actual_original))**2)
r_squared_base = 1 - (ss_res_base / ss_tot_base)
print(f"Bayesian Variable Selection Model R2: {r_squared_base:.4f}")

mae_base = np.mean(np.abs(y_actual_original - y_pred_mean_original_base))
print(f"MAE: ${mae_base:.2f}")

rmse_base = np.sqrt(np.mean((y_actual_original - y_pred_mean_original_base)**2))
print(f"RMSE: ${rmse_base:.2f}")

# %%

with pm.Model() as m:
    sigma = pm.HalfCauchy("sigma_squared", beta=1)

    spike_scale = 0.001
    slab_scale = 1.0

    tau = pm.Beta("tau", alpha=1, beta=3)
    gamma = pm.Bernoulli("gamma", p=tau, shape=n_features)
    beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, shape=n_features)
    beta = pm.Deterministic("beta", beta_raw * (spike_scale * (1 - gamma) + slab_scale * gamma))
    alpha = pm.Normal("alpha", mu=0, sigma=1)
    mu = alpha + pm.math.dot(X_data, beta)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=pm.math.sqrt(sigma), observed=y_data)

    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        return_inferencedata=True,
        target_accept=0.97,
        random_seed=GTID,
        idata_kwargs={'log_likelihood': True},
    )

# %%

az.plot_trace(trace, var_names=["alpha", "sigma_squared", "tau", "beta"])
plt.suptitle('Trace Plots for Model Parameters', y=1.02)
plt.savefig('images/04-trace-plots-model-parameters.png', )
plt.show()

# %%

summary = az.summary(trace, var_names=["beta", "tau", "sigma_squared"])
summary['feature'] = [''] * len(summary)

for i, feature in enumerate(feature_names):
    if f"beta[{i}]" in summary.index:
        summary.at[f"beta[{i}]", 'feature'] = feature

summary_with_features = summary.copy()
summary_with_features = summary_with_features.sort_values('mean', ascending=False)
print("Posterior Summary (sorted by coefficient mean):")
summary_with_features = summary_with_features[['feature'] + list(summary_with_features.columns)[:-1]]
summary_betas = summary_with_features.loc[summary_with_features.index.str.contains('beta'), :]
summary_betas['abs_mean'] = summary_betas['mean'].abs()
summary_betas = summary_betas.sort_values('abs_mean', ascending=False)
summary_betas

# %%

inclusion_probs = az.summary(trace.posterior['gamma'], kind='stats')
inclusion_probs['feature'] = feature_names

inclusion_probs_sorted = inclusion_probs.sort_values('mean', ascending=False)
print("Posterior Inclusion Probabilities:")
pd.set_option('display.float_format', '{:.3f}'.format)
inclusion_probs_sorted.round(4)

inclusion_probs_with_betas = (
    pd.merge(
        inclusion_probs_sorted.reset_index(),
        summary_betas[['feature', 'mean']].reset_index(),
        on='feature',
        how='left',
        suffixes=('', '_beta'),
    ).rename(columns={'mean_beta': 'mean_beta'}).sort_values('mean', ascending=False).head(15))
print(inclusion_probs_with_betas)
print(inclusion_probs_with_betas.to_csv(index=False))

# %%  [markdown]
# # 7. Visualizing Results

# %% 7. Visualizing Results

top_predictors = []
for feature in feature_names:
    base_feature = feature.split('_lag')[0] if '_lag' in feature else feature
    if base_feature not in top_predictors and base_feature in predictors:
        top_predictors.append(base_feature)

n_predictors = len(top_predictors)
n_cols = 3
n_rows = (n_predictors + n_cols - 1) // n_cols

fig = plt.figure(figsize=(24, 12))

for i, predictor in enumerate(top_predictors):
    predictor_indices = [j for j, feature in enumerate(feature_names) if feature == predictor or feature.startswith(f"{predictor}_lag")]
    predictor_features = [feature_names[j] for j in predictor_indices]

    lag_numbers = [0 if '_lag' not in feature else int(feature.split('_lag')[1]) for feature in predictor_features]

    sorted_indices = np.argsort(lag_numbers)
    sorted_lag_numbers = [lag_numbers[j] for j in sorted_indices]
    sorted_features = [predictor_features[j] for j in sorted_indices]

    coef_means = [summary.loc[f"beta[{predictor_indices[j]}]", 'mean'] for j in sorted_indices]
    inclusion_probs_values = [inclusion_probs.loc[f"gamma[{predictor_indices[j]}]", 'mean'] for j in sorted_indices]

    row = i // n_cols
    col = i % n_cols
    subplot_idx = row * n_cols + col + 1

    plt.subplot(n_rows, n_cols, subplot_idx)

    ax1 = plt.gca()
    ax1.plot(sorted_lag_numbers, coef_means, 'o-', color='b', markersize=8, label='Coefficient')
    ax1.set_ylabel('Coefficient Value', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(f'Lag Structure for {predictor}')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.bar(sorted_lag_numbers, inclusion_probs_values, alpha=0.3, color='r', label='Inclusion Prob')
    ax2.set_ylabel('Inclusion Probability', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Lag (months)')
    ax1.set_xticks(sorted_lag_numbers)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.suptitle('Lag Structures for Top Predictors', fontsize=16)
plt.tight_layout()
plt.savefig('images/07-combined-lag-structures.png', )
plt.show()

# %%   [markdown]
# 8. Model Validation

with m:
    posterior_predictive = pm.sample_posterior_predictive(trace, random_seed=GTID)

# %%

posterior_predictive_idata = posterior_predictive['posterior_predictive']
#%%
fig, axes = plt.subplots(2, 1, figsize=(8, 10))
az.plot_ppc(posterior_predictive_base, ax=axes[0], )
axes[0].set_title('Base Model With Original Features')
az.plot_ppc(posterior_predictive, ax=axes[1], )
axes[1].set_title('Model With Lagged Features')
plt.tight_layout()
plt.savefig('images/08-posterior-predictive-check-comparison-both.png', )
plt.show()

# %%

post_samples = trace.posterior.stack(sample=("chain", "draw"))


def get_expected_value(sample):
    return sample["alpha"].values + np.dot(X_data, sample["beta"].values)


y_pred = np.array([get_expected_value(post_samples.isel(sample=i)) for i in range(len(post_samples.sample))])
y_pred_mean = y_pred.mean(axis=0)

y_pred_mean_original = y_pred_mean * target_std + target_mean
y_actual_original = y_data * target_std + target_mean

plt.figure(figsize=(12, 8))
plt.plot(data_with_lags.index, y_actual_original, 'b-', label='Actual')
plt.plot(data_with_lags.index, y_pred_mean_original, 'r--', label='Predicted')
plt.fill_between(
    data_with_lags.index,
    np.percentile(y_pred, 2.5, axis=0) * target_std + target_mean,
    np.percentile(y_pred, 97.5, axis=0) * target_std + target_mean,
    alpha=0.3,
    color='r',
)
plt.title('Actual vs. Predicted Average Transaction Prices')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/09-actual-vs-predicted-prices.png', )
plt.show()

# %%

ss_res_base = np.sum((y_actual_original - y_pred_mean_original_base)**2)
ss_tot_base = np.sum((y_actual_original - np.mean(y_actual_original))**2)
r_squared_base = 1 - (ss_res_base / ss_tot_base)
print(f"Bayesian Variable Selection Model R2: {r_squared_base:.4f}")

mae_base = np.mean(np.abs(y_actual_original - y_pred_mean_original_base))
print(f"MAE: ${mae_base:.2f}")

rmse_base = np.sqrt(np.mean((y_actual_original - y_pred_mean_original_base)**2))
print(f"RMSE: ${rmse_base:.2f}")

ss_res = np.sum((y_actual_original - y_pred_mean_original)**2)
ss_tot = np.sum((y_actual_original - np.mean(y_actual_original))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Model R2: {r_squared:.4f}")

mae = np.mean(np.abs(y_actual_original - y_pred_mean_original))
print(f"MAE: ${mae:.2f}")

rmse = np.sqrt(np.mean((y_actual_original - y_pred_mean_original)**2))
print(f"RMSE: ${rmse:.2f}")

comparison_df = pd.DataFrame({
    'Metric': ['R2', 'MAE ($)', 'RMSE ($)'],
    'Base_Model': [f"{r_squared_base:.4f}", f"${mae_base:.2f}", f"${rmse_base:.2f}"],
    'Model_Lagged_Features': [f"{r_squared:.4f}", f"${mae:.2f}", f"${rmse:.2f}"]
})

print("\nModel Performance Comparison:")
print(comparison_df)
comparison_df.to_latex()
# %%  [markdown]
# # 9. Interpreting Results and Drawing Conclusions

# %%

top_features_by_inclusion = inclusion_probs_sorted.head(12)

summary_table = pd.DataFrame({
    'Feature':
    top_features_by_inclusion['feature'],
    'Inclusion Probability':
    top_features_by_inclusion['mean'],
    'Coefficient Mean': [summary.loc[f"beta[{i}]", 'mean'] for i in range(n_features) if f"gamma[{i}]" in top_features_by_inclusion.index],
    'Coefficient 95% HDI': [
        f"({summary.loc[f'beta[{i}]', 'hdi_3%']:.3f}, {summary.loc[f'beta[{i}]', 'hdi_97%']:.3f})" for i in range(n_features)
        if f"gamma[{i}]" in top_features_by_inclusion.index
    ],
})

print("Top Predictors:")
summary_table
