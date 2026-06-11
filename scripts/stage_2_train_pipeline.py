#%% 1. Import libraries.
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

from src.utils.tools import (
    add_quarterly_vix,
    calculate_cross_sectional_ndcg,
    calculate_cross_sectional_spearman,
    random_k_score,
    rank_gauss_transform,
    top_k_score
)
from src.modelling.model_pipeline import FactorModelPipeline

K = 5
FEATURES = ['price_to_sales', 'roe']
TARGET = 'label_2000_100_252_pct_change'

#%% 2. Load data.
df = pd.read_feather('data/abt/label_2000_100_252_clean.feather')
df = df.sort_values(['date', 'ticker'])
df = df[FEATURES + [TARGET, 'date', 'ticker']].dropna()

#%% 3. Split dataset.
tr = df[df.date <= '2014Q1'].copy()
va = df[(df.date > '2015Q1') & (df.date <= '2021Q1')].copy()
te = df[(df.date > '2021Q1') & (df.date <= '2024Q1')].copy()

#%% 4. Save dataset.
tr.to_feather('data/abt_clean/tr.feather')
va.to_feather('data/abt_clean/va.feather')
te.to_feather('data/abt_clean/te.feather')

#%% 5. Transform datasets.
tr['y'] = tr.groupby('date')[TARGET].transform(rank_gauss_transform)
va['y'] = va.groupby('date')[TARGET].transform(rank_gauss_transform)
te['y'] = te.groupby('date')[TARGET].transform(rank_gauss_transform)

for feature in FEATURES:
    tr[f'{feature}_rg'] = tr.groupby('date')[feature].transform(rank_gauss_transform)
    va[f'{feature}_rg'] = va.groupby('date')[feature].transform(rank_gauss_transform)
    te[f'{feature}_rg'] = te.groupby('date')[feature].transform(rank_gauss_transform)

tr = add_quarterly_vix(tr, date_col='date', vix_col='vix')
va = add_quarterly_vix(va, date_col='date', vix_col='vix')
te = add_quarterly_vix(te, date_col='date', vix_col='vix')

vix_min = 11.39  # lowest observed value
vix_max = 44.14  # highest observed value
tr['vix_norm'] = (tr['vix'] - vix_min) / (vix_max - vix_min)
va['vix_norm'] = (va['vix'] - vix_min) / (vix_max - vix_min)
te['vix_norm'] = (te['vix'] - vix_min) / (vix_max - vix_min)

#%% 6. Fitting model.
model = ols(data=tr,formula='y ~ -1 + price_to_sales_rg:vix_norm + roe_rg')\
    .fit(cov_type='HAC', cov_kwds={'maxlags': 4})

#%% 7. Model validation.
pred_tr = model.predict(tr)
pred_va = model.predict(va)
pred_te = model.predict(te)

tr['pred'] = pred_tr.values
va['pred'] = pred_va.values
te['pred'] = pred_te.values

cs_corr_tr = calculate_cross_sectional_spearman(tr, 'y', 'pred')
cs_corr_va = calculate_cross_sectional_spearman(va, 'y', 'pred')
cs_corr_te = calculate_cross_sectional_spearman(te, 'y', 'pred')

print('Raw model results:')
print('\tCORR TR: {}'.format(np.round(cs_corr_tr, 3)))  # 0.061
print('\tCORR VA: {}'.format(np.round(cs_corr_va, 3)))  # 0.044
print('\tCORR TE: {}'.format(np.round(cs_corr_te, 3)))  # 0.047

#%% 8. Checking advanced metrics.
cs_tr, cs_tr_random = calculate_cross_sectional_ndcg(tr, TARGET, 'pred', k=K, n_random=10)
cs_va, cs_va_random = calculate_cross_sectional_ndcg(va, TARGET, 'pred', k=K, n_random=10)
cs_te, cs_te_random = calculate_cross_sectional_ndcg(te, TARGET, 'pred', k=K, n_random=10)

print(f"\n\tNDCG TR vs random: {cs_tr:.3f} vs {cs_tr_random:.3f}")  # 0.527
print(f"\tNDCG VA vs random: {cs_va:.3f} vs {cs_va_random:.3f}")    # 0.523
print(f"\tNDCG TE vs random: {cs_te:.3f} vs {cs_te_random:.3f}")    # 0.635

print(f'\n\tTR model score vs random: {top_k_score(tr, TARGET, K).round(2):.2f}% vs {random_k_score(tr, TARGET, K).round(2):.2f}%')   # 31.59% vs 11.73%
print(f'\tVA model score vs random: {top_k_score(va, TARGET, K).round(2):.2f}% vs {random_k_score(va, TARGET, K).round(2):.2f}%')   # 36.27% vs 13.14%
print(f'\tTE model score vs random: {top_k_score(te, TARGET, K).round(2):.2f}% vs {random_k_score(te, TARGET, K).round(2):.2f}%')   # 37.25% vs 12.78%

#%% 9. Serializing model pipeline.

tr.drop(columns=['vix', 'vix_norm'], inplace=True)
va.drop(columns=['vix', 'vix_norm'], inplace=True)
te.drop(columns=['vix', 'vix_norm'], inplace=True)

pipeline = FactorModelPipeline(features=FEATURES)
pipeline.fit(tr, target=TARGET)
version = datetime.now().strftime('%Y%m%d_%H%M')
pipeline.save(f'models/ols_{version}.joblib')

#%% 10. Additional check of key metrics
tr['pred'] = pipeline.predict(tr).values
va['pred'] = pipeline.predict(va).values
te['pred'] = pipeline.predict(te).values

cs_corr_tr = calculate_cross_sectional_spearman(tr, 'y', 'pred')
cs_corr_va = calculate_cross_sectional_spearman(va, 'y', 'pred')
cs_corr_te = calculate_cross_sectional_spearman(te, 'y', 'pred')

print('\nSerialized pipeline results:')
print('\tCORR TR: {}'.format(np.round(cs_corr_tr, 3)))  # 0.061
print('\tCORR VA: {}'.format(np.round(cs_corr_va, 3)))  # 0.044
print('\tCORR TE: {}'.format(np.round(cs_corr_te, 3)))  # 0.047

print(f"\n\tNDCG TR vs random: {cs_tr:.3f} vs {cs_tr_random:.3f}")  # 0.527
print(f"\tNDCG VA vs random: {cs_va:.3f} vs {cs_va_random:.3f}")    # 0.523
print(f"\tNDCG TE vs random: {cs_te:.3f} vs {cs_te_random:.3f}")    # 0.635

print(f'\n\tTR model score vs random: {top_k_score(tr, TARGET, K).round(2):.2f}% vs {random_k_score(tr, TARGET, K).round(2):.2f}%')   # 31.59% vs 11.73%
print(f'\tVA model score vs random: {top_k_score(va, TARGET, K).round(2):.2f}% vs {random_k_score(va, TARGET, K).round(2):.2f}%')   # 36.27% vs 13.14%
print(f'\tTE model score vs random: {top_k_score(te, TARGET, K).round(2):.2f}% vs {random_k_score(te, TARGET, K).round(2):.2f}%')   # 37.25% vs 12.78%
