#%% 1. Import libraries.
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from src.utils.tools import (
    calculate_financial_ratios,
    top_k_score,
    random_k_score)

TARGET = 'label_2000_100_252_pct_change'

#%% 2. Load data.
df = pd.read_feather('data/abt/label_2000_100_252_clean.feather')
df = df.sort_values(['date', 'ticker'])

#%% 3. Prepare dataset.
df = df[df['date'] < '2024Q1']
cols_to_model = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False)<300].index.tolist()
cols_to_model = [col for col in cols_to_model if col not in ['y', 'date', 'ticker', 'industry', 'sector', 'label_1000_100_250_days_to_event', 'label_1000_100_250_target', 'label_1000_100_250_event_date', 'label_1000_100_250_pct_change', 'label_1000_100_250_final_price', 'adj_close']]
df = df.dropna(subset=cols_to_model)

#%% 4. Split dataset.
tr = df[df.date <= '2010Q1'].copy()
va = df[(df.date > '2010Q1') & (df.date <= '2021Q1')].copy()
te = df[df.date > '2021Q1'].copy()

tr.dropna(inplace=True)
va.dropna(inplace=True)
te.dropna(inplace=True)

#%% 5. Save dataset.
tr.to_feather('data/abt_clean/tr.feather')
va.to_feather('data/abt_clean/va.feather')
te.to_feather('data/abt_clean/te.feather')

#%% 6. Fitting model.
features_to_model = ['graham_number_vs_price', 
                     'eps',
                     'price_to_sales', 
                     'book_value_per_share',
                     'price_to_earnings',
                     'market_cap',
                     'price_to_book']

model = RandomForestRegressor(n_estimators=5000, max_depth=3, min_samples_leaf=0.1, random_state=2001, n_jobs=3)
model.fit(tr[features_to_model], tr[TARGET])

#%% 7. Model validation.
pred_tr = model.predict(tr[features_to_model])
pred_va = model.predict(va[features_to_model])
pred_te = model.predict(te[features_to_model])

corr_tr, _ = spearmanr(tr[TARGET], pred_tr)
corr_va, _ = spearmanr(va[TARGET], pred_va)
corr_te, _ = spearmanr(te[TARGET], pred_te)

print('CORR TR: {}'.format(np.round(corr_tr, 3)))  # 0.222
print('CORR VA: {}'.format(np.round(corr_va, 3)))  # 0.075
print('CORR TE: {}'.format(np.round(corr_te, 3)))  # 0.055

#%% 8. Checking advanced metrics.
tr['pred'] = pred_tr
va['pred'] = pred_va
te['pred'] = pred_te

print('TR model score: ', top_k_score(tr, TARGET).round(2))  # 32.2
print('TR randm score: ', random_k_score(tr, TARGET).round(2))  # 18.49
print('')
print('VA model score: ', top_k_score(va, TARGET).round(2))  # 22.14
print('VA randm score: ', random_k_score(va, TARGET).round(2))  # 16.19
print('')
print('TE model score: ', top_k_score(te, TARGET).round(2))  # 17.35
print('TE randm score: ', random_k_score(te, TARGET).round(2))  # -2.65

#%% 9. Saving model.
version = datetime.now().strftime('%Y%m%d_%H%M')
joblib.dump(model, f'models/rf_{version}.joblib')