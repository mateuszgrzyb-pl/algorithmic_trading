#%% 1. Import libraries.
import warnings
from datetime import datetime

import pandas as pd

from src.utils.tools import add_quarterly_vix, fetch_live_features
warnings.filterwarnings('ignore')

URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
DATE = '2026Q1'
FEATURES = ['roe', 'price_to_sales']

#%% 2. Load tickers.
sp500 = pd.read_html(
    URL,
    storage_options={
        "User-Agent": "Mozilla/5.0"
    }
)[0]
tickers = sp500['Symbol'].tolist()

#%% 3. Download data and create features.

df_results = fetch_live_features(DATE, tickers, verbose=True)
df_results = df_results.dropna(subset=['price_to_sales', 'roe'])
df_results = add_quarterly_vix(df_results, date_col='price_date', vix_col='vix')
df_results = df_results[FEATURES + ['price_date']].rename(columns={'price_date': 'date'})
df_results = df_results.dropna()

#%% 4. Save the data.
version = datetime.now().strftime('%Y%m%d')
df_results.to_feather(f'data/scoring/scoring_set_{version}.feather')
