#%% 1. Import libraries.
import warnings
from datetime import datetime

import pandas as pd

from src.utils.tools import process_multiple_tickers
warnings.filterwarnings('ignore')

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATE = '2026Q1'

#%% 2. Load tickers.
sp500 = pd.read_html(
    URL,
    storage_options={
        "User-Agent": "Mozilla/5.0"
    }
)[0]
tickers = sp500['Symbol'].tolist()

#%% 3. Download data and create features.
features_to_model = ['graham_number_vs_price', 
                     'eps',
                     'price_to_sales', 
                     'book_value_per_share',
                     'price_to_earnings',
                     'market_cap',
                     'price_to_book']
df_results = process_multiple_tickers(tickers, DATE, verbose=True)
df_results = df_results.set_index('ticker')[features_to_model]
df_results = df_results[features_to_model]

#%% 4. Save the data.
version = datetime.now().strftime('%Y%m%d')
df_results.to_feather(f'data/scoring/scoring_set_{version}.feather')
