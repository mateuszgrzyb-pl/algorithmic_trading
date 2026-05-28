#%% 1. Import libraries.
import joblib
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings('ignore')

DATA_ID = '20260509'
DATE = '2026_Q1'
MODEL_NAME = 'rf_20260506_0103'

#%% 2. Load data to score.
df = pd.read_feather(f'data/scoring/scoring_set_{DATA_ID}.feather')
df = df.dropna()

#%% 3. Load model.
model = joblib.load(f'models/{MODEL_NAME}.joblib')

#%% 4. Score new data.
df['pred'] = model.predict(df)

#%% 5. Save predictions
df['pred'].sort_values(ascending=False).to_csv(f'data/predictions/{DATE}.csv')
