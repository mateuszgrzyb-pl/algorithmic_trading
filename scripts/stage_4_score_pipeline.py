#%% 1. Import libraries.
import warnings

import pandas as pd

from src.modelling.model_pipeline import FactorModelPipeline

warnings.filterwarnings('ignore')

DATASET_ID = '20260611'
DATE = '2026Q1'
FEATURES = ['roe', 'price_to_sales']
MODEL_NAME = 'ols_20260611_1034'

#%% 2. Load data to score.
df = pd.read_feather(f'data/scoring/scoring_set_{DATASET_ID}.feather')

#%% 3. Load the pipeline.
pipeline = FactorModelPipeline.load(f'models/{MODEL_NAME}.joblib')

#%% 5. Score new data.
df['pred'] = pipeline.predict(df)

#%% 6. Save predictions
df['pred'].sort_values(ascending=False).to_csv(f'data/predictions/{DATE}.csv')
