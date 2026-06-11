import joblib

import pandas as pd
from statsmodels.formula.api import ols

from src.utils.tools import (
    add_quarterly_vix,
    rank_gauss_transform
)

class FactorModelPipeline:
    """
    Pipeline for cross-sectional factor modeling using OLS.

    This class encapsulates feature transformations (RankGauss, VIX addition, 
    and normalization) and the training/prediction logic of the OLS model.
    It ensures that Out-Of-Time (OOT) data is transformed exactly as the training data.

    Parameters
    ----------
    features : list of str
        List of numerical features to apply RankGauss transformation.
    vix_min : float, default 11.39
        Minimum historical value of VIX for normalization.
    vix_max : float, default 44.14
        Maximum historical value of VIX for normalization.
    """

    def __init__(self, features: list[str], vix_min: float = 11.39, vix_max: float = 44.14):
        self.features = features
        self.vix_min = vix_min
        self.vix_max = vix_max
        self.model = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data containing dates and raw features.
            
        Returns
        -------
        pd.DataFrame
            Transformed dataset ready for model inference.
        """
        X = df.copy()

        # 1. RankGauss features (działa per data, więc przy OOT na 1 dniu też zadziała)
        for feature in self.features:
            X[f"{feature}_rg"] = X.groupby("date")[feature].transform(rank_gauss_transform)

        # 2. Add and normalize VIX
        X = add_quarterly_vix(X, date_col="date", vix_col="vix")
        X["vix_norm"] = (X["vix"] - self.vix_min) / (self.vix_max - self.vix_min)

        return X

    def fit(self, df: pd.DataFrame, target: str) -> "FactorModelPipeline":
        """
        Transform training data and fit the OLS model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data containing features and the target variable.
        target : str
            The column name of the dependent variable.
            
        Returns
        -------
        FactorModelPipeline
            The fitted pipeline instance.
        """
        # Transformacja X
        data = self.transform(df)

        # Transformacja targetu Y (tylko w fazie treningu!)
        data["y"] = data.groupby("date")[target].transform(rank_gauss_transform)

        # Formuła (można ją sparametryzować w __init__, jeśli planujesz testować różne)
        formula = "y ~ -1 + price_to_sales_rg:vix_norm + roe_rg"
        
        self.model = ols(data=data, formula=formula).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4}
        )
        
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Transform out-of-sample data and generate predictions.
        
        Parameters
        ----------
        df : pd.DataFrame
            New data to score. Target column is not required.
            
        Returns
        -------
        pd.Series
            Model predictions.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
            
        X = self.transform(df)
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        """Save the entire pipeline to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FactorModelPipeline":
        """Load a saved pipeline from disk."""
        return joblib.load(filepath)
