"""
Mean encoding for categorical features.
"""
from typing import List, Dict, Union, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MeanEncoder(BaseEstimator, TransformerMixin):
    """
    Mean encoding transformer for categorical features.
    
    This transformer replaces categorical values with the mean of the target
    variable for each category. It follows the scikit-learn transformer pattern
    with fit() and transform() methods.
    
    Parameters
    ----------
    features : list of str, optional
        List of feature names to encode. If None, all object/category dtype
        columns will be encoded.
    
    Attributes
    ----------
    encoding_dict_ : dict
        Dictionary mapping feature names to their encoding dictionaries.
        Format: {feature_name: {category: mean_value}}
    global_mean_ : float
        Global mean of the target variable, used as fallback for unseen categories.
    features_ : list of str
        List of features that were actually encoded.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df_train = pd.DataFrame({
    ...     'city': ['Warsaw', 'Krakow', 'Warsaw', 'Gdansk', 'Krakow'],
    ...     'target': [100, 200, 150, 300, 250]
    ... })
    >>> df_test = pd.DataFrame({'city': ['Warsaw', 'Krakow', 'Poznan']})
    >>> 
    >>> encoder = MeanEncoder(features=['city'])
    >>> encoder.fit(df_train, df_train['target'])
    >>> print(encoder.encoding_dict_)
    >>> df_test_encoded = encoder.transform(df_test)
    """
    
    def __init__(self, features: Optional[List[str]] = None):
        self.features = features
        self.encoding_dict_: Dict[str, Dict] = {}
        self.global_mean_: Optional[float] = None
        self.features_: Optional[List[str]] = None
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'MeanEncoder':
        """
        Fit the mean encoder on training data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Training data containing categorical features.
        y : pandas.Series or numpy.ndarray
            Target variable.
        
        Returns
        -------
        self : MeanEncoder
            Fitted encoder instance.
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)
        
        self.global_mean_ = y.mean()
        
        if self.features is None:
            self.features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.features_ = self.features
        
        self.encoding_dict_ = {}
        for feature in self.features_:
            # Create temporary dataframe with feature and target
            temp_df = pd.DataFrame({
                'feature': X[feature],
                'target': y
            })
            means = temp_df.groupby('feature')['target'].mean()
            self.encoding_dict_[feature] = means.to_dict()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using fitted mean encoding.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Data to transform.
        
        Returns
        -------
        pandas.DataFrame
            Transformed data with encoded features.
        """
        X = X.copy()
        
        for feature in self.features_:
            X[feature] = X[feature].map(self.encoding_dict_[feature]).fillna(self.global_mean_)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Fit the encoder and transform the data in one step.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Training data containing categorical features.
        y : pandas.Series or numpy.ndarray
            Target variable.
        
        Returns
        -------
        pandas.DataFrame
            Transformed training data.
        """
        return self.fit(X, y).transform(X)
    
    def get_encoding_dict(self, feature: Optional[str] = None) -> Union[Dict, Dict[str, Dict]]:
        """
        Get the encoding dictionary for inspection.
        
        Parameters
        ----------
        feature : str, optional
            Specific feature name. If None, returns all encoding dictionaries.
        
        Returns
        -------
        dict or dict of dict
            Encoding dictionary for the specified feature, or all encodings if
            feature is None.
        """
        if feature is None:
            return self.encoding_dict_
        return self.encoding_dict_.get(feature, {})
