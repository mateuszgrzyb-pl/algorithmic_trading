"""
Binning features in many ways. ;-)
"""
from typing import List, Union, Optional, Dict
import copy

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import tqdm


class TreeBinner(BaseEstimator, TransformerMixin):
    """
    Bin features using decision trees.
    
    This transformer creates bins for specified features by fitting a decision tree
    on each feature individually and replacing the original values with the leaf
    indices from the tree.
    
    Parameters
    ----------
    features : list of str, optional
        List of feature names to bin. If None, all numeric columns will be binned.
    min_samples_leaf : int or float, default=0.05
        Minimum number of samples required to be at a leaf node. If float in range
        (0, 1), it is treated as a percentage of samples.
    max_depth : int, default=3
        Maximum depth of the decision tree.
    random_state : int, default=42
        Random state used to control the randomness of the decision tree estimator.
    regression : bool, default=False
        Flag indicating whether to use a regressor (True) or classifier (False).
    verbose : bool, default=False
        Flag to enable (True) or disable (False) the progress bar.
    
    Attributes
    ----------
    estimators_ : dict
        Dictionary mapping feature names to their fitted decision tree estimators.
        Format: {feature_name: fitted_estimator}
    features_ : list of str
        List of features that were actually binned.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    >>> df_train = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    >>> df_test = pd.DataFrame(X[:100], columns=[f'feature_{i}' for i in range(5)])
    >>> 
    >>> binner = TreeBinner(features=['feature_0', 'feature_1'], max_depth=3)
    >>> binner.fit(df_train, y)
    >>> df_train_binned = binner.transform(df_train)
    >>> df_test_binned = binner.transform(df_test)
    >>> print(df_train_binned['feature_0'].nunique())  # Number of bins created
    
    Notes
    -----
    - The transformer creates copies of input dataframes, leaving originals unchanged.
    - Each feature is binned independently based on its relationship with the target.
    - Binning strategy is learned during fit() and applied during transform().
    """
    
    def __init__(
        self,
        features: Optional[List[str]] = None,
        min_samples_leaf: Union[int, float] = 0.05,
        max_depth: int = 3,
        random_state: int = 42,
        regression: bool = False,
        verbose: bool = False
    ):
        self.features = features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.regression = regression
        self.verbose = verbose
        
        self.estimators_: Dict[str, Union[DecisionTreeRegressor, DecisionTreeClassifier]] = {}
        self.features_: Optional[List[str]] = None
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'TreeBinner':
        """
        Fit the tree binner on training data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Training data containing features to bin.
        y : pandas.Series or numpy.ndarray
            Target variable used to fit the decision trees.
        
        Returns
        -------
        self : TreeBinner
            Fitted binner instance.
        """
        if self.features is None:
            self.features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.features_ = self.features
        
        if self.regression:
            estimator_class = DecisionTreeRegressor
        else:
            estimator_class = DecisionTreeClassifier
        
        self.estimators_ = {}
        for feature in tqdm.tqdm(self.features_, disable=(not self.verbose)):
            estimator = estimator_class(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            estimator.fit(X[[feature]], y)
            self.estimators_[feature] = estimator
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted binning strategy.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Data to transform.
        
        Returns
        -------
        pandas.DataFrame
            Transformed data with binned features.
        """
        X = X.copy()
        
        for feature in self.features_:
            X[feature] = self.estimators_[feature].apply(X[[feature]]).astype(float)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Fit the binner and transform the data in one step.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Training data containing features to bin.
        y : pandas.Series or numpy.ndarray
            Target variable.
        
        Returns
        -------
        pandas.DataFrame
            Transformed training data with binned features.
        """
        return self.fit(X, y).transform(X)
    
    def get_estimators(self, feature: Optional[str] = None) -> Union[Dict, object]:
        """
        Get fitted estimators for inspection.
        
        Parameters
        ----------
        feature : str, optional
            Specific feature name. If None, returns all estimators.
        
        Returns
        -------
        dict or estimator object
            Dictionary of all estimators if feature is None, otherwise the
            estimator for the specified feature.
        """
        if feature is None:
            return self.estimators_
        return self.estimators_.get(feature)

