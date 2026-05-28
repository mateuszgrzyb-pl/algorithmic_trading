"""
One-dimensional feature analysis using decision trees.
"""
from typing import List, Union, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import tqdm
import warnings


"""
One-dimensional feature analysis using decision trees.
"""
from typing import List, Union, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import tqdm


def one_dim_analysis(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    features_to_check: List[str],
    scoring: Optional[str] = None,
    cv: int = 5,
    max_depth: int = 3,
    min_samples_leaf: Union[int, float] = 0.05,
    n_jobs: int = 1,
    random_state: int = 42,
    regression: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Perform one-dimensional feature analysis using decision trees.
    
    This function evaluates each feature independently by training a decision tree
    on that single feature and computing cross-validation scores. This helps identify
    which features have the strongest individual predictive power.

    Parameters
    ----------
    X : pandas.DataFrame
        Input data containing independent variables.
    y : pandas.Series or numpy.ndarray
        Target variable to predict.
    features_to_check : list of str
        List of feature names (columns) to evaluate in the analysis.
    scoring : str, optional
        Scoring metric used during cross-validation. If None, uses
        'neg_root_mean_squared_error' for regression or 'roc_auc' for classification.
    cv : int, default=5
        Number of cross-validation folds.
    max_depth : int, default=3
        Maximum depth of the decision tree.
    min_samples_leaf : int or float, default=0.05
        Minimum number of samples required at a leaf node. If float in range
        (0, 1), it is treated as a percentage of samples.
    n_jobs : int, default=1
        Number of parallel jobs to run during cross-validation. Use -1 for all cores,
        -2 for all cores minus one. Set to 1 to disable parallelization (recommended
        for Jupyter notebooks to avoid pickling issues).
    random_state : int, default=42
        Random state for controlling the randomness of the decision tree.
    regression : bool, default=False
        Flag indicating whether to use a regressor (True) or classifier (False).
    verbose : bool, default=True
        Flag to enable (True) or disable (False) the progress bar.

    Returns
    -------
    pandas.DataFrame
        DataFrame with features as index and columns:
        - 'mean_score': Mean cross-validation score for each feature
        - 'std_score': Standard deviation of cross-validation scores
        Sorted by mean_score in descending order (best features first).

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    >>> results = one_dim_analysis(df, y, df.columns.tolist(), regression=False)
    >>> print(results.head())
    >>> # Shows top 5 features ranked by predictive power

    Notes
    -----
    - Higher scores are better for all metrics (including negative metrics like RMSE)
    - Each feature is evaluated independently, ignoring interactions
    - Useful for initial feature selection and understanding feature importance
    """
    if regression:
        estimator = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        if scoring is None:
            scoring = 'neg_root_mean_squared_error'
    else:
        estimator = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        if scoring is None:
            scoring = 'roc_auc'
    
    results = []
    for feature in tqdm.tqdm(features_to_check, disable=(not verbose)):
        cv_scores = cross_val_score(
            estimator,
            X[[feature]],
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )
        results.append({
            'feature': feature,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores)
        })
    
    results_df = pd.DataFrame(results)
    results_df.set_index('feature', inplace=True)
    results_df.sort_values('mean_score', ascending=False, inplace=True)
    
    return results_df


def correlation_features_selection(
    X: pd.DataFrame,
    one_dim_results: pd.DataFrame,
    corr_level: float = 0.5,
    method: str = 'pearson',
    verbose: bool = True
) -> List[str]:
    """
    Select features based on correlation analysis.
    
    This function iteratively selects features by taking the best feature from
    one-dimensional analysis results, then removing all features that are highly
    correlated with it. The process continues until all features are either
    selected or removed due to high correlation with a selected feature.

    Parameters
    ----------
    X : pandas.DataFrame
        Input data containing independent variables.
    one_dim_results : pandas.DataFrame
        Results from one-dimensional analysis containing feature rankings.
        Should have features as index and 'mean_score' column (or be sorted
        with best features first).
    corr_level : float, default=0.5
        Correlation threshold above which features are removed. Features with
        absolute correlation greater than this value with any selected feature
        will be excluded.
    method : str, default='pearson'
        Method for calculating correlation. Options: 'pearson', 'spearman', 'kendall'.
    verbose : bool, default=True
        Flag to enable (True) or disable (False) the progress bar.

    Returns
    -------
    list of str
        List of selected feature names that passed the correlation threshold.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> 
    >>> # Generate data
    >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    >>> df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    >>> 
    >>> # Run one-dimensional analysis first
    >>> one_dim_results = one_dim_analysis(df, y, df.columns.tolist())
    >>> 
    >>> # Select features with low correlation
    >>> selected = correlation_features_selection(df, one_dim_results, corr_level=0.7)
    >>> print(f"Selected {len(selected)} features from {df.shape[1]}")

    Notes
    -----
    - Features are processed in order of their one-dimensional analysis scores
    - When a feature is selected, all highly correlated features are removed
    - This is a greedy algorithm that may not find the optimal subset
    - Lower corr_level values result in fewer selected features
    """
    features_to_check = one_dim_results.index.tolist()
    selected_features = []

    progress_bar = tqdm.tqdm(total=len(features_to_check), disable=(not verbose))

    # Suppress runtime warnings for zero-variance features
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        while len(features_to_check) >= 1:
            # Take the best remaining feature
            current_feature = features_to_check.pop(0)
            selected_features.append(current_feature)
            progress_bar.update(1)
            
            if len(features_to_check) == 0:
                break
            
            # Calculate correlations with remaining features
            corr_tab = X[features_to_check]\
                .corrwith(X[current_feature], method=method)\
                .abs()\
                .sort_values(ascending=False)
            
            # Find features to drop (highly correlated)
            features_to_drop = corr_tab[corr_tab > corr_level].index.tolist()
            
            # Remove highly correlated features
            features_to_check = [f for f in features_to_check if f not in features_to_drop]
            
            # Update progress bar for dropped features
            progress_bar.update(len(features_to_drop))
    
    progress_bar.close()
    
    return selected_features

