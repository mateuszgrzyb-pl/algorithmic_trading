import time
import logging
from pathlib import Path
from typing import List, Set, Any, Dict, Union, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyxirr import xirr
import yfinance as yf
from scipy.stats import norm, rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import ndcg_score
from statsmodels.formula.api import ols

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def standardize_column_names(columns: List[str]) -> List[str]:
    """Convert column names to snake_case style.

    This function lowercases all characters and replaces spaces and hyphens
    with underscores.

    Args:
        columns (List[str]): List of column names.

    Returns:
        List[str]: Standardized column names.
    """
    return [col.lower().replace(" ", "_").replace("-", "_") for col in columns]


def ensure_directory(path: Path | str) -> None:
    """Ensure that a directory exists at the given path.

    Creates the directory and any missing parent directories if they
    do not already exist.

    Args:
        path (Path | str): Path to the directory.

    Returns:
        None
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_available_tickers(
    data_dir: Path | str = "data/raw/price_history/STAGE_1",
) -> List[str]:
    """Return a sorted list of available tickers from feather files.

    The function scans the given directory for files with the `.feather`
    extension and extracts ticker symbols from their filenames.

    Args:
        data_dir (Path | str, optional): Directory containing feather files.
            Defaults to "data/raw/price_history/STAGE_1".

    Returns:
        List[str]: Sorted list of unique ticker symbols.
    """
    p = Path(data_dir)
    tickers = [
        f.stem.replace(".feather", "") if f.suffix != "" else f.stem
        for f in p.glob("*.feather")
    ]
    # if filenames like "AAPL.feather" -> stem is "AAPL"
    tickers = sorted({t for t in tickers})
    return tickers


def _validate_and_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validates input and prepares a clean copy for calculations."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if df.empty:
        logger.warning("Input DataFrame is empty. Returning an empty copy.")
        return df.copy()

    # Define all columns that are absolutely required for the function to run
    required_columns: Set[str] = {
        "total_current_assets",
        "total_current_liabilities",
        "cash_and_cash_equivalents",
        "total_debt",
        "total_shareholder_equity",
        "total_assets",
        "net_income",
        "operating_income",
        "revenue",
        "weighted_average_shares",
    }

    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(list(missing_cols))}"
        )

    return df.copy()


def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures optional columns exist, filling with defaults if not."""
    optional_columns = {
        "short_term_investments": 0,
        "accounts_receivable": 0,
        "inventory": 0,
        "cost_of_goods_sold": np.nan,
        "gross_profit": np.nan,
        "ebitda": np.nan,
        "long_term_debt": 0,
        "interest_expense": 0,
        "goodwill": 0,
        "intangible_assets": 0,
        "retained_earnings": np.nan,
        "accumulated_other_comprehensive_income": 0,
        "income_before_tax": np.nan,
        "accounts_payable": 0,
        "eps": np.nan,
        "adj_close": np.nan,
        "short_term_debt": 0,
        "net_debt": np.nan,
    }

    for col, default in optional_columns.items():
        if col not in df.columns:
            df[col] = default
            logger.debug(
                "Added missing optional column '%s' with default: %s", col, default
            )

    return df


def _calculate_liquidity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates liquidity ratios."""
    return df.assign(
        current_ratio=safe_divide(
            df["total_current_assets"], df["total_current_liabilities"]
        ),
        quick_ratio=safe_divide(
            df["cash_and_cash_equivalents"]
            + df["short_term_investments"]
            + df["accounts_receivable"],
            df["total_current_liabilities"],
        ),
        cash_ratio=safe_divide(
            df["cash_and_cash_equivalents"], df["total_current_liabilities"]
        ),
    )


def _calculate_leverage_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates leverage (debt) ratios."""
    return df.assign(
        debt_to_equity=safe_divide(df["total_debt"], df["total_shareholder_equity"]),
        debt_to_assets=safe_divide(df["total_debt"], df["total_assets"]),
        net_debt_to_ebitda=safe_divide(df["net_debt"], df["ebitda"]),
        long_term_debt_to_equity=safe_divide(
            df["long_term_debt"], df["total_shareholder_equity"]
        ),
    )


def _calculate_profitability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates profitability ratios."""
    return df.assign(
        roa=safe_divide(df["net_income"], df["total_assets"]),
        roe=safe_divide(df["net_income"], df["total_shareholder_equity"]),
        roic=safe_divide(
            df["operating_income"], df["total_debt"] + df["total_shareholder_equity"]
        ),
    )


def _calculate_valuation_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates market valuation ratios."""
    book_value_per_share = safe_divide(
        df["total_shareholder_equity"], df["weighted_average_shares"]
    )

    revenue_per_share = safe_divide(df["revenue"], df["weighted_average_shares"])

    eps_bvps_product = df["eps"] * book_value_per_share
    graham_number = np.where(
        (eps_bvps_product > 0) & (~eps_bvps_product.isna()),
        np.sqrt(22.5 * eps_bvps_product),
        np.nan,
    )

    return df.assign(
        book_value_per_share=book_value_per_share,
        price_to_book=safe_divide(df["adj_close"], book_value_per_share),
        price_to_sales=safe_divide(df["adj_close"], revenue_per_share),
        price_to_earnings=safe_divide(df["adj_close"], df["eps"]),
        graham_number=graham_number,
        graham_number_vs_price=safe_divide(graham_number, df["adj_close"]),
        market_cap=(df["adj_close"] * df["weighted_average_shares"]),
        enterprise_value=(
            (df["adj_close"] * df["weighted_average_shares"]) + df["net_debt"]
        ),
        earnings_yield=safe_divide(
            df["operating_income"],
            (df["adj_close"] * df["weighted_average_shares"]) + df["net_debt"],
        ),
    )


def calculate_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a comprehensive set of financial ratios for a given DataFrame.

    This function serves as a pipeline that validates the input data, ensures all
    necessary columns are present, and then calculates various categories of
    financial ratios in a structured manner.

    Args:
        df (pd.DataFrame): DataFrame containing raw financial statement data.
            It must include a set of required columns for core calculations.

    Returns:
        pd.DataFrame: A new DataFrame with added columns for each calculated ratio.

    Raises:
        TypeError: If the input 'df' is not a pandas DataFrame.
        ValueError: If the DataFrame does not contain all required columns.
    """
    try:
        # Step 1: Validate and prepare a clean copy of the DataFrame
        prepared_df = _validate_and_prepare_df(df)

        # Step 2: Ensure all optional columns exist, adding them with defaults if not
        df_with_all_cols = _ensure_optional_columns(prepared_df)

        # Step 3: Sequentially calculate ratio categories
        final_df = (
            df_with_all_cols.pipe(_calculate_liquidity_ratios)
            .pipe(_calculate_leverage_ratios)
            .pipe(_calculate_profitability_ratios)
            .pipe(_calculate_valuation_ratios)
        )

        logger.info("Successfully calculated all financial ratios.")
        return final_df

    except (TypeError, ValueError) as e:
        logger.error("Input data validation failed: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred during ratio calculation: %s", e)
        raise


def calculate_portfolio_xirr(
    df: pd.DataFrame,
    x: pd.Index,
    pred: Union[pd.Series, np.ndarray],
    best_thresh: float,
    label: str,
    label_length: int = 12,
    investment_amount: float = 1000.0,
) -> Dict[str, Any]:
    """Simulate a trading strategy and calculate its XIRR performance.

    The function builds a simple portfolio backtest by:
    - Selecting candidate stocks whose prediction score exceeds a threshold.
    - Buying the best candidate each period (quarterly) until a max buy date.
    - Selling positions after a fixed horizon, using triple-barrier labels.
    - Tracking invested capital, cash flows, and final profit.
    - Calculating XIRR based on the resulting cash flow log.

    Args:
        df (pd.DataFrame): DataFrame with stock data and labeling columns.
        x: Features index used to align `pred` with `df`.
        pred: Predicted scores for stock selection.
        best_thresh (float): Threshold for stock selection.
        label (str): Base name of the label columns, e.g. "tb".
        label_length (int, optional): Holding horizon in months. Defaults to 12.
        investment_amount (int, optional): Amount of cash added at each buy date.
            Defaults to 1000.

    Returns:
        Dict[str, Any]: Dictionary with:
            - "log" (pd.DataFrame): Transactions log.
            - "xirr_percent" (float): Annualized return in percent.
            - "total_amount_invested" (float): Total invested capital.
            - "final_capital" (float): Portfolio final value.
            - "profit" (float): Profit = final capital - invested capital.
    """
    stocks = df.loc[x.index].copy()
    max_buy_date = stocks["date"].max().to_timestamp(how="end") - pd.DateOffset(
        months=label_length
    )
    stocks["score"] = pred
    stocks["end_date"] = stocks[f"{label}_event_date"].astype(str)
    stocks["end_adj_close"] = stocks.adj_close + (
        stocks.adj_close * (stocks[f"{label}_pct_change"] / 100)
    )
    cols = [
        "date",
        "end_date",
        "ticker",
        f"{label}_pct_change",
        "adj_close",
        "end_adj_close",
    ]
    stocks = stocks.loc[stocks["score"] > best_thresh]
    stocks = stocks.loc[stocks.groupby("date")["score"].idxmax()][cols]
    stocks["date"] = stocks["date"].astype(str)
    stocks["date"] = (
        pd.PeriodIndex(stocks["date"], freq="Q")
        .to_timestamp(how="end")
        .date.astype(str)
    )
    unique_dates = np.unique(stocks["date"].tolist() + stocks["end_date"].tolist())
    total_amount_invested = 0
    capital = 0
    wallet = {}
    log = []

    for _, todays_date in enumerate(unique_dates):
        # 1. Sprzedaż akcji (jeśli możliwa).
        stock_to_sell = wallet.get(todays_date)  # pobieram akcje do sprzedania
        if stock_to_sell is not None:  # gdy są jakieś akcje do sprzedania
            # 1.1. Pobieram akcje.
            stock_ticker = stock_to_sell["ticker"]
            stock_buy_price = stock_to_sell["buy_price"]
            stock_sell_price = stock_to_sell["sell_price"]
            stock_num_of_shares = stock_to_sell["num_of_shares"]

            # 1.2. Aktualizuję stan konta.
            total_amount = stock_num_of_shares * stock_sell_price
            capital = capital + total_amount

            # 1.3. Usuwam akcje z portfela.
            del wallet[todays_date]

            # dodanie loga
            log.append(
                [
                    todays_date,
                    "sprzedaż",
                    stock_ticker,
                    stock_sell_price,
                    stock_num_of_shares,
                    total_amount,
                    capital - total_amount,
                    capital,
                ]
            )

        # 2. Kupno akcji.
        if (todays_date in stocks["date"].tolist()) & (
            pd.to_datetime(todays_date) <= max_buy_date
        ):
            capital += investment_amount  # dodaję kapitał
            total_amount_invested += investment_amount

            # 2.1. Pobieram akcje.
            stock_to_buy = stocks[stocks["date"] == todays_date]
            stock_ticker = stock_to_buy["ticker"].values[0]
            stock_buy_price = stock_to_buy["adj_close"].values[0]
            stock_sell_price = stock_to_buy["end_adj_close"].values[0]
            stock_sell_date = stock_to_buy["end_date"].values[0]
            stock_num_of_shares = int(np.floor(capital / stock_buy_price))

            if stock_num_of_shares > 0:
                # 2.2. Aktualizuję stan konta.
                total_amount = stock_num_of_shares * stock_buy_price
                capital = capital - total_amount

                # 2.3. Dodaje akcje do portfela.
                wallet[stock_sell_date] = {
                    "ticker": stock_ticker,
                    "buy_price": stock_buy_price,
                    "sell_price": stock_sell_price,
                    "num_of_shares": stock_num_of_shares,
                    "buy_date": todays_date,
                }

                # dodanie loga
                log.append(
                    [
                        todays_date,
                        "kupno",
                        stock_ticker,
                        stock_buy_price,
                        stock_num_of_shares,
                        -total_amount,
                        capital + total_amount,
                        capital,
                    ]
                )

    log = pd.DataFrame(
        log,
        columns=[
            "data",
            "operacja",
            "ticker",
            "cena",
            "liczba_sztuk",
            "kwota_calkowita",
            "stan_konta_przed",
            "stan_konta_po",
        ],
    )
    if log.empty:
        srednioroczny_zwrot = 0
    else:
        log["data"] = pd.to_datetime(log["data"])
        xirr_value = xirr(log[["data", "kwota_calkowita"]])
        if xirr_value is not None:
            srednioroczny_zwrot = np.round(xirr_value * 100, 2)
        else:
            srednioroczny_zwrot = 0.0
    profit = capital - total_amount_invested
    return {
        "log": log,
        "xirr_percent": srednioroczny_zwrot,
        "total_amount_invested": total_amount_invested,
        "final_capital": capital,
        "profit": profit,
    }


def filter_sp500_companies(
    df: pd.DataFrame, sp500_path: Path | str = "data/raw/tickers_sp500.csv"
) -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows for tickers that were part of
    the S&P 500 in a given quarter.

    This function performs the filtering by:
    1. Loading the historical S&P 500 constituents from a CSV file.
    2. Creating a set of valid (quarter, ticker) pairs for efficient lookup.
    3. Applying a vectorized mask to the input DataFrame to select matching rows.

    Args:
        df (pd.DataFrame): The input DataFrame to filter. Must contain 'date'
            (as pd.Period[Q-DEC]) and 'ticker' columns.
        sp500_path (Path | str, optional): The path to the CSV file containing
            historical S&P 500 constituents. The CSV must have 'date' and
            'tickers' (comma-separated string) columns.
            Defaults to "data/raw/tickers_sp500.csv".

    Returns:
        pd.DataFrame: A new DataFrame containing only the filtered rows.

    Raises:
        FileNotFoundError: If the sp500_path does not exist.
        KeyError: If the required columns are missing in the input DataFrame or CSV file.
    """
    sp500_path = Path(sp500_path)
    logger.info(
        "Filtering DataFrame based on S&P 500 constituents from %s", sp500_path
    )

    try:
        # --- Step 1: Load and prepare the S&P 500 data ---
        sp500_df = pd.read_csv(sp500_path)
        sp500_df["date"] = pd.to_datetime(sp500_df["date"])

        # Convert date to the same quarterly period format as the main DataFrame
        sp500_df["quarter"] = sp500_df["date"].dt.to_period("Q-DEC")

        # --- Step 2: Create the lookup set using vectorized operations (much faster) ---
        # Explode the comma-separated ticker strings into multiple rows
        sp500_long = (
            sp500_df.assign(ticker=sp500_df["tickers"].str.split(","))
            .explode("ticker")
            .reset_index(drop=True)
        )
        # Strip whitespace from ticker symbols
        sp500_long["ticker"] = sp500_long["ticker"].str.strip()

        # Create the set of (quarter, ticker) tuples
        sp500_pairs = set(zip(sp500_long["quarter"], sp500_long["ticker"]))
        logger.debug(
            "Created a lookup set with %d (quarter, ticker) pairs.", len(sp500_pairs)
        )

        # --- Step 3: Filter the main DataFrame efficiently ---
        # The core business logic remains: check for membership in the set.
        # This implementation avoids a slow .apply() loop.
        # We create a temporary series of tuples from the DataFrame rows.
        df_pairs = pd.Series(zip(df["date"], df["ticker"]))

        # The `isin` method on a Series is highly optimized for checking against a set.
        mask = df_pairs.isin(sp500_pairs)

        filtered_df = df[mask].copy()

        logger.info("Filtering complete. Kept %d of %d rows.", len(filtered_df), len(df))
        return filtered_df

    except FileNotFoundError:
        logger.error("S&P 500 constituents file not found at: %s", sp500_path)
        raise
    except KeyError as e:
        logger.error("A required column is missing from the input data: %s", e)
        raise


def load_sp500_tickers(csv_path: str = "data/raw/tickers_sp500.csv") -> List[str]:
    """
    Load and parse S&P 500 tickers from CSV file.

    Args:
        csv_path: Path to CSV file containing S&P 500 tickers

    Returns:
        List of unique ticker symbols

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV file is empty or malformed
    """
    try:
        logger.info("Loading tickers from: %s", csv_path)

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Ticker CSV file not found: {csv_path}")

        tickers_df = pd.read_csv(csv_path)

        if "tickers" not in tickers_df.columns:
            raise ValueError("CSV file must contain 'tickers' column")

        tickers_df = tickers_df["tickers"].drop_duplicates()
        tickers = []

        for row in tickers_df:
            if pd.isna(row):
                continue
            for ticker in str(row).split(","):
                ticker = ticker.strip().upper()
                if ticker:
                    tickers.append(ticker)

        unique_tickers = np.unique(tickers).tolist()
        logger.info("Loaded %d unique tickers", len(unique_tickers))

        if not unique_tickers:
            raise ValueError("No valid tickers found in CSV file")

        return unique_tickers

    except Exception as e:
        logger.error("Failed to load tickers: %s", e)
        raise


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validation of a DataFrame for required columns.
    """
    required_columns = [
        'total_current_assets', 'total_current_liabilities', 'cash_and_cash_equivalents',
        'total_debt', 'total_assets', 'total_shareholder_equity', 'net_income',
        'revenue', 'operating_income', 'weighted_average_shares'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.error("Missing columns: %s", missing_columns)
        return False

    return True


def top_k_score(df: pd.DataFrame, target: str, k: int = 5) -> float:
    """Calculate the mean target value for top-k predictions grouped by date.

    For each date group, selects the k rows with the highest predicted values
    and computes the mean of their true target values. The final score is the
    average across all date groups.

    Args:
        df (pd.DataFrame): DataFrame containing 'date', 'pred', and target columns.
        target (str): Name of the target column.
        k (int): Number of top predictions to select per date group. Defaults to 5.

    Returns:
        float: Mean of per-date average true target values for top-k predictions.
    """
    scores = []
    for q, g in df.groupby('date'):
        top_pred_idx = g['pred'].nlargest(k).index
        top_true_sum = g.loc[top_pred_idx, target].mean()
        scores.append(top_true_sum)
    return np.mean(scores)


def random_k_score(df: pd.DataFrame, target: str, k: int = 5, random_state: int = 2001) -> float:
    """Calculate the mean target value for k randomly selected rows grouped by date.

    For each date group, samples k rows at random and computes the mean of their
    true target values. Serves as a random baseline to compare against top_k_score.

    Args:
        df (pd.DataFrame): DataFrame containing 'date', 'pred', and target columns.
        target (str): Name of the target column.
        k (int): Number of rows to sample per date group. Defaults to 5.
        random_state (int): Random seed for reproducibility. Defaults to 2001.

    Returns:
        float: Mean of per-date average true target values for randomly selected rows.
    """
    scores = []
    for q, g in df.groupby('date'):
        top_pred_idx = g.sample(k, random_state=random_state).index
        top_true_sum = g.loc[top_pred_idx, target].mean()
        scores.append(top_true_sum)
    return np.mean(scores)


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide two values, returning np.nan where division is undefined.

    Handles edge cases such as zero denominator or NaN inputs without raising
    exceptions. Intended for element-wise use on scalar values extracted from
    a Series or DataFrame.

    Args:
        numerator (float): The dividend.
        denominator (float): The divisor.

    Returns:
        float: Result of numerator / denominator, or np.nan if the denominator
            is zero, either argument is NaN, or an unexpected error occurs.
    """
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], np.nan)
        return result
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return np.nan
        return numerator / denominator
    except Exception:
        return np.nan


def get_quarter_price(ticker: str, quarter: str) -> Optional[float]:
    """Retrieve the closing price at the end of a given quarter.

    Fetches the last available closing price within a 7-day window ending on
    the final day of the specified quarter. Returns np.nan if no data is found
    or an error occurs during retrieval.

    Args:
        ticker (str): Stock ticker symbol (e.g. 'AAPL').
        quarter (str): Quarter string in pandas Period format (e.g. '2023Q4').

    Returns:
        Optional[float]: Closing price at quarter-end, or np.nan if unavailable.
    """
    try:
        end_dt = pd.Period(quarter).end_time
        start_dt = end_dt - pd.Timedelta(days=7)
        
        hist = yf.Ticker(ticker).history(
            start=start_dt, 
            end=end_dt, 
            auto_adjust=False
        )
        
        if hist.empty or "Adj Close" not in hist.columns:
            return np.nan
            
        return hist["Adj Close"].iloc[-1]
    except Exception as e:
        print(f"  ⚠️  Errot. Could not download price data for ticker: {ticker}: {e}")
        return np.nan





def get_top_interactions(shap_interaction_values: np.ndarray, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Extract and rank the strongest global feature interactions from SHAP values.

    Aggregates local SHAP interaction values by calculating the mean absolute 
    interaction across all observations. Since SHAP splits the interaction 
    strength symmetrically between two cells in the matrix (i.e., at [i, j] 
    and [j, i]), this function extracts the upper triangle and doubles the 
    values to reconstruct the total interaction effect.

    Args:
        shap_interaction_values (np.ndarray): A 3D NumPy array of shape 
            (n_samples, n_features, n_features) generated by a SHAP TreeExplainer.
        X (pd.DataFrame): The feature DataFrame used for the model, used here 
            to map indices to actual feature names.
        top_n (int): The number of top-ranked interaction pairs to return. 
            Defaults to 10.

    Returns:
        pd.DataFrame: A ranked DataFrame with columns 'Feature_A', 'Feature_B', 
            and 'Interaction_Strength', sorted in descending order of strength.
    """
    # 1. Obliczamy średnią wartość bezwzględną interakcji dla każdej pary cech w całym zbiorze
    # Wynik to macierz (n_features, n_features)
    mean_abs_interactions = np.abs(shap_interaction_values).mean(0)
    
    interactions = []
    feature_names = X.columns
    
    # 2. Iterujemy po macierzy (bierzemy tylko górny trójkąt, żeby nie dublować A-B i B-A)
    # Pomijamy przekątną (i==j), bo tam znajdują się "main effects", a nie interakcje
    for i in range(mean_abs_interactions.shape[0]):
        for j in range(i + 1, mean_abs_interactions.shape[1]):
            val = mean_abs_interactions[i, j]
            # Mnożymy razy 2, ponieważ w macierzy SHAP interakcja jest rozdzielona symetrycznie
            # (połowa wpływu jest w komórce [i,j], a połowa w [j,i])
            interactions.append({
                'Feature_A': feature_names[i],
                'Feature_B': feature_names[j],
                'Interaction_Strength': val * 2 
            })
    # 3. Tworzymy DataFrame i sortujemy
    df_interactions = pd.DataFrame(interactions)
    df_interactions = df_interactions.sort_values(by='Interaction_Strength', ascending=False)
    
    return df_interactions.head(top_n)


class CrossSectionalStationarityTransformer(BaseEstimator, TransformerMixin):
    """Transformer for removing non-stationarity and inflationary trends in financial data.

    Applies transformations cross-sectionally per quarter (or point in time).
    Preserves relative distances and proportions between entities within the same period,
    enabling tree-based models (e.g., Random Forest) to capture non-linear relationships
    such as U-shaped patterns.
    """

    def __init__(
        self, features_to_scale: List[str], log_features: Optional[List[str]] = None
    ) -> None:
        """Initialize the transformer with specific feature groups.

        Args:
            features_to_scale (List[str]): Column names to undergo cross-sectional Robust Scaling.
            log_features (Optional[List[str]]): Column names to undergo Symmetric Log transformation first.
        """
        self.features_to_scale = features_to_scale
        self.log_features = log_features if log_features else []

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CrossSectionalStationarityTransformer":
        """Fit the transformer.

        This transformer is cross-sectionally stateless across time. Assets within
        period T are evaluated solely against their peers within period T. Thus,
        fit only returns self.

        Args:
            X (pd.DataFrame): Input features dataframe.
            y (Optional[pd.Series]): Target values (ignored).

        Returns:
            CrossSectionalStationarityTransformer: The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features cross-sectionally to enforce stationarity.

        Applies a symmetric log transformation to highly skewed volumetric features,
        followed by a vectorized, group-wise Robust Scaling per quarter. Missing 
        values (NaNs) are natively propagated.

        Args:
            X (pd.DataFrame): Dataframe containing features and a 'date' column.

        Returns:
            pd.DataFrame: Transformed dataframe with normalized, stationary features.
        """
        df_out = X.copy()

        # 1. Symmetric Log transformation: sign(x) * log(1 + |x|)
        # Handles zeros and negative values safely without global shifts or data leakage.
        for col in self.log_features:
            df_out[col] = np.sign(df_out[col]) * np.log1p(np.abs(df_out[col]))

        # 2. Vectorized helper for localized Robust Scaling per quarter
        def fast_robust_scale(x: pd.Series) -> pd.Series:
            """Apply robust scaling on a single cross-sectional slice.

            Args:
                x (pd.Series): Feature values for a single date group.

            Returns:
                pd.Series: Scaled feature slice where NaNs are preserved.
            """
            clean_x = x.dropna()
            if len(clean_x) == 0:
                return x

            q75, q25 = np.percentile(clean_x, [75, 25])
            iqr = q75 - q25
            median = np.median(clean_x)

            # Fallback to mean de-meaning if the feature has no variance in the quarter
            if iqr == 0:
                return x - median

            # Arithmetic operations on the full Series automatically propagate NaNs
            return (x - median) / iqr

        # 3. Apply transformation group-wise to isolate quarters completely
        for col in self.features_to_scale:
            df_out[col] = df_out.groupby("date", group_keys=False)[col].transform(
                fast_robust_scale
            )
        return df_out


def calculate_cross_sectional_spearman(
    df: pd.DataFrame,
    target: str,
    pred: str,
    date_col: str = "date",
) -> float:
    """
    Calculate the mean cross-sectional Spearman correlation using optimized ranking.

    This function computes the cross-sectional correlation by first filtering out 
    periods that have 5 or fewer valid (non-NaN) target-prediction pairs. It then 
    simultaneously ranks the valid pairs in a vectorized manner (Cython) and 
    calculates the Pearson correlation on those ranks to avoid the high overhead 
    of pandas non-Pearson groupby correlations.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the date, target, and prediction columns.
    target : str
        The column name of the target variable.
    pred : str
        The column name of the predicted variable.
    date_col : str, default "date"
        The column name used to group the data.

    Returns
    -------
    float
        The average cross-sectional Spearman correlation, or np.nan if no
        valid periods exist.

    Raises
    ------
    KeyError
        If any of the specified columns (date_col, target, pred) are missing
        from the DataFrame.
    """
    for col in (date_col, target, pred):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    # Filtrowanie po liczbie valid pairs, aby uniknąć fałszywych korelacji dla grup z brakami
    valid_pairs = df[[date_col, target, pred]].dropna().groupby(date_col).size()
    valid_dates = valid_pairs[valid_pairs > 5].index

    if valid_dates.empty:
        return np.nan

    filtered = df.loc[df[date_col].isin(valid_dates), [date_col, target, pred]].copy()

    # Jednoczesne, wektorowe rangowanie obu kolumn w jednym wywołaniu groupby
    filtered[[target, pred]] = filtered.groupby(date_col)[[target, pred]].rank(method="average")

    # Obliczenie korelacji Pearsona na zrangowanych danych
    correlations = (
        filtered.groupby(date_col)[[target, pred]]
        .corr(method="pearson")
        .xs(target, level=1)[pred]
    )

    mean_corr = correlations.mean()
    return np.nan if pd.isna(mean_corr) else float(mean_corr)


def rank_gauss_transform(series: pd.Series) -> pd.Series:
    """
    Apply Rank Gauss transformation to a pandas Series using NumPy and SciPy.

    This method transforms the numerical values of a Series to a standard normal
    distribution. It converts the input to a float NumPy array to prevent dtype 
    coercion bugs (e.g., with integer Series), computes ranks using SciPy, and 
    applies the inverse cumulative distribution function (PPF).

    Parameters
    ----------
    series : pd.Series
        The input pandas Series to transform.

    Returns
    -------
    pd.Series
        The transformed Series with values following a standard normal distribution.
        Missing values (NaN), original indices, and names are preserved.
    """
    valid_mask = series.notna().values
    valid_count = int(valid_mask.sum())

    if valid_count < 10:
        return series.copy()

    arr = series.to_numpy(dtype=float, na_value=np.nan, copy=True)

    ranks = rankdata(arr[valid_mask], method="average")
    uniform = (ranks - 0.5) / valid_count
    arr[valid_mask] = norm.ppf(uniform)

    return pd.Series(arr, index=series.index, name=series.name)


def add_quarterly_vix(
    df: pd.DataFrame,
    date_col: str = "date",
    vix_col: str = "vix",
) -> pd.DataFrame:
    """
    Download VIX and attach last-of-quarter closing price to a quarterly DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a quarterly column (either strings like '2023Q1' or pd.Period).
    date_col : str, default "date"
        Column name containing the quarterly Period values.
    vix_col : str, default "vix"
        Name for the new VIX column in the output.

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with `vix_col`. A left join is used so missing
        quarters produce NaN rather than silently dropping rows.
    """
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in the DataFrame.")

    out_df = df.copy()

    # Upewnienie się, że mamy typ Period, co zapobiega błędom przy merge
    if not pd.api.types.is_period_dtype(out_df[date_col]):
        out_df[date_col] = pd.PeriodIndex(out_df[date_col], freq="Q")

    periods = out_df[date_col]
    start = periods.min().start_time.strftime("%Y-%m-%d")
    
    # max() + 1 tworzy bufor. Gwarantuje obejście "exclusive end date" w yfinance 
    # i pobranie faktycznie ostatniego dnia kwartału
    end = (periods.max() + 1).end_time.strftime("%Y-%m-%d")

    raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)

    if raw.empty or "Close" not in raw:
        out_df[vix_col] = float("nan")
        return out_df

    vix_close = raw["Close"]
    
    # Bezpieczniejsza alternatywa dla .squeeze(). Eliminuje ryzyko zredukowania 
    # DataFrame'u 1x1 do pojedynczego float'a (skalara).
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]

    vix_q = (
        vix_close
        .resample("QE")         
        .last()
        .to_period("Q")
        .rename(vix_col)
    )

    return out_df.merge(vix_q, left_on=date_col, right_index=True, how="left")


def calculate_cross_sectional_ndcg(
    df: pd.DataFrame,
    target: str,
    pred: str,
    date_col: str = "date",
    k: int = 10,
    n_random: int = 50,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Calculate the cross-sectional Normalized Discounted Cumulative Gain (nDCG).

    This function computes the mean nDCG@k for a given model's predictions and 
    compares it against a baseline of random predictions. Data is grouped by 
    a date column. Targets are converted to ranks to serve as strictly non-negative 
    relevance scores (required by scikit-learn).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the date, target, and prediction columns.
    target : str
        The column name of the target variable (ground truth).
    pred : str
        The column name of the predicted variable.
    date_col : str, default 'date'
        The column name used to group the data cross-sectionally.
    k : int, default 10
        The number of top predictions to consider for the nDCG metric.
    n_random : int, default 50
        The number of random baseline evaluations per group.
    random_state : int, default 42
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    tuple of (float, float)
        A tuple containing the mean model nDCG score and the mean random 
        baseline nDCG score. Returns (np.nan, np.nan) if no valid periods exist.
    """
    for col in (date_col, target, pred):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    model_scores, random_scores = [], []
    rng = np.random.default_rng(random_state)

    for _, group in df.groupby(date_col):
        valid_group = group[[target, pred]].dropna()
        n_items = len(valid_group)

        # nDCG wymaga przynajmniej 2 elementów, aby jakkolwiek oceniać ranking
        if n_items > 1:
            # sklearn's ndcg_score wymaga nieujemnych relevance scores.
            # Użycie ranks/pct rozwiązuje ten problem. Higher values = higher relevance.
            true_ranks = valid_group[target].rank(method="average").values.reshape(1, -1)
            pred_vals = valid_group[pred].values.reshape(1, -1)
            
            model_scores.append(ndcg_score(true_ranks, pred_vals, k=k))

            # np.broadcast_to tworzy "wirtualną" macierz powieloną wierszami (bez alokacji nowej pamięci)
            true_ranks_broadcast = np.broadcast_to(true_ranks, (n_random, n_items))
            
            # Generowanie macierzy losowych floatów
            random_preds = rng.random((n_random, n_items))
            
            # ndcg_score oblicza nDCG wiersz po wierszu i sam zwraca uśrednioną wartość
            avg_random_ndcg = ndcg_score(true_ranks_broadcast, random_preds, k=k)
            random_scores.append(avg_random_ndcg)

    if not model_scores:
        return np.nan, np.nan

    return float(np.mean(model_scores)), float(np.mean(random_scores))


def check_for_signal(
    df: pd.DataFrame,
    x: str,
    y: str,
    num_of_samples: int = 1000,
    clip_x_outliers: bool = False,
    clip_y_outliers: bool = False,
) -> None:
    """
    Evaluate and plot the linear relationship between a signal and a target.

    This function cleans the data (removes NaNs), optionally clips extreme values 
    assuming Z-scored data (keeps values between -3 and 3), samples the dataset 
    for plotting performance, and fits an OLS model without an intercept. 
    It displays a scatter plot with a regression line, along with slope and p-value.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the feature and target columns.
    x : str
        The column name of the independent variable (signal/feature).
    y : str
        The column name of the dependent variable (target).
    num_of_samples : int, default 1000
        Maximum number of data points to sample for the plot. If the dataset 
        has fewer valid rows, all available valid rows are used.
    clip_x_outliers : bool, default False
        If True, filters out `x` values outside the [-3, 3] range.
    clip_y_outliers : bool, default False
        If True, filters out `y` values outside the [-3, 3] range.

    Returns
    -------
    None
        Displays a seaborn scatter plot with an OLS regression line.
    """
    # Sprawdzenie, czy kolumny istnieją
    for col in (x, y):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    # 1. Wycięcie tylko niezbędnych kolumn i od razu usunięcie NaN
    df_clean = df[[x, y]].dropna()

    # 2. Usuwanie outlierów
    if clip_x_outliers:
        shape_before = df_clean.shape[0]
        df_clean = df_clean[df_clean[x].between(-3, 3)]
        removed_fraction = 100 * (shape_before - df_clean.shape[0]) / shape_before
        print(f"Usunąłem {removed_fraction:.2f}% zbioru dla X.")

    if clip_y_outliers:
        shape_before = df_clean.shape[0]
        df_clean = df_clean[df_clean[y].between(-3, 3)]
        removed_fraction = 100 * (shape_before - df_clean.shape[0]) / shape_before
        print(f"Usunąłem {removed_fraction:.2f}% zbioru dla Y.")

    n_available = len(df_clean)
    if n_available == 0:
        print("Brak danych do narysowania wykresu po usunięciu NaN i outlierów.")
        return

    # 3. Bezpieczne próbkownie (zabezpiecza przed ValueError gdy num_of_samples > n_available)
    sample_size = min(num_of_samples, n_available)
    df_sample = df_clean.sample(n=sample_size, random_state=42)

    # 4. Modelowanie OLS (bez wyrazu wolnego: ~ -1)
    # Uwaga: formula API może zawieść, jeśli nazwy kolumn zawierają spacje lub znaki specjalne.
    model = ols(data=df_sample, formula=f"{y} ~ -1 + {x}").fit()

    # 5. Rysowanie wykresu
    g = sns.lmplot(
        x=x,
        y=y,
        data=df_sample,
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"},
    )

    # Bezpieczne pobranie osi bezpośrednio z obiektu seaborn
    ax = g.ax

    # 6. Dodanie tekstu na wykresie
    ax.text(
        0.05,
        0.95,
        f"Nachylenie: {model.params.iloc[0]:.4f}\nP-value: {model.pvalues.iloc[0]:.4f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.show()
    
#%% live prediction
# Maksymalny dopuszczalny "wiek" najnowszego raportu wzgledem konca kwartalu T-1.    
MAX_STALENESS_DAYS = 200
TTM_QUARTERS = 4
# Walidacja ciaglosci kwartalow (kwartaly fiskalne maja 13-14 tygodni)
MIN_QUARTER_GAP_DAYS = 70
MAX_QUARTER_GAP_DAYS = 120


def parse_quarter(quarter: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Parse a quarter string to obtain the normalized boundary dates for T and T-1 quarters.
    
    Parameters
    ----------
    quarter : str
        The quarter representation (e.g., '2026Q1', '2025Q4').
    
    Returns
    -------
    tuple of (pd.Timestamp, pd.Timestamp)
        A tuple containing:
        - The normalized end date of the target quarter T (at 00:00:00).
        - The normalized end date of the preceding quarter T-1 (at 00:00:00).
    """
    period = pd.Period(quarter, freq="Q-DEC")
    t_end = period.end_time.normalize()
    t_minus_1_end = (period - 1).end_time.normalize()
    return t_end, t_minus_1_end
 
 
def _strip_tz(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx
 
 
def pick_ttm_columns(columns, cutoff: pd.Timestamp,
                     n: int = TTM_QUARTERS) -> pd.DatetimeIndex | None:
    """
    Select the contiguous TTM quarters ending on or before the cutoff date.
    
    Parameters
    ----------
    columns : array-like
        The list of available dates (columns) from the financial statement.
    cutoff : pd.Timestamp
        The latest acceptable date for the quarters (point-in-time boundary).
    n : int, default TTM_QUARTERS
        The number of consecutive quarters required to build a TTM window.
    
    Returns
    -------
    pd.DatetimeIndex or None
        The selected chronological list of n quarter dates (ascending), 
        or `None` if the dates are insufficient or do not form a consistent 
        consecutive window.
    """
    cols = _strip_tz(pd.Index(columns)).sort_values()
    eligible = cols[cols <= cutoff]
    if len(eligible) < n:
        return None
    sel = eligible[-n:]
    gaps = np.diff(sel.values).astype("timedelta64[D]").astype(int)
    if any(g < MIN_QUARTER_GAP_DAYS or g > MAX_QUARTER_GAP_DAYS for g in gaps):
        logger.warning("Niespojne okno TTM (odstepy w dniach: %s) dla dat %s",
                       list(gaps), [str(d.date()) for d in sel])
        return None
    return sel
 
 
def _get_line_item(stmt: pd.DataFrame, col: pd.Timestamp, names: list[str]):
    """
    Sum a specific financial statement item over a TTM (trailing twelve months) window.
    
    Parameters
    ----------
    stmt : pd.DataFrame
        The financial statement (e.g., income statement) from yfinance.
    cols : pd.DatetimeIndex
        The quarterly column timestamps representing the TTM window.
    names : list of str
        Possible candidate names for the line item in the statement.
    ticker : str
        The stock ticker symbol, used for logging purposes.
    label : str
        A user-friendly label of the metric being fetched (e.g., 'revenue').
    
    Returns
    -------
    float
        The TTM sum of the specified item, or `np.nan` if any quarter is missing 
        or invalid, or if `cols` is empty.
    """
    if stmt is None or stmt.empty:
        return np.nan
    col_map = {pd.Timestamp(c).tz_localize(None) if pd.Timestamp(c).tz else pd.Timestamp(c): c
               for c in stmt.columns}
    real_col = col_map.get(col)
    if real_col is None:
        return np.nan
    for name in names:
        if name in stmt.index:
            val = stmt.loc[name, real_col]
            if pd.notna(val):
                return float(val)
    return np.nan
 
 
def _ttm_sum(stmt: pd.DataFrame, cols: pd.DatetimeIndex, names: list[str], ticker: str, label: str):
    """
    Calculate the Trailing Twelve Months (TTM) sum for a specific financial line item.
    
    Parameters
    ----------
    stmt : pd.DataFrame
        Financial statement (income or balance sheet) where rows are line items 
        and columns are report dates.
    cols : pd.DatetimeIndex
        The specific dates (usually 4 quarters) to be summed.
    names : list of str
        Potential labels/keys for the line item in the DataFrame (to handle variations).
    ticker : str
        Stock ticker symbol for logging purposes.
    label : str
        Human-readable name of the line item (e.g., 'revenue') for logging.
    
    Returns
    -------
    float
        The TTM sum of the requested item. Returns `np.nan` if any period is 
        missing or non-finite.
    """
    vals = [_get_line_item(stmt, c, names) for c in cols]
    if any(not np.isfinite(v) for v in vals):
        missing = [str(c.date()) for c, v in zip(cols, vals) if not np.isfinite(v)]
        logger.warning("%s: brak '%s' dla kwartalow %s - TTM = NaN", ticker, label, missing)
        return np.nan
    return float(np.sum(vals))
 

def get_price_asof(tk: yf.Ticker, asof: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    """
    Retrieve the adjusted close price on or prior to a given date.
    
    Parameters
    ----------
    tk : yf.Ticker
        The yfinance Ticker object for the target equity.
    asof : pd.Timestamp
        The target reference date.
    
    Returns
    -------
    tuple of (float, pd.Timestamp or None)
        A tuple containing:
        - The adjusted close price (float), or `np.nan` if no data is found.
        - The actual trading date associated with the price (pd.Timestamp), 
          or `None` if no data is found.
    """
    hist = tk.history(
        start=(asof - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
        end=(asof + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
    )
    if hist is None or hist.empty:
        return np.nan, None
    hist.index = _strip_tz(hist.index)
    hist = hist[hist.index <= asof]
    if hist.empty:
        return np.nan, None
    return float(hist["Close"].iloc[-1]), hist.index[-1]
 
 
def fetch_ticker_features(ticker: str, quarter: str) -> dict:
    """
    Extract and calculate financial features for a single stock ticker.
    
    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g., 'AAPL').
    quarter : str
        The target quarter representation (e.g., '2026Q1').
    
    Returns
    -------
    dict
        A dictionary containing the parsed fundamentals, period metadata, 
        and calculated financial ratios (e.g., P/S, ROE).
    """
    t_end, t_minus_1_end = parse_quarter(quarter)
    tk = yf.Ticker(ticker)
 
    row = {
        "ticker": ticker,
        "quarter": quarter,
        "price_date": None,
        "adj_close": np.nan,
        "fundamentals_period_end": None,
        "ttm_window": None,
        "revenue_ttm": np.nan,
        "net_income_ttm": np.nan,
        "weighted_average_shares": np.nan,
        "total_shareholder_equity": np.nan,
        "revenue_per_share": np.nan,
        "price_to_sales": np.nan,
        "roe": np.nan,
    }

    try:
        adj_close, price_date = get_price_asof(tk, t_end)
        row["adj_close"] = adj_close
        row["price_date"] = price_date
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: blad pobierania ceny: %s", ticker, exc)
 
    try:
        inc = tk.quarterly_income_stmt
        bs = tk.quarterly_balance_sheet
 
        ttm_cols = (pick_ttm_columns(inc.columns, t_minus_1_end)
                    if inc is not None and not inc.empty else None)
        bs_col_idx = (_strip_tz(pd.Index(bs.columns)).sort_values()
                      if bs is not None and not bs.empty else pd.DatetimeIndex([]))

        bs_cutoff = ttm_cols[-1] if ttm_cols is not None else t_minus_1_end
        bs_eligible = bs_col_idx[bs_col_idx <= bs_cutoff]
        bs_col = bs_eligible.max() if len(bs_eligible) else None
 
        if ttm_cols is None and bs_col is None:
            logger.warning("%s: brak wystarczajacych sprawozdan <= %s",
                           ticker, t_minus_1_end.date())
            return row

        newest_inc = ttm_cols[-1] if ttm_cols is not None else None
        for label, col in (("income", newest_inc), ("balance", bs_col)):
            if col is not None and (t_minus_1_end - col).days > MAX_STALENESS_DAYS:
                logger.warning(
                    "%s: raport %s z %s jest starszy niz %d dni wzgledem %s - pomijam",
                    ticker, label, col.date(), MAX_STALENESS_DAYS, t_minus_1_end.date(),
                )
                if label == "income":
                    ttm_cols = None
                    newest_inc = None
                else:
                    bs_col = None
 
        if ttm_cols is not None:
            row["fundamentals_period_end"] = newest_inc
            row["ttm_window"] = [str(c.date()) for c in ttm_cols]

            row["revenue_ttm"] = _ttm_sum(
                inc, ttm_cols, ["Total Revenue", "Operating Revenue"], ticker, "revenue")
            row["net_income_ttm"] = _ttm_sum(
                inc, ttm_cols, ["Net Income", "Net Income Common Stockholders"], ticker, "net_income")

            for c in reversed(list(ttm_cols)):
                sh = _get_line_item(inc, c, ["Basic Average Shares", "Diluted Average Shares"])
                if np.isfinite(sh):
                    if c != newest_inc:
                        logger.warning("%s: brak shares w %s - uzywam %s",
                                       ticker, newest_inc.date(), c.date())
                    row["weighted_average_shares"] = sh
                    break
 
        if bs_col is not None:
            if row["fundamentals_period_end"] is None:
                row["fundamentals_period_end"] = bs_col
            row["total_shareholder_equity"] = _get_line_item(
                bs, bs_col, ["Stockholders Equity", "Common Stock Equity"])
            if newest_inc is not None and newest_inc != bs_col:
                logger.warning("%s: rozne okresy: income=%s, balance=%s",
                               ticker, newest_inc.date(), bs_col.date())
    except Exception as exc:
        logger.warning("%s: blad pobierania fundamentow: %s", ticker, exc)
        return row
 
    row["revenue_per_share"] = safe_divide(row["revenue_ttm"], row["weighted_average_shares"])
    row["price_to_sales"] = safe_divide(row["adj_close"], row["revenue_per_share"])
    row["roe"] = safe_divide(row["net_income_ttm"], row["total_shareholder_equity"])
    return row
 
 
def fetch_live_features(quarter: str, tickers: list[str], pause_s: float = 0.5, verbose=False) -> pd.DataFrame:
    """
    Fetch and aggregate live features for a list of tickers in a given quarter.
    
    Parameters
    ----------
    quarter : str
        The target quarter for which features are retrieved (e.g., '2026Q1').
    tickers : list of str
        List of stock ticker symbols to process.
    pause_s : float, default 0.5
        Time in seconds to pause between API requests to mitigate rate limiting.
    verbose : bool, default False
        If True, prints progress updates to the standard output.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the fetched features, indexed by ticker.
    
    Raises
    ------
    KeyError
        If the underlying feature-fetching function does not return a 'ticker' key,
        or if the input ticker list is empty.
    """
    rows = []
    for i, ticker in enumerate(tickers):
        if verbose:
            print(f"[{i+1}/{len(tickers)}] Processing {ticker}...")
        rows.append(fetch_ticker_features(ticker, quarter))
        if pause_s and i < len(tickers) - 1:
            time.sleep(pause_s)  # zmniejsza ryzyko throttlingu Yahoo
    return pd.DataFrame(rows).set_index("ticker")
 