"""
Tools used in data preparation.
"""
import logging
from pathlib import Path
from typing import List, Set, Any, Dict, Union, Optional

import numpy as np
import pandas as pd
from pyxirr import xirr
import yfinance as yf


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
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return np.nan
        return numerator / denominator
    except:
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
        
        if hist.empty or "Close" not in hist.columns:
            return np.nan
            
        return hist["Close"].iloc[-1]
    except Exception as e:
        print(f"  ⚠️  Errot. Could not download price data for ticker: {ticker}: {e}")
        return np.nan


def get_financial_value(df: pd.DataFrame, field: str, date: str, ticker: str = "") -> float:
    """Safely retrieve a single value from a financial DataFrame.

    Looks up the given field (row) and date (column) in a financial statement
    DataFrame. If the exact date is not found, falls back to the first available
    column. Returns np.nan for any missing, null, or otherwise unresolvable value.

    Args:
        df (pd.DataFrame): Financial statement DataFrame with fields as the index
            and period dates as columns (e.g. from FinanceToolkit).
        field (str): Row label to retrieve (e.g. 'Total Revenue').
        date (str): Column label representing the reporting period (e.g. '2023-12-31').
        ticker (str): Stock ticker symbol, used for logging purposes. Defaults to ''.

    Returns:
        float: The retrieved value cast to float, or np.nan if the value is
            missing, null, or an exception occurs.
    """
    try:
        if df is None or df.empty:
            return np.nan
        
        if field not in df.index:
            return np.nan
            
        if date not in df.columns:
            # Try to find the closest available date
            available_dates = df.columns.tolist()
            if not available_dates:
                return np.nan
            # Fall back to the first available date
            date = available_dates[0]
            
        value = df.loc[field, date]
        return float(value) if not pd.isna(value) else np.nan
        
    except Exception as e:
        return np.nan


def create_features_per_ticker(ticker: str, date: str, verbose: bool = True) -> pd.DataFrame:
    """Compute fundamental financial features for a given ticker and quarter.

    Retrieves the quarter-end closing price and financial statement data via
    yfinance, then derives a set of valuation metrics. If the price is unavailable
    or a critical error occurs, returns a single-row DataFrame filled with np.nan.

    Args:
        ticker (str): Stock ticker symbol (e.g. 'AAPL').
        date (str): Reporting quarter in pandas Period format (e.g. '2024Q1').
        verbose (bool): Whether to print warnings on non-critical failures.
            Defaults to True.

    Returns:
        pd.DataFrame: Single-row DataFrame with the following columns:
            - ticker: Stock ticker symbol.
            - date: Reporting quarter.
            - adj_price: Quarter-end closing price.
            - graham_number_vs_price: Graham Number divided by price.
            - eps: Basic earnings per share.
            - price_to_sales: Price divided by revenue per share.
            - book_value_per_share: Stockholders equity divided by shares outstanding.
            - price_to_earnings: Price divided by EPS.
            - market_cap: Price multiplied by shares outstanding.
            - price_to_book: Price divided by book value per share.
            All numeric columns are set to np.nan if data is unavailable.
    """
    # Default result returned on unrecoverable failure
    default_result = pd.DataFrame({
        'ticker': [ticker],
        'date': [date],
        'adj_price': [np.nan],
        'graham_number_vs_price': [np.nan],
        'eps': [np.nan],
        'price_to_sales': [np.nan],
        'book_value_per_share': [np.nan],
        'price_to_earnings': [np.nan],
        'market_cap': [np.nan],
        'price_to_book': [np.nan]
    })
    
    try:
        # Fetch quarter-end closing price
        adj_price = get_quarter_price(ticker, date)
        if pd.isna(adj_price):
            if verbose:
                print(f"  ⚠️  No closing price for {ticker} w {date}")
            return default_result
        
        # Derive the last calendar date of the previous quarter
        try:
            date_long = str((pd.Period(date) - 1).end_time.date())
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error during parsing data for {ticker}: {e}")
            return default_result
        
        # Fetch quarterly financial statements
        try:
            ticker_obj = yf.Ticker(ticker)
            balance_sheet = ticker_obj.quarterly_balance_sheet
            financials = ticker_obj.quarterly_financials
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error during downloading data for {ticker}: {e}")
            return default_result
        
        # Extract line items from financial statements
        total_shareholder_equity = get_financial_value(
            balance_sheet, "Stockholders Equity", date_long, ticker
        )
        weighted_average_shares = get_financial_value(
            financials, "Basic Average Shares", date_long, ticker
        )
        revenue = get_financial_value(
            financials, "Total Revenue", date_long, ticker
        )
        eps = get_financial_value(
            financials, "Basic EPS", date_long, ticker
        )
        
        # Derive per-share metrics
        book_value_per_share = safe_divide(
            total_shareholder_equity, weighted_average_shares
        )
        revenue_per_share = safe_divide(revenue, weighted_average_shares)
        
        # Graham Number: sqrt(22.5 * EPS * BVPS); undefined for non-positive products
        try:
            eps_bvps_product = eps * book_value_per_share
            if pd.isna(eps_bvps_product) or eps_bvps_product <= 0:
                graham_number = np.nan
            else:
                graham_number = np.sqrt(22.5 * eps_bvps_product)
        except:
            graham_number = np.nan
        
        # Compute valuation ratios
        graham_number_vs_price = safe_divide(graham_number, adj_price)
        price_to_sales = safe_divide(adj_price, revenue_per_share)
        price_to_earnings = safe_divide(adj_price, eps)
        
        try:
            market_cap = adj_price * weighted_average_shares
            if pd.isna(market_cap):
                market_cap = np.nan
        except:
            market_cap = np.nan
            
        price_to_book = safe_divide(adj_price, book_value_per_share)
        
        return pd.DataFrame({
            'ticker': [ticker],
            'date': [date],
            'adj_price': [adj_price],
            'graham_number_vs_price': [graham_number_vs_price],
            'eps': [eps],
            'price_to_sales': [price_to_sales],
            'book_value_per_share': [book_value_per_share],
            'price_to_earnings': [price_to_earnings],
            'market_cap': [market_cap],
            'price_to_book': [price_to_book]
        })
        
    except Exception as e:
        if verbose:
            print(f"  ❌ Unexpected error for {ticker} on {date}: {e}")
        return default_result


def process_multiple_tickers(tickers: list, date: str, verbose: bool = True) -> pd.DataFrame:
    """Process a list of tickers and concatenate their feature DataFrames.

    Iterates over the provided tickers, calls create_features_per_ticker for
    each, and collects the results. If a ticker raises an unexpected exception,
    a fallback row filled with np.nan is appended so that no ticker is silently
    dropped from the output.

    Args:
        tickers (list): List of stock ticker symbols (e.g. ['AAPL', 'MSFT']).
        date (str): Reporting quarter in pandas Period format (e.g. '2024Q1').
        verbose (bool): Whether to print progress and error messages.
            Defaults to True.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all per-ticker feature rows,
            with a reset index. Returns an empty DataFrame if the input list
            is empty or all calls fail before appending any results.
    """
    results = []
    
    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"[{i}/{len(tickers)}] Processing {ticker}...")
        
        try:
            df = create_features_per_ticker(ticker, date, verbose=verbose)
            results.append(df)
        except Exception as e:
            # Append a blank row so the ticker is still represented in the output
            if verbose:
                print(f"  ❌ Critical error for {ticker}: {e}")
            results.append(pd.DataFrame({
                'ticker': [ticker],
                'date': [date],
                'adj_price': [np.nan],
                'graham_number_vs_price': [np.nan],
                'eps': [np.nan],
                'price_to_sales': [np.nan],
                'book_value_per_share': [np.nan],
                'price_to_earnings': [np.nan],
                'market_cap': [np.nan],
                'price_to_book': [np.nan]
            }))
    
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)


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
