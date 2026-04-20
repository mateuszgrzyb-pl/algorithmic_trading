import logging
import os
from typing import List, Optional

from financetoolkit import Toolkit

from app.utils.tools import standardize_column_names, ensure_directory


logger = logging.getLogger(__name__)


def download_price_history(
    tickers: List[str], start_date: str, end_date: str, api_key: str
) -> None:
    """
    Download historical price data for given tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: FinanceToolkit API key
    """

    for ticker in tickers:
        output_path = f"data/raw/price_history/STAGE_1/{ticker}.feather"

        try:
            logger.debug(f"Downloading price history for {ticker}")
            toolkit = Toolkit([ticker], api_key=api_key, start_date=start_date, end_date=end_date, progress_bar=False)
            data = toolkit.get_historical_data(
                period="daily",
                return_column="Adj Close"
            )

            if data.empty:
                logger.warning(f"No price data returned for {ticker}")
                continue

            ensure_directory(os.path.dirname(output_path))
            data.reset_index(inplace=True)

            columns_to_select = [("date", ""), ("Adj Close", ticker)]

            available_columns = data.columns.tolist()
            if len(available_columns) < 2:
                logger.error(
                    f"Unexpected data structure for {ticker}: {available_columns}"
                )
                continue

            data = data.loc[:, columns_to_select]
            data.columns = ["date", "adj_close"]

            if data.isnull().all().any():
                logger.warning(f"Price data for {ticker} contains all null values")

            data.to_feather(output_path)
            logger.info(f"Price history for {ticker} saved: {len(data)} records")

        except Exception as e:
            error_msg = f"Failed to download price history for {ticker}: {str(e)}"
            logger.error(error_msg)


def download_balance_sheets(
    tickers: List[str], start_date: str, end_date: str, api_key: str
) -> None:
    """
    Download balance sheet data for given tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: FinanceToolkit API key
    """
    for ticker in tickers:
        output_path = f"data/raw/balance_sheets/{ticker}.feather"

        try:
            logger.debug(f"Downloading balance sheet for {ticker}")
            toolkit = Toolkit([ticker], api_key=api_key, start_date=start_date, end_date=end_date, quarterly=True, progress_bar=False)
            data = toolkit.get_balance_sheet_statement()

            if data.empty:
                logger.warning(f"No balance sheet data returned for {ticker}")
                continue

            ensure_directory(os.path.dirname(output_path))
            data.columns = data.columns.astype(str)
            data = data.transpose().reset_index()
            data.columns = standardize_column_names(data.columns)

            data.to_feather(output_path)
            logger.info(f"Balance sheet for {ticker} saved: {len(data)} records")

        except Exception as e:
            error_msg = f"Failed to download balance sheet for {ticker}: {str(e)}"
            logger.error(error_msg)


def download_income_statements(
    tickers: List[str], start_date: str, end_date: str, api_key: str
) -> None:
    """
    Download income statement data for given tickers.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: FinanceToolkit API key
    """
    for ticker in tickers:
        output_path = f"data/raw/income_statements/{ticker}.feather"

        try:
            logger.debug(f"Downloading income statement for {ticker}")
            toolkit = Toolkit([ticker], api_key=api_key, start_date=start_date, end_date=end_date, quarterly=True, progress_bar=False)
            data = toolkit.get_income_statement()

            if data.empty:
                logger.warning(f"No income statement data returned for {ticker}")
                continue

            ensure_directory(os.path.dirname(output_path))
            data.columns = data.columns.astype(str)
            data = data.transpose().reset_index()
            data.columns = standardize_column_names(data.columns)

            data.to_feather(output_path)
            logger.info(f"Income statement for {ticker} saved: {len(data)} records")

        except Exception as e:
            error_msg = f"Failed to download income statement for {ticker}: {str(e)}"
            logger.error(error_msg)
