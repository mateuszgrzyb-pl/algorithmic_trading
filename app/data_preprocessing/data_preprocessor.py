import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from app.utils.tools import get_available_tickers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(
    ticker: str, base_path: Path | str = "data/raw"
) -> Dict[str, pd.DataFrame]:
    """
    The function attempts to read up to four data types (price history,
    balance sheets, income statements, company profiles) from feather
    files located under the given base path. Loaded DataFrames are returned
    in a dictionary keyed by data type.

    Args:
        ticker (str): Ticker symbol (e.g., "AAPL").
        base_path (Path | str, optional): Base directory containing raw data
            subfolders. Defaults to "data/raw".

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping data type names to
        corresponding DataFrames that were successfully loaded.

    Raises:
        FileNotFoundError: If no files are found for the given ticker.
        Exception: If an unexpected error occurs while reading a file.
    """
    base_path = Path(base_path)
    data = {}

    paths = {
        "price_history": base_path / "price_history" / "STAGE_4" / f"{ticker}.feather",
        "balance_sheets": base_path / "balance_sheets" / f"{ticker}.feather",
        "income_statements": base_path / "income_statements" / f"{ticker}.feather",
        "company_profiles": base_path / "company_profiles" / f"{ticker}.feather",
    }

    loaded_count = 0
    for key, path in paths.items():
        if os.path.exists(path):
            try:
                df = pd.read_feather(path)
                if "date" in df.columns:
                    df["date"] = pd.PeriodIndex(df["date"], freq="Q-DEC")
                data[key] = df
                loaded_count += 1
                logger.debug(f"Loaded {key} for {ticker}: {len(df)} records")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {str(e)}")
        else:
            logger.debug(f"File {path} does not exist.")
    if loaded_count == 0:
        error_msg = f"Could not find any data for ticker: {ticker}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Successfully loaded {loaded_count}/4 data types for {ticker}")
    return data


def merge_data(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merges different data sources from a dictionary into a single DataFrame.

    This function uses 'price_history' as the base DataFrame. It then performs
    an inner join with quarterly reports ('balance_sheets', 'income_statements')
    on the 'date' column. A one-period offset is applied to the dates of
    quarterly reports to align them with the price data. Finally, it adds
    company profile information as new columns, broadcasting the values to all rows.

    Args:
        data (Dict[str, pd.DataFrame]): A dictionary of DataFrames, typically
            the output of the `load_data` function. Must contain the key
            'price_history'.

    Returns:
        pd.DataFrame: A single, merged DataFrame containing all provided data.

    Raises:
        KeyError: If the 'price_history' key is not found in the input dictionary.
        ValueError: If 'price_history' DataFrame is empty.
    """
    if "price_history" not in data:
        raise KeyError("Input data must contain 'price_history' to serve as a base.")
    if data["price_history"].empty:
        raise ValueError("'price_history' DataFrame cannot be empty.")

    merged_df = data["price_history"].copy()
    logger.debug(f"Starting merge with 'price_history', shape: {merged_df.shape}")

    quarterly_reports = ["balance_sheets", "income_statements"]

    for report_key in quarterly_reports:
        if report_key in data:
            report_df = data[report_key].copy()
            report_df["date"] = report_df["date"] + 1
            merged_df = merged_df.merge(report_df, on="date", how="inner")
            logger.debug(f"Merged with {report_key}, new shape: {merged_df.shape}")

    if "company_profiles" in data:
        profile_df = data["company_profiles"]
        if not profile_df.empty:
            profile_data = profile_df.iloc[0]
            for col, value in profile_data.items():
                merged_df[col] = value
            logger.debug(f"Added {len(profile_data)} columns from company profile.")

    logger.info(f"Final merged DataFrame has shape: {merged_df.shape}")
    return merged_df


def save_processed_data(
    processed_df: pd.DataFrame,
    ticker: str,
    base_path: Path | str = "data/processed",
) -> None:
    """
    Saves the processed DataFrame to a Feather file within a specified directory.

    This function constructs the full output path from the base path and the ticker.
    It automatically creates the destination directory if it does not exist.

    Args:
        processed_df (pd.DataFrame): The DataFrame containing processed data to save.
        ticker (str): The ticker symbol, used to name the output file (e.g., "AAPL").
        base_path (Path | str, optional): The base directory where the file will be
            saved. Defaults to "data/processed".

    Raises:
        IOError: If there is a problem writing the file to the disk (e.g.,
            permission errors, disk full).
    """
    try:
        base_path = Path(base_path)
        output_path = base_path / f"{ticker}.feather"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        processed_df.to_feather(output_path)
        logger.info(f"Processed data for {ticker} saved successfully to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save data for {ticker} to {output_path}: {e}")
        raise IOError(f"Could not save file for {ticker}") from e


def create_abt(
    target: str,
    input_path: Path | str = "data/processed",
    output_path: Path | str = "data/abt",
) -> None:
    """
    Creates an Analytical Base Table (ABT) for a specified target label.

    This function reads all processed ticker data from the input directory,
    combines them into a single DataFrame, filters out irrelevant label columns
    and other unnecessary columns, and saves the final ABT to a Feather file.

    Args:
        target (str): The substring used to identify the target label column to keep
            (e.g., "1M" for "label_1M"). All other label columns will be dropped.
        input_path (Path | str, optional): Directory containing the processed
            .feather files for each ticker. Defaults to "data/processed".
        output_path (Path | str, optional): Directory where the final ABT
            .feather file will be saved. Defaults to "data/abt".

    Raises:
        FileNotFoundError: If no processed ticker files are found in the input path.
        IOError: If there is a problem writing the final ABT file to disk.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    logger.info(f"Starting ABT creation for target: '{target}'")
    logger.debug(f"Reading processed data from: {input_path}")

    tickers = get_available_tickers(str(input_path))
    if not tickers:
        msg = f"No processed data files found in {input_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Found {len(tickers)} tickers to process.")

    all_dataframes = []
    for ticker in tickers:
        try:
            file_path = input_path / f"{ticker}.feather"
            df = pd.read_feather(file_path)
            df["ticker"] = ticker
            all_dataframes.append(df)
        except Exception as e:
            logger.warning(f"Could not read or process file for ticker {ticker}: {e}")

    if not all_dataframes:
        msg = "Failed to read any ticker dataframes. ABT creation aborted."
        logger.error(msg)
        raise ValueError(msg)

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(
        f"Combined data into a single DataFrame with shape: {combined_df.shape}"
    )

    all_labels = [col for col in combined_df.columns if "label" in col]
    labels_to_drop = [label for label in all_labels if target not in label]

    columns_to_drop = labels_to_drop
    if "company_name" in combined_df.columns:
        columns_to_drop.append("company_name")

    abt_df = combined_df.drop(columns=columns_to_drop)
    logger.info(f"Removed {len(columns_to_drop)} columns. ABT shape: {abt_df.shape}")
    logger.debug(f"Columns removed: {columns_to_drop}")

    try:
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{target}.feather"
        abt_df.to_feather(output_file)
        logger.info(f"ABT for target '{target}' saved successfully to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save final ABT file to {output_file}: {e}")
        raise IOError("Could not save the ABT file.") from e


def deduplicate_price_data(
    ticker: str,
    input_dir: Path | str = "data/raw/price_history/STAGE_2",
    output_dir: Path | str = "data/raw/price_history/STAGE_3",
) -> None:
    input_path = Path(input_dir) / f"{ticker}.feather"
    output_path = Path(output_dir) / f"{ticker}.feather"
    logger.info(f"Deduplicating price data for {ticker} from {input_path}")

    try:
        df = pd.read_feather(input_path)
        logger.debug(f"Loaded raw price data with shape: {df.shape}")

        # 1. Bezpieczna konwersja na Timestamp
        if hasattr(df["date"].dt, 'to_timestamp'):
            df["date_ts"] = df["date"].dt.to_timestamp()
        else:
            df["date_ts"] = pd.to_datetime(df["date"])

        # 2. Sortowanie i wyznaczenie kwartału
        df = df.sort_values("date_ts").reset_index(drop=True)
        df["quarter"] = df["date_ts"].dt.to_period("Q")

        # 3. Deduplikacja - zostawiamy ostatni dzień kwartału
        # WAŻNE: Nie usuwamy kolumny 'quarter' jeszcze!
        deduplicated_df = (
            df.drop_duplicates(subset="quarter", keep="last")
            .reset_index(drop=True)
        )

        # 4. Przypisujemy kwartał do kolumny date
        # Robimy to bezpośrednio, bez użycia .dt.to_period()
        deduplicated_df["date"] = deduplicated_df["quarter"]

        # 5. Teraz usuwamy zbędne kolumny pomocnicze
        deduplicated_df = deduplicated_df.drop(columns=["quarter", "date_ts"])

        logger.debug(f"Deduplicated data shape: {deduplicated_df.shape}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        deduplicated_df.to_feather(output_path)
        logger.info(f"Successfully saved cleaned data for {ticker} to {output_path}")

    except FileNotFoundError:
        logger.error(f"Input file not found for ticker {ticker} at {input_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {ticker}: {e}")
        raise IOError(f"Failed to process or save data for {ticker}") from e


def remove_overlapped_observations(
    ticker: str,
    offset: int,
    label_time: int,
    input_dir: Path | str = "data/raw/price_history/STAGE_3",
    output_dir: Path | str = "data/raw/price_history/STAGE_4",
) -> None:
    """
    Removes overlapping observations to ensure target variables are independent.

    This function downsamples a time-series DataFrame by selecting rows at
    regular intervals. It's a crucial step to prevent data leakage when
    creating target variables that look forward in time (e.g., "what will the
    price be in 3 months?").

    Args:
        ticker (str): The ticker symbol to process (e.g., "AAPL").
        offset (int): The starting index for the selection (e.g., 0 to start
            from the beginning).
        label_time (int): The step or interval for selection. For a 3-month
            label, this would typically be 3, selecting every 3rd observation.
        input_dir (Path | str, optional): The directory containing the input
            .feather files. Defaults to "data/raw/price_history/STAGE_3".
        output_dir (Path | str, optional): The directory where the final
            .feather file will be saved. Defaults to "data/raw/price_history/STAGE_4".

    Raises:
        FileNotFoundError: If the input file for the ticker does not exist.
        IOError: If there is a problem writing the output file to disk.
    """
    input_path = Path(input_dir) / f"{ticker}.feather"
    output_path = Path(output_dir) / f"{ticker}.feather"
    logger.info(
        f"Downsampling data for {ticker} with offset={offset}, step={label_time}"
    )

    try:
        df = pd.read_feather(input_path)

        logger.debug(f"Loaded data for {ticker} with shape: {df.shape}")

        downsampled_df = df.iloc[offset::label_time].reset_index(drop=True)

        if downsampled_df.empty:
            logger.warning(
                f"Downsampling for {ticker} resulted in an empty DataFrame. "
                f"Check offset ({offset}) and label_time ({label_time}) "
                f"against data length ({len(df)})."
            )

        logger.debug(f"Downsampled data shape: {downsampled_df.shape}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        downsampled_df.to_feather(output_path)
        logger.info(
            f"Successfully saved downsampled data for {ticker} to {output_path}"
        )

    except FileNotFoundError:
        logger.error(f"Input file not found for ticker {ticker} at {input_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {ticker}: {e}")
        raise IOError(f"Failed to process or save data for {ticker}") from e
