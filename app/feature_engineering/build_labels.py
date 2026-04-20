import logging
from typing import List
import pandas as pd

from app.config import settings
from app.feature_engineering.labeling import triple_barrier_labeling_custom

logger = logging.getLogger(__name__)


def build_triple_barier_labels_custom(
    ticker: str,
    profit_targets: List[float],
    stop_losses: List[float],
    max_days: List[int],
    overwrite: bool = False,
) -> None:
    """
    Generates and saves triple barrier labels for a given stock ticker.

    This function reads historical price data, iterates through a combination of
    profit targets, stop losses, and time horizons (max_days), and applies the
    triple barrier labeling method for each combination. The resulting DataFrame,
    enriched with multiple label sets, is saved to a new location.

    Args:
        ticker (str): The stock ticker symbol to process (e.g., 'AAPL').
        profit_targets (List[float]): A list of profit-take percentages.
        stop_losses (List[float]): A list of stop-loss percentages.
        max_days (List[int]): A list of maximum holding periods (vertical barrier).
        overwrite (bool): If True, existing label files will be overwritten.
                          If False, the function will skip processing if the
                          output file already exists. Defaults to False.

    Returns:
        None. The function saves the output to a file.

    Raises:
        FileNotFoundError: If the input price history file for the ticker does not exist.
        ValueError: If the lengths of profit_targets, stop_losses, and max_days
                    are not equal.
    """
    if not (len(profit_targets) == len(stop_losses) == len(max_days)):
        raise ValueError(
            "Input lists (profit_targets, stop_losses, max_days) must have the same length."
        )

    input_path = (
        settings.base_path
        / "data"
        / "raw"
        / "price_history"
        / "STAGE_1"
        / f"{ticker}.feather"
    )
    output_path = (
        settings.base_path
        / "data"
        / "raw"
        / "price_history"
        / "STAGE_2"
        / f"{ticker}.feather"
    )

    if not overwrite and output_path.exists():
        logger.info(
            f"Labels for {ticker} already exist. Skipping. Use overwrite=True to regenerate."
        )
        return

    if not input_path.exists():
        logger.error(
            f"Input price data for {ticker} not found at {input_path}. Skipping."
        )
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(
        f"Processing {ticker}: Generating {len(profit_targets)} sets of labels."
    )

    try:
        data = pd.read_feather(input_path)

        for pt, sl, md in zip(profit_targets, stop_losses, max_days):
            label_prefix = f"label_{pt}_{sl}_{md}"
            logger.debug(f"Generating labels for {ticker} with prefix: {label_prefix}")

            data = triple_barrier_labeling_custom(
                df=data,
                price_col="adj_close",
                label_name=label_prefix,
                date_col="date",
                profit_target=pt,
                stop_loss=sl,
                max_days=md,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_feather(output_path)
        logger.info(f"Successfully saved labeled data for {ticker} to {output_path}")

    except Exception as e:
        logger.error(
            f"An unexpected error occurred while processing {ticker}: {e}",
            exc_info=True,
        )
