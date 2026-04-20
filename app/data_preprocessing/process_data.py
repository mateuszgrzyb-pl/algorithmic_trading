import logging

import pandas as pd
from tqdm import tqdm

from app.feature_engineering.build_labels import build_triple_barier_labels_custom
from app.data_preprocessing.data_preprocessor import (
    deduplicate_price_data,
    remove_overlapped_observations,
    load_data,
    merge_data,
    save_processed_data,
    create_abt,
)
from app.utils.tools import (
    get_available_tickers,
    filter_sp500_companies,
    calculate_financial_ratios,
)
from app.config import settings


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def run_label_building() -> None:
    """Stage 1: Build triple-barrier labels for each ticker."""
    logging.info("--- Stage 1: Starting label construction ---")

    # Paths are now derived from the central settings
    raw_path = settings.base_path / "data/raw"
    stage1_dir = raw_path / "price_history" / "STAGE_1"
    stage2_dir = raw_path / "price_history" / "STAGE_2"

    tickers = get_available_tickers(str(stage1_dir))
    logging.info(f"Found {len(tickers)} tickers for label building.")

    for ticker in tqdm(tickers, desc="Stage 1: Building labels"):
        build_triple_barier_labels_custom(
            ticker=ticker,
            # Parameters are unpacked from the type-safe settings model
            **settings.label_params.model_dump(),
        )
    logging.info("--- Stage 1: Label construction finished ---")


def run_price_preprocessing() -> None:
    """Stages 2 & 3: Deduplicate price data and remove overlapping observations."""
    logging.info("--- Stages 2 & 3: Starting price preprocessing ---")
    raw_path = settings.base_path / "data/raw"
    stage2_dir = raw_path / "price_history" / "STAGE_2"
    stage3_dir = raw_path / "price_history" / "STAGE_3"
    stage4_dir = raw_path / "price_history" / "STAGE_4"

    tickers = get_available_tickers(str(stage2_dir))

    # Stage 2: Deduplicate to get the last price per quarter
    for ticker in tqdm(tickers, desc="Stage 2: Deduplicating"):
        deduplicate_price_data(ticker, input_dir=stage2_dir, output_dir=stage3_dir)

    # Stage 3: Remove overlaps to ensure target independence
    label_time = settings.overlap_params.label_time
    for idx, ticker in enumerate(tqdm(tickers, desc="Stage 3: Removing overlaps")):
        offset = idx % label_time
        remove_overlapped_observations(
            ticker, offset, label_time, input_dir=stage3_dir, output_dir=stage4_dir
        )
    logging.info("--- Stages 2 & 3: Price preprocessing finished ---")


def run_data_merge_and_save() -> None:
    """Stages 4, 5, 6: Find common tickers, load, merge, and save processed data."""
    logging.info("--- Stages 4-6: Starting data merging and saving ---")
    raw_path = settings.base_path / "data/raw"
    processed_path = settings.base_path / "data/processed"

    # Use sets for efficient intersection of available tickers
    tickers_bs = set(get_available_tickers(str(raw_path / "balance_sheets")))
    tickers_is = set(get_available_tickers(str(raw_path / "income_statements")))
    tickers_pr = set(get_available_tickers(str(raw_path / "price_history" / "STAGE_4")))

    common_tickers = tickers_pr.intersection(tickers_bs, tickers_is)
    logging.info(f"Found {len(common_tickers)} tickers with all required data sources.")

    for ticker in tqdm(common_tickers, desc="Stages 4-6: Merging data"):
        data = load_data(ticker, base_path=raw_path)  # Stage 4
        merged_df = merge_data(data)  # Stage 5
        save_processed_data(merged_df, ticker, base_path=processed_path)  # Stage 6
    logging.info("--- Stages 4-6: Data merging and saving finished ---")


def run_abt_creation_and_cleaning() -> None:
    """Stage 7 & post-processing: Create and clean the final Analytical Base Table."""
    logging.info("--- Stage 7: Starting ABT creation and cleaning ---")
    processed_path = settings.base_path / "data/processed"
    abt_path = settings.base_path / "data/abt"
    target_label = settings.target_label_name

    # Stage 7: Create the initial ABT
    create_abt(target=target_label, input_path=processed_path, output_path=abt_path)

    # Post-processing steps on the ABT
    logging.info("Starting ABT post-processing.")
    abt_raw_file = abt_path / f"{target_label}.feather"
    abt_df = pd.read_feather(abt_raw_file)

    abt_clean = filter_sp500_companies(abt_df)
    abt_clean = calculate_financial_ratios(abt_clean)
    abt_clean = abt_clean.sort_values(by="date", ascending=True).reset_index(drop=True)

    abt_clean_file = abt_path / f"{target_label}_clean.feather"
    abt_clean.to_feather(abt_clean_file)
    logging.info(f"Final clean ABT saved to {abt_clean_file}")
    logging.info("--- Stage 7: ABT creation and cleaning finished ---")


def main():
    """Main pipeline orchestrator."""
    logging.info("=========================================")
    logging.info("=== STARTING DATA PROCESSING PIPELINE ===")
    logging.info("=========================================")
    try:
        run_label_building()
        run_price_preprocessing()
        run_data_merge_and_save()
        run_abt_creation_and_cleaning()
        logging.info("===========================================")
        logging.info("=== PIPELINE COMPLETED SUCCESSFULLY! ===")
        logging.info("===========================================")
    except Exception:
        logging.critical("!!! PIPELINE FAILED WITH A CRITICAL ERROR:", exc_info=True)


if __name__ == "__main__":
    main()
