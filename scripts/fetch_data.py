import sys
from pathlib import Path
import logging

from tqdm import tqdm

from app.config import settings
from app.utils.tools import get_available_tickers, load_sp500_tickers
from app.data_preprocessing.data_loader import (
    download_price_history,
    download_balance_sheets,
    download_income_statements
)


sys.path.append(str(Path(__file__).parent.parent))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_ticker_data(ticker: str) -> bool:
    """
    Download all data types for a single ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True if all downloads successful, False otherwise
    """
    try:
        start_date = settings.analysis_start_date.strftime("%Y-%m-%d")
        end_date = settings.analysis_end_date.strftime("%Y-%m-%d")
        api_key = settings.finance_toolkit_key

        if not api_key:
            raise ValueError("API key not configured in settings")
        if not start_date or not end_date:
            raise ValueError("Analysis dates not configured in settings")

        logger.debug(f"Downloading data for {ticker}")

        download_price_history([ticker], start_date, end_date, api_key)
        download_balance_sheets([ticker], start_date, end_date, api_key)
        download_income_statements([ticker], start_date, end_date, api_key)

        logger.debug(f"Successfully downloaded data for {ticker}")
        return True

    except Exception as e:
        logger.error(f"Failed to download data for {ticker}: {str(e)}")
        return False


def main() -> None:
    """
    Main function to orchestrate the data fetching process.

    Process:
    1. Load S&P 500 tickers from CSV
    2. Check which tickers already exist locally
    3. Download missing data with progress tracking
    4. Report final statistics
    """
    try:
        logger.info("Starting S&P 500 data fetching process")

        # Load tickers
        all_tickers = load_sp500_tickers()
        logger.info(f"Total S&P 500 tickers: {len(all_tickers)}")

        # Check which tickers are already available
        available_tickers = get_available_tickers()
        logger.info(f"Already downloaded tickers: {len(available_tickers)}")

        # Find tickers that need to be downloaded
        tickers_to_download = [ticker for ticker in all_tickers if ticker not in available_tickers]
        logger.info(f"Tickers to download: {len(tickers_to_download)}")

        if not tickers_to_download:
            logger.info("All tickers already downloaded. Nothing to do.")
            return

        # Download missing data with progress tracking
        successful_downloads = 0
        failed_downloads = []

        for ticker in tqdm(tickers_to_download, desc="Downloading financial data"):
            success = download_ticker_data(ticker)
            if success:
                successful_downloads += 1
            else:
                failed_downloads.append(ticker)

        # Report final statistics
        logger.info("=" * 50)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total tickers processed: {len(tickers_to_download)}")
        logger.info(f"Successful downloads: {successful_downloads}")
        logger.info(f"Failed downloads: {len(failed_downloads)}")

        if failed_downloads:
            logger.warning(f"Failed tickers: {failed_downloads}")

        logger.info("Data fetching process completed")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
