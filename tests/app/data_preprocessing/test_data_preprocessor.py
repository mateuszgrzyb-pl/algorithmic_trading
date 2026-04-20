from pathlib import Path

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from app.data_preprocessing.data_preprocessor import load_data, save_processed_data


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-03-31"],
            "value": [100, 200],
        }
    )


def test_load_data_some_files(tmp_path: Path, sample_data):
    ticker = "AAPL"
    base = tmp_path / "data" / "raw"
    (base / "price_history/STAGE_4").mkdir(parents=True)
    (base / "income_statements").mkdir(parents=True)

    sample_data.to_feather(base / "price_history/STAGE_4" / f"{ticker}.feather")
    sample_data.to_feather(base / "income_statements" / f"{ticker}.feather")
    data = load_data(ticker, base_path=base)
    assert "price_history" in data
    assert "income_statements" in data
    assert "balance_sheets" not in data


def test_save_processed_data_creates_file_with_correct_content(tmp_path: Path):
    df_to_save = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ticker = "TEST_TICKER"
    expected_file_path = tmp_path / f"{ticker}.feather"
    save_processed_data(df_to_save, ticker, base_path=tmp_path)
    assert expected_file_path.exists(), "Plik .feather nie został utworzony!"
    df_loaded = pd.read_feather(expected_file_path)
    assert_frame_equal(
        df_loaded, df_to_save, "Zawartość zapisanego pliku różni się od oryginału!"
    )
