import pandas as pd
import pytest

from app.utils.tools import (
    calculate_portfolio_xirr,
    ensure_directory,
    get_available_tickers,
    standardize_column_names,
)


def test_standardize_column_names_basic():
    cols = ["First Name", "Last-Name", "Age"]
    expected = ["first_name", "last_name", "age"]
    assert standardize_column_names(cols) == expected


def test_standardize_column_names_empty_list():
    assert standardize_column_names([]) == []


def test_standardize_column_names_already_snake_case():
    cols = ["first_name", "last_name"]
    expected = ["first_name", "last_name"]
    assert standardize_column_names(cols) == expected


def test_standardize_column_names_mixed_case_and_symbols():
    cols = ["Some Column", "Another-Column", "X Y-Z"]
    expected = ["some_column", "another_column", "x_y_z"]
    assert standardize_column_names(cols) == expected


def test_ensure_directory_creates_new(tmp_path):
    # tmp_path jest katalogiem tymczasowym
    new_dir = tmp_path / "new_folder"
    assert not new_dir.exists()

    ensure_directory(new_dir)
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_ensure_directory_existing(tmp_path):
    existing_dir = tmp_path / "existing_folder"
    existing_dir.mkdir()
    assert existing_dir.exists()

    ensure_directory(existing_dir)
    assert existing_dir.exists()
    assert existing_dir.is_dir()


def test_get_available_tickers_basic(tmp_path):
    files = ["AAPL.feather", "GOOG.feather", "MSFT.feather"]
    for f in files:
        (tmp_path / f).touch()
    result = get_available_tickers(tmp_path)
    assert result == sorted(["AAPL", "GOOG", "MSFT"])


def test_get_available_tickers_with_non_feather(tmp_path):
    files = ["AAPL.feather", "ignore.txt", "MSFT.feather"]
    for f in files:
        (tmp_path / f).touch()

    result = get_available_tickers(tmp_path)
    assert result == sorted(["AAPL", "MSFT"])


def test_get_available_tickers_duplicates(tmp_path):
    files = ["AAPL.feather", "AAPL.feather", "GOOG.feather"]
    for f in files:
        (tmp_path / f).touch()

    result = get_available_tickers(tmp_path)
    # powinno zwrócić unikalne tickery
    assert result == sorted(["AAPL", "GOOG"])


def test_get_available_tickers_empty_dir(tmp_path):
    # katalog pusty
    result = get_available_tickers(tmp_path)
    assert result == []


@pytest.fixture
def sample_df():
    df = pd.DataFrame(
        {
            "date": pd.PeriodIndex(["2024-01-01", "2024-04-01"], freq="Q-DEC"),
            "ticker": ["AAPL", "GOOG"],
            "adj_close": [100, 200],
            "tb_event_date": pd.to_datetime(["2024-06-01", "2024-09-01"]),
            "tb_pct_change": [10, -5],
        }
    )
    return df


def test_portfolio_basic(sample_df):
    pred = pd.Series([0.9, 0.2], index=sample_df.index)
    X = sample_df
    best_thresh = 0.5
    label = "tb"

    result = calculate_portfolio_xirr(sample_df, X, pred, best_thresh, label)

    assert not result["total_amount_invested"] == 1000
