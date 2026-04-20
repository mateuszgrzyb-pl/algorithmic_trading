import pandas as pd
from typing import Optional


def triple_barrier_labeling_custom(
    df: pd.DataFrame,
    price_col: str,
    label_name: str,
    date_col: Optional[str] = None,
    profit_target: float = 40.0,
    stop_loss: float = 20.0,
    max_days: int = 250,
) -> pd.DataFrame:
    """
    Implements a custom Triple Barrier Method (first introduced by Marcos López de Prado) to generate labels for financial time series.

    This method assigns a label {-1, 0, 1} to each price point based on future
    price movements. The barriers are:
    1. Upper Barrier (Profit Take): A target price increase.
    2. Lower Barrier (Stop Loss): A maximum acceptable price decrease.
    3. Vertical Barrier (Time Limit): A maximum number of days to hold the position.

    The logic is as follows:
    - The stop-loss is a "first-touch" barrier. If the price hits the stop-loss
      threshold at any point within `max_days`, the event is triggered, and the
      label is -1.
    - The profit-target is evaluated ONLY at the end of the `max_days` window.
      If the stop-loss was not triggered, the price at `t + max_days` is checked.
      If it meets or exceeds the profit target, the label is 1; otherwise, it is 0.

    Args:
        df (pd.DataFrame): Input DataFrame containing price and optional date columns.
        price_col (str): The name of the column with price data.
        label_name (str): The prefix for the new columns that will be created.
        date_col (str, optional): The name of the date column. If None, the function
                                  assumes a DatetimeIndex. Defaults to None.
        profit_target (float): The percentage profit target (e.g., 40 for 40%).
        stop_loss (float): The percentage stop-loss limit (e.g., 20 for 20%).
        max_days (int): The maximum number of trading days for the vertical barrier.

    Returns:
        pd.DataFrame: The original DataFrame with the following new columns added:
            - `{label_name}_target`: The final label {-1, 0, 1}.
            - `{label_name}_final_price`: The price at which the event was triggered.
            - `{label_name}_days_to_event`: Number of days until the barrier was hit.
            - `{label_name}_event_date`: The date when the event occurred.
            - `{label_name}_pct_change`: The percentage price change at the event time.
    """
    df = df.copy()
    profit_target /= 100
    stop_loss /= 100
    n = len(df)

    # Initialize new columns
    labels = [0] * n
    final_prices = df[price_col].values.copy()
    days_to_events = [max_days] * n
    event_dates = [pd.NaT] * n
    pct_changes = [0.0] * n

    prices = df[price_col].values
    dates = df[date_col].dt.to_timestamp().values

    for i in range(n):
        current_price = prices[i]
        stop_triggered = False

        # Checking the STOP LOSS condition during max_days
        for j in range(1, min(max_days + 1, n - i)):
            future_price_j = prices[i + j]
            change_j = (future_price_j - current_price) / current_price
            if change_j <= -stop_loss:
                labels[i] = -1
                final_prices[i] = future_price_j
                days_to_events[i] = j
                event_dates[i] = dates[i + j]
                pct_changes[i] = change_j * 100
                stop_triggered = True
                break

        # Checking gwoth after max_days, if stop loss did not occur
        if not stop_triggered and i + max_days < n:
            future_price = prices[i + max_days]
            change = (future_price - current_price) / current_price
            labels[i] = 1 if change >= profit_target else 0
            final_prices[i] = future_price
            days_to_events[i] = max_days
            event_dates[i] = dates[i + max_days]
            pct_changes[i] = change * 100

        # Case handling when there is not enough days
        elif not stop_triggered:
            labels[i] = 0  # Użyj 0 zamiast None
            final_prices[i] = prices[-1]
            days_to_events[i] = n - i - 1
            event_dates[i] = dates[-1]
            pct_changes[i] = ((prices[-1] - current_price) / current_price) * 100

    # Adding new columns to DataFrame
    df[f"{label_name}_target"] = labels
    df[f"{label_name}_final_price"] = final_prices
    df[f"{label_name}_days_to_event"] = days_to_events
    df[f"{label_name}_event_date"] = event_dates
    df[f"{label_name}_pct_change"] = pct_changes
    return df
