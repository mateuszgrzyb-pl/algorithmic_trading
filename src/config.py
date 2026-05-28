from datetime import date
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class LabelParams(BaseModel):
    """Parameters for the triple-barrier labeling process."""

    profit_targets: List[int] = [2000]
    stop_losses: List[int] = [100]
    max_days: List[int] = [252]
    overwrite: bool = True


class OverlapParams(BaseModel):
    """Parameters for removing overlapping observations."""

    label_time: int = 5


class Settings(BaseSettings):
    """
    Main application settings.

    Aggregates all configuration parameters and loads them from environment
    variables or a .env file, providing a single, type-safe source of truth.
    """

    finance_toolkit_key: str
    analysis_start_date: Optional[date] = None
    analysis_end_date: Optional[date] = None

    base_path: Path = Path("data")
    target_label_name: str = "label_2000_100_252"

    label_params: LabelParams = LabelParams()
    overlap_params: OverlapParams = OverlapParams()

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )


settings = Settings()
