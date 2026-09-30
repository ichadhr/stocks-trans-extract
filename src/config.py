import json
import os
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Automatically load environment variables from .env
load_dotenv()

# US Market Domain Constants
MARKET_TIMEZONE: str = "America/New_York"
US_MARKET_OPEN_TIME: time = time(9, 30)


@dataclass(frozen=True)
class AppConfig:
    local_timezone: str = "Asia/Jakarta"
    date_format: str = "MM/DD/YYYY"
    time_format: str = "auto"
    table_top: int = 116
    table_bottom: int = 638
    table_left: int = 26
    table_right: int = 780
    gemini_model: str = "gemini-3.5-flash"

    @property
    def table_crop_box(self) -> tuple[int, int, int, int]:
        """Returns (left, top, right, bottom) crop box for table OCR."""
        return (self.table_left, self.table_top, self.table_right, self.table_bottom)

    @property
    def local_tz(self) -> ZoneInfo:
        return ZoneInfo(self.local_timezone)

    @property
    def market_tz(self) -> ZoneInfo:
        return ZoneInfo(MARKET_TIMEZONE)

    @property
    def is_day_first(self) -> bool:
        """True if date starts with day (e.g. DD/MM/YYYY or d/m/yyyy)."""
        clean = self.date_format.strip().lower()
        return clean.startswith("d")

    @property
    def strptime_date_format(self) -> str:
        """Converts user-friendly date format to Python strptime format."""
        f = self.date_format.strip()
        mapping = {
            "MM/DD/YYYY": "%m/%d/%Y",
            "M/D/YYYY": "%m/%d/%Y",
            "DD/MM/YYYY": "%d/%m/%Y",
            "D/M/YYYY": "%d/%m/%Y",
            "YYYY-MM-DD": "%Y-%m-%d",
            "YYYY/MM/DD": "%Y/%m/%d",
        }
        for k, v in mapping.items():
            if f.lower() == k.lower():
                return v
        return f if "%" in f else "%m/%d/%Y"


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load user preferences from config.json or environment variables."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
    else:
        config_path = Path(config_path)

    local_tz = os.getenv("LOCAL_TIMEZONE")
    date_fmt = os.getenv("DATE_FORMAT")
    time_fmt = os.getenv("TIME_FORMAT")
    t_top = os.getenv("TABLE_TOP")
    t_bottom = os.getenv("TABLE_BOTTOM")
    t_left = os.getenv("TABLE_LEFT")
    t_right = os.getenv("TABLE_RIGHT")

    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not local_tz:
                    local_tz = data.get("local_timezone")
                if not date_fmt:
                    date_fmt = data.get("date_format")
                if not time_fmt:
                    time_fmt = data.get("time_format")
                nested_crop = data.get("table_crop", {})
                if not t_top:
                    t_top = data.get("table_top", nested_crop.get("top"))
                if not t_bottom:
                    t_bottom = data.get("table_bottom", nested_crop.get("bottom"))
                if not t_left:
                    t_left = data.get("table_left", nested_crop.get("left"))
                if not t_right:
                    t_right = data.get("table_right", nested_crop.get("right"))
                gem_model = data.get("gemini_model")
        except Exception as e:
            print(f"[Warning] Could not read {config_path}: {e}")

    return AppConfig(
        local_timezone=local_tz or "Asia/Jakarta",
        date_format=date_fmt or "MM/DD/YYYY",
        time_format=time_fmt or "auto",
        table_top=int(t_top) if t_top is not None else 116,
        table_bottom=int(t_bottom) if t_bottom is not None else 638,
        table_left=int(t_left) if t_left is not None else 26,
        table_right=int(t_right) if t_right is not None else 780,
        gemini_model=gem_model or "gemini-3.5-flash",
    )
