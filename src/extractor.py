import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from PIL import Image
import winocr
from dateutil import parser as date_parser

try:
    from .config import AppConfig, US_MARKET_OPEN_TIME, MARKET_TIMEZONE
except (ImportError, ValueError):
    from config import AppConfig, US_MARKET_OPEN_TIME, MARKET_TIMEZONE

# 12-hour regex: matches "4:31 PM", "1023 AM", "04:32:15 PM"
REGEX_TIME_12H = re.compile(r"\b(\d{1,2}):?(\d{2})(?::\d{2})?\s*(AM|PM)\b", re.IGNORECASE)

# 24-hour regex: matches "16:31", "04:32", "16:31:13"
REGEX_TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b")

# Date regex: matches "9/23/2026", "23/09/2026", "2026-09-23", "23-09-2026"
REGEX_DATE = re.compile(r"\b(\d{1,4}[/-]\d{1,2}[/-]\d{2,4})\b")


@dataclass
class TimestampExtraction:
    image_name: str
    local_raw_time: str | None = None
    local_raw_date: str | None = None
    local_datetime: datetime | None = None
    us_market_datetime: datetime | None = None
    us_trade_date: str | None = None
    session_rolled_over: bool = False
    error: str | None = None


def get_previous_trading_day(dt: datetime) -> datetime:
    """Return the previous trading day, skipping weekends (Saturday & Sunday)."""
    prev = dt - timedelta(days=1)
    while prev.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        prev -= timedelta(days=1)
    return prev


def parse_time_from_text(text: str, mode: str = "auto") -> str | None:
    """Extract time string based on mode ('auto', '12h', '24h')."""
    m_12h = REGEX_TIME_12H.search(text)
    m_24h = REGEX_TIME_24H.search(text)

    if mode == "12h":
        if m_12h:
            return f"{m_12h.group(1)}:{m_12h.group(2)} {m_12h.group(3).upper()}"
        return None

    if mode == "24h":
        if m_24h:
            return f"{m_24h.group(1)}:{m_24h.group(2)}"
        return None

    # 'auto' mode: prefer 12h if AM/PM indicator found, otherwise check 24h
    if m_12h:
        return f"{m_12h.group(1)}:{m_12h.group(2)} {m_12h.group(3).upper()}"
    if m_24h:
        return f"{m_24h.group(1)}:{m_24h.group(2)}"

    return None


def extract_timestamp_from_taskbar(img: Image.Image, config: AppConfig) -> tuple[str | None, str | None]:
    """Crops the bottom-right taskbar area, upscales, and OCRs time and date."""
    w, h = img.size

    # Crop the taskbar clock area (bottom-right ~130px width, ~70px height)
    crop = img.crop((max(0, w - 130), max(0, h - 70), w, h))
    upscaled = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.LANCZOS)

    result = winocr.recognize_pil_sync(upscaled)
    full_text = " ".join(line["text"] for line in result["lines"])

    time_str = parse_time_from_text(full_text, mode=config.time_format)
    d_match = REGEX_DATE.search(full_text)
    date_str = d_match.group(1) if d_match else None

    # Fallback to wider area if either is missing
    if not (time_str and date_str):
        wider_crop = img.crop((int(w * 0.75), int(h * 0.88), w, h))
        wider_upscaled = wider_crop.resize((wider_crop.width * 2, wider_crop.height * 2), Image.Resampling.BILINEAR)
        wider_result = winocr.recognize_pil_sync(wider_upscaled)
        wider_text = " ".join(line["text"] for line in wider_result["lines"])

        if not time_str:
            time_str = parse_time_from_text(wider_text, mode=config.time_format)
        if not date_str:
            d_match = REGEX_DATE.search(wider_text)
            if d_match:
                date_str = d_match.group(1)

    return time_str, date_str


def parse_datetime_with_config(date_str: str, time_str: str, config: AppConfig) -> datetime:
    """Parse date and time using configured format rules."""
    combined = f"{date_str} {time_str}".strip()

    # 1. Try explicit strptime with configured date format
    for tf in ["%I:%M %p", "%H:%M", "%I:%M:%S %p", "%H:%M:%S"]:
        fmt = f"{config.strptime_date_format} {tf}"
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            pass

    # 2. Flexible fallback using dateutil with configured day_first setting
    return date_parser.parse(combined, dayfirst=config.is_day_first)


def extract_us_trade_date(image_path: Path | str, config: AppConfig) -> TimestampExtraction:
    """Extract local timestamp from image pixels and convert to authentic US Trading Date.

    Uses US_MARKET_OPEN_TIME constant (09:30 Eastern Time):
    - If screenshot is taken before 09:30 AM ET, the displayed completed orders
      belong to the previous trading session (e.g. yesterday or Friday if Monday).
    - If taken at or after 09:30 AM ET, orders belong to the current day.
    """
    path = Path(image_path)
    extraction = TimestampExtraction(image_name=path.name)

    try:
        with Image.open(path) as img:
            time_str, date_str = extract_timestamp_from_taskbar(img, config)
            extraction.local_raw_time = time_str
            extraction.local_raw_date = date_str

            if not date_str:
                extraction.error = "Could not detect date from image"
                return extraction

            if not time_str:
                time_str = "12:00 PM"
                extraction.local_raw_time = "(estimated) 12:00 PM"

            # Parse datetime using configured date/time rules
            dt_naive = parse_datetime_with_config(date_str, time_str, config)
            dt_local = dt_naive.replace(tzinfo=config.local_tz)
            extraction.local_datetime = dt_local

            # Convert to US market timezone (America/New_York)
            dt_us = dt_local.astimezone(config.market_tz)
            extraction.us_market_datetime = dt_us

            # Apply market open rollover constant (09:30 ET)
            if dt_us.time() < US_MARKET_OPEN_TIME:
                session_dt = get_previous_trading_day(dt_us)
                extraction.session_rolled_over = True
            else:
                session_dt = dt_us
                extraction.session_rolled_over = False

            extraction.us_trade_date = session_dt.strftime("%Y-%m-%d")

    except Exception as e:
        extraction.error = str(e)

    return extraction
