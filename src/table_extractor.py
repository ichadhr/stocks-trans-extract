import re
from dataclasses import dataclass, asdict
from pathlib import Path
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# Lazy singleton OCR instance to avoid reloading models
_OCR_ENGINE = None

def get_ocr_engine() -> RapidOCR:
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE

# Table crop boundaries on standard 1280x719 screenshot
# X1=26: skips icon column to prevent icon noise from merging with symbols
# Y1=116: starts directly below the table column header line
# X2=780: ends after Fill Price column
# Y2=638: maximum inner table boundary; complete rows are kept, sliced rows dropped
DEFAULT_TABLE_CROP = (26, 116, 780, 638)

ACCOUNT_REGEX = re.compile(r"U\d{8}")
LMT_PRICE_REGEX = re.compile(r"(?:LMT|MID|Cap)?\s*(\d+\.\d+)")
PRICE_NUMBER_REGEX = re.compile(r"(\d+\.\d+)")
QTY_REGEX = re.compile(r"^(\d+/\d+|\d+)$")
SYMBOL_REGEX = re.compile(r"\b([A-Z]{1,5})\b")


@dataclass
class Transaction:
    image_name: str
    us_trade_date: str
    account: str
    symbol: str
    action: str          # BUY or SELL
    quantity: str        # e.g. "0/7", "7", "44"
    price: float = 0.0   # Limit / order price (Details column in TWS)
    fill_price: float | None = None
    status: str = ""     # WORKING, PARTIAL, FILLED (informational only; derived from quantity)
    raw_text: str = ""
    details: float | None = None

    def __post_init__(self):
        if self.details is not None and (self.price == 0.0 or self.price is None):
            self.price = float(self.details)
        elif self.price is not None and self.details is None:
            self.details = float(self.price)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("details") is None:
            d["details"] = d.get("price")
        return d


def parse_row_boxes(items: list[tuple], image_name: str, us_trade_date: str) -> Transaction | None:
    """Parse clustered OCR boxes belonging to a single table row.
    
    items: list of (x_min, x_max, text, score, y_center) sorted left-to-right.
    """
    row_text = " | ".join(t[2] for t in items)

    # 1. Account validation: must match U\d{8}
    account = None
    account_idx = None
    for i, it in enumerate(items):
        m = ACCOUNT_REGEX.search(it[2])
        if m:
            account = m.group(0)
            account_idx = i
            break

    # If no valid account (e.g. sliced bottom row where 'U' is corrupted to '11'), drop row
    if not account or account_idx is None:
        return None

    # 2. Symbol: extract from items before the account
    symbol = None
    for i in range(account_idx):
        t = items[i][2].strip()
        m_sym = SYMBOL_REGEX.search(t)
        if m_sym:
            candidate = m_sym.group(1)
            if candidate not in ("BUY", "SELL", "LMT", "MID"):
                symbol = candidate
                break

    # Fallback for symbol if merged before account
    if not symbol:
        before_str = " ".join(items[i][2] for i in range(account_idx))
        m_sym = SYMBOL_REGEX.search(before_str)
        if m_sym and m_sym.group(1) not in ("BUY", "SELL", "LMT", "MID"):
            symbol = m_sym.group(1)

    if not symbol:
        return None

    # 3. Action: BUY or SELL
    all_after_text = " ".join(it[2] for it in items[account_idx:])
    if "BUY" in all_after_text.upper():
        action = "BUY"
    elif any(k in all_after_text for k in ["SELL", "S...", "S.", "MID"]):
        action = "SELL"
    else:
        action = "UNKNOWN"

    # 4. Quantity and Fill Price (right side of table, x_min > 480)
    right_items = [it for it in items if it[0] > 480]
    quantity = None
    fill_price = None

    for it in right_items:
        t = it[2].strip()
        if t in ("Cancel", "V", "▼", "、"):
            continue
        if quantity is None:
            m_qty = QTY_REGEX.search(t)
            if m_qty:
                quantity = m_qty.group(1)
                continue
        if fill_price is None:
            m_px = PRICE_NUMBER_REGEX.search(t)
            if m_px:
                fill_price = float(m_px.group(1))
                continue

    # 5. Order / Limit price (search in items starting from account)
    limit_price = None
    for it in items[account_idx:]:
        m_lmt = LMT_PRICE_REGEX.search(it[2])
        if m_lmt:
            if "LMT" in it[2] or it[0] < 520:
                limit_price = float(m_lmt.group(1))
                break

    price = limit_price if limit_price is not None else fill_price
    if price is None:
        price = 0.0

    # 6. Status: derived strictly from quantity.
    # CRITICAL: Status is purely informational. NEVER use status as a detector or for dedup!
    if quantity:
        if "/" in quantity:
            parts = quantity.split("/")
            filled = int(parts[0])
            status = "WORKING" if filled == 0 else "PARTIAL"
        else:
            status = "FILLED"
    else:
        status = "UNKNOWN"

    return Transaction(
        image_name=image_name,
        us_trade_date=us_trade_date,
        account=account,
        symbol=symbol,
        action=action,
        quantity=quantity or "0",
        price=price,
        fill_price=fill_price,
        status=status,
        raw_text=row_text,
        details=price,
    )


def extract_table_transactions(
    image_path: Path | str,
    us_trade_date: str,
    crop_box: tuple[int, int, int, int] = DEFAULT_TABLE_CROP,
) -> list[Transaction]:
    """Crops the transaction table, runs RapidOCR, clusters rows, and extracts structured transactions."""
    path = Path(image_path)
    ocr = get_ocr_engine()

    with Image.open(path) as img:
        crop = img.crop(crop_box)
        results, _ = ocr(crop)

    if not results:
        return []

    # Calculate y_center and x coordinates for each detected bounding box
    items = []
    for box, text, score in results:
        y_center = (box[0][1] + box[2][1]) / 2.0
        x_min = box[0][0]
        x_max = box[1][0]
        items.append((x_min, x_max, text, score, y_center))

    # Sort vertically to group into horizontal rows
    items.sort(key=lambda x: x[4])

    rows = []
    current_row = []
    current_y = None
    row_tolerance_px = 8

    for it in items:
        if current_y is None or abs(it[4] - current_y) <= row_tolerance_px:
            current_row.append(it)
            current_y = sum(x[4] for x in current_row) / len(current_row)
        else:
            current_row.sort(key=lambda x: x[0])  # Sort row left-to-right
            rows.append(current_row)
            current_row = [it]
            current_y = it[4]

    if current_row:
        current_row.sort(key=lambda x: x[0])
        rows.append(current_row)

    transactions = []
    for r in rows:
        txn = parse_row_boxes(r, path.name, us_trade_date)
        if txn:
            transactions.append(txn)

    return transactions
