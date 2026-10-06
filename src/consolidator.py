import csv
import json
from pathlib import Path
from typing import Sequence

try:
    from .table_extractor import Transaction
except (ImportError, ValueError):
    from table_extractor import Transaction


class MasterLedger:
    """Deduplication and order lifecycle rules.

    Dedup key: [us_trade_date + account + symbol + action + price].
    Lifecycle updates (e.g. WORKING 0/7 -> FILLED 7) are detected purely by comparing
    quantity. STATUS IS NEVER USED AS A DETECTOR.

    The ledger is rebuilt from scratch from the full.json per-image records on every
    consolidation — never merged incrementally — so manual edits to full.json always
    take effect.
    """

    def __init__(self):
        # Key: (us_trade_date, account, symbol, action, price) -> dict record
        self._records: dict[tuple, dict] = {}

    def process_transactions(self, transactions: Sequence[Transaction]) -> None:
        """Process a list of transactions (assumed to be ordered chronologically)."""
        for txn in transactions:
            details_val = str(getattr(txn, "details", "")).strip()
            if not details_val:
                details_val = f"{txn.price:g}" if txn.price else ""
            key = (txn.us_trade_date, txn.account, txn.symbol, txn.action, details_val)

            if key not in self._records:
                # New order entry
                self._records[key] = {
                    "us_trade_date": txn.us_trade_date,
                    "account": txn.account,
                    "symbol": txn.symbol,
                    "action": txn.action,
                    "details": details_val,
                    "quantity": txn.quantity,
                    "price": txn.price,
                    "fill_price": txn.fill_price,
                    "status": txn.status,
                    "first_seen_image": txn.image_name,
                    "last_updated_image": txn.image_name,
                    "history": [
                        {
                            "image": txn.image_name,
                            "quantity": txn.quantity,
                            "fill_price": txn.fill_price,
                            "status": txn.status,
                        }
                    ],
                }
            else:
                existing = self._records[key]
                # Compare quantity to determine if lifecycle progressed (e.g. 0/7 -> 7)
                if txn.quantity != existing["quantity"]:
                    existing["quantity"] = txn.quantity
                    existing["status"] = txn.status
                    if txn.fill_price is not None:
                        existing["fill_price"] = txn.fill_price
                    existing["last_updated_image"] = txn.image_name
                    existing["history"].append(
                        {
                            "image": txn.image_name,
                            "quantity": txn.quantity,
                            "fill_price": txn.fill_price,
                            "status": txn.status,
                        }
                    )
                else:
                    # Duplicate entry with identical quantity
                    if existing["fill_price"] is None and txn.fill_price is not None:
                        existing["fill_price"] = txn.fill_price

    def get_ledger(self) -> list[dict]:
        """Return the consolidated master ledger as a list of dicts."""
        return list(self._records.values())


def load_image_records(file_path: Path | str, on_legacy: str = "backup") -> list[dict]:
    """Load full.json per-image records: [{"index": 1, "image": ..., "transactions": [...]}, ...]

    Returns [] when the file is missing or in the legacy flat-ledger format.
    on_legacy: "backup" renames a legacy file to full.legacy.json (before it would be
    overwritten by normal processing); "skip" leaves it untouched (derived-output mode).
    """
    path = Path(file_path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Warning] Could not read {path}: {e} — starting fresh records.")
        return []
    if isinstance(data, list) and all(
        isinstance(e, dict) and "transactions" in e for e in data
    ):
        return data
    if on_legacy == "skip":
        print(f"[Skipped] {path}: legacy flat-ledger format; not touched.")
        return []
    backup = path.with_name("full.legacy.json")
    if backup.exists():
        path.unlink()
        print(f"[Migration] Removed legacy full.json (backup already at {backup.name}).")
    else:
        path.replace(backup)
        print(f"[Migration] Legacy full.json backed up as {backup.name}; indexed records start fresh.")
    return []


def save_image_records(file_path: Path | str, entries: Sequence[dict]) -> None:
    """Save the per-image indexed records to full.json (the editable source of truth)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(entries), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def append_image_entry(
    entries: Sequence[dict], transactions: Sequence[Transaction], image_name: str
) -> tuple[list[dict], int]:
    """Append (or replace) an image's transaction records; index = processing order.

    Re-processing an image with the same name replaces its records instead of
    duplicating them. Returns the new entries list and the entry's index.
    """
    entry_txns = [t.to_dict() if isinstance(t, Transaction) else dict(t) for t in transactions]
    kept = [e for e in entries if e.get("image") != image_name]
    index = max((e.get("index", 0) for e in kept), default=0) + 1
    kept.append({"index": index, "image": image_name, "transactions": entry_txns})
    return kept, index


def consolidate_entries(entries: Sequence[dict]) -> list[dict]:
    """Rebuild the deduplicated master ledger from all per-image records, in index order."""
    ledger = MasterLedger()
    valid_fields = set(Transaction.__dataclass_fields__.keys()) if hasattr(Transaction, "__dataclass_fields__") else None
    for entry in sorted(entries, key=lambda e: e.get("index", 0)):
        for txn_dict in entry.get("transactions", []):
            try:
                kwargs = {k: v for k, v in txn_dict.items() if k in valid_fields} if valid_fields else txn_dict
                txn = Transaction(**kwargs)
            except (TypeError, ValueError) as err:
                print(f"[Warning] Skipping malformed record in image #{entry.get('index')}: {txn_dict.get('symbol')} ({err})")
                continue
            ledger.process_transactions([txn])
    return ledger.get_ledger()


def save_consolidated_json(data: Sequence[dict], file_path: Path | str) -> None:
    """Save the consolidated ledger to consolidated.json."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(list(data), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_ledger_csv(data: Sequence[dict], file_path: Path | str) -> None:
    """Save the consolidated ledger to full.csv."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "account",
        "action",
        "details",
        "quantity",
        "fill_price",
        "status",
        "us_trade_date",
        "first_seen_image",
        "last_updated_image",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            row_dict = dict(row)
            if "details" not in row_dict or row_dict["details"] is None:
                row_dict["details"] = row_dict.get("price")
            writer.writerow(row_dict)


def save_image_json(transactions: Sequence[Transaction], output_file: Path | str) -> None:
    """Save raw extracted transactions for an individual screenshot to <image_name>.json."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [t.to_dict() for t in transactions]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_image_csv(transactions: Sequence[Transaction], output_file: Path | str) -> None:
    """Save raw extracted transactions for an individual screenshot to <image_name>.csv."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "account",
        "action",
        "details",
        "quantity",
        "fill_price",
        "status",
        "us_trade_date",
        "image_name",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for t in transactions:
            row_dict = t.to_dict() if hasattr(t, "to_dict") else dict(t)
            if "details" not in row_dict or row_dict["details"] is None:
                row_dict["details"] = row_dict.get("price")
            writer.writerow(row_dict)
