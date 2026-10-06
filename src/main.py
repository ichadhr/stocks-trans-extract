import os
import shutil
import sys
from pathlib import Path

# Ensure src directory is in sys.path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from PIL import Image

try:
    from .config import load_config, MARKET_TIMEZONE, US_MARKET_OPEN_TIME
    # table_extractor (onnxruntime) must be imported before extractor (winocr/WinRT)
    from .table_extractor import extract_table_transactions
    from .gemini_extractor import validate_and_fill_with_gemini
    from .extractor import extract_us_trade_date
    from .consolidator import (
        load_image_records,
        save_image_records,
        append_image_entry,
        consolidate_entries,
        save_consolidated_json,
        save_ledger_csv,
        save_image_json,
        save_image_csv,
    )
except (ImportError, ValueError):
    from config import load_config, MARKET_TIMEZONE, US_MARKET_OPEN_TIME
    from table_extractor import extract_table_transactions
    from gemini_extractor import validate_and_fill_with_gemini
    from extractor import extract_us_trade_date
    from consolidator import (
        load_image_records,
        save_image_records,
        append_image_entry,
        consolidate_entries,
        save_consolidated_json,
        save_ledger_csv,
        save_image_json,
        save_image_csv,
    )


def print_txn_table(txns) -> None:
    """Print transactions as an aligned console table with | column separators."""
    if not txns:
        print("    (no rows)")
        return
    header = (
        f"    {'SYMBOL':<7}| {'ACCOUNT':<10}| {'ACTION':<8}| {'DETAILS':<11}| {'QTY':<6}|"
        f" {'FILL':>8} | STATUS"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    for t in txns:
        fill = "-" if t.fill_price is None else f"{t.fill_price:g}"
        details_val = str(getattr(t, "details", "")).strip() or (f"{t.price:g}" if t.price else "-")
        print(
            f"    {t.symbol:<7}| {t.account:<10}| {t.action:<8}| {details_val:<11}| {t.quantity:<6}|"
            f" {fill:>8} | {t.status}"
        )


def rebuild_derived_outputs(mode: str, archive_base_dir: Path) -> None:
    """Rebuild --consolidate / --csv outputs from full.json records. No OCR, no Gemini."""
    found_any = False
    for date_dir in sorted(archive_base_dir.glob("*")) if archive_base_dir.exists() else []:
        full_json_path = date_dir / "full.json"
        if not date_dir.is_dir() or not full_json_path.is_file():
            continue
        found_any = True
        entries = load_image_records(full_json_path, on_legacy="skip")
        data = consolidate_entries(entries)
        if mode == "consolidate":
            save_consolidated_json(data, date_dir / "consolidated.json")
            target = "consolidated.json"
        else:
            save_ledger_csv(data, date_dir / "full.csv")
            target = "full.csv"
        print(f"  archives/{date_dir.name}/ : {len(data)} unique orders -> {target}")
    if not found_any:
        print("  No date folders with full.json found under archives/.")


def main() -> None:
    flags = set(sys.argv[1:])
    config = load_config()
    project_root = Path(__file__).resolve().parent.parent
    uploads_dir = project_root / "uploads"
    archive_base_dir = project_root / "archives"

    if "--consolidate" in flags or "--csv" in flags:
        mode = "consolidate" if "--consolidate" in flags else "csv"
        print(f"Rebuilding {mode} output from full.json records (no OCR / Gemini calls)...")
        rebuild_derived_outputs(mode, archive_base_dir)
        return

    if not uploads_dir.exists():
        print(f"Uploads directory not found: {uploads_dir}")
        return

    image_paths = sorted(uploads_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No screenshots found in {uploads_dir}. Folder is clean.")
        return

    print("=" * 115)
    print(
        f"US Stock Transaction Extractor | Local TZ: '{config.local_timezone}' -> Market TZ: '{MARKET_TIMEZONE}' | "
        f"Market Open: {US_MARKET_OPEN_TIME.strftime('%H:%M')} ET"
    )
    print(f"Crop Config: top={config.table_top}, bottom={config.table_bottom}, left={config.table_left}, right={config.table_right}")
    print("=" * 115)

    # Step 1: Detect authentic US trading dates for all incoming images
    print(f"\n[Step 1/4] Detecting US Trading Dates for {len(image_paths)} incoming screenshot(s)...")
    images_with_metadata = []
    for img_path in image_paths:
        res = extract_us_trade_date(img_path, config)
        if res.error:
            print(f"  [ERROR] {img_path.name}: {res.error} (leaving in uploads)")
            continue
        images_with_metadata.append((img_path, res))
        session_note = "Pre-market (Previous Session)" if res.session_rolled_over else "Active / Post Session"
        print(
            f"  {res.image_name:<32} | Taskbar: {res.local_raw_time} {res.local_raw_date} | "
            f"US Trade Date: {res.us_trade_date} | {session_note}"
        )

    # Sort incoming images strictly by authentic local datetime
    images_with_metadata.sort(key=lambda x: x[1].local_datetime)

    # Step 2: Process each image into its respective archives/{US Trade Date}/ folder
    print("\n[Step 2/4] Extracting transactions (local OCR) and archiving per US Trading Date...")
    processed_count = 0
    date_summaries = {}

    for img_path, res in images_with_metadata:
        trade_date = res.us_trade_date
        date_archive_dir = archive_base_dir / trade_date
        date_archive_dir.mkdir(parents=True, exist_ok=True)

        # 2a. Extract table transactions from this image
        txns = extract_table_transactions(img_path, trade_date, crop_box=config.table_crop_box)
        print(f"\n  [Step 2] Local OCR draft for {img_path.name} ({len(txns)} rows):")
        print_txn_table(txns)

        # 2b. Gemini Vision audit: validates every row and fixes gap/wrong/missing
        # values (falls back to local heuristics when the API is unavailable)
        with Image.open(img_path) as img_obj:
            crop_img = img_obj.crop(config.table_crop_box)
            txns = validate_and_fill_with_gemini(
                crop_img,
                txns,
                image_name=img_path.name,
                us_trade_date=trade_date,
                model_name=config.gemini_model,
            )
        print(f"\n  [Step 3] Gemini audited result ({len(txns)} rows):")
        print_txn_table(txns)

        # 2c. Save per-image JSON and CSV: archives/{trade_date}/{image_name}.json / .csv
        img_json_path = date_archive_dir / f"{img_path.stem}.json"
        img_csv_path = date_archive_dir / f"{img_path.stem}.csv"
        save_image_json(txns, img_json_path)
        save_image_csv(txns, img_csv_path)

        # 2d. Store this image's records in full.json (indexed by processing order),
        # then rebuild the consolidated ledger from ALL stored records (deterministic,
        # no AI). Manual edits to full.json take effect on every rebuild.
        full_json_path = date_archive_dir / "full.json"
        full_csv_path = date_archive_dir / "full.csv"
        entries = load_image_records(full_json_path)
        entries, entry_index = append_image_entry(entries, txns, img_path.name)
        save_image_records(full_json_path, entries)

        consolidated = consolidate_entries(entries)
        save_consolidated_json(consolidated, date_archive_dir / "consolidated.json")
        save_ledger_csv(consolidated, full_csv_path)

        # 2e. Move the screenshot to the date archive folder
        dest_img_path = date_archive_dir / img_path.name
        shutil.move(str(img_path), str(dest_img_path))
        processed_count += 1

        date_summaries[trade_date] = len(consolidated)

        print(
            f"  Processed {img_path.name}:\n"
            f"    -> Extracted {len(txns)} orders -> saved {img_json_path.name}\n"
            f"    -> Stored in full.json as image #{entry_index} "
            f"({len(entries)} image record(s) for {trade_date})\n"
            f"    -> Consolidated: {len(consolidated)} unique orders -> consolidated.json + full.csv\n"
            f"    -> Moved image to archives/{trade_date}/{img_path.name}"
        )

    # Step 3: Verify clean uploads folder and print final summary
    print("\n[Step 4/4] Execution Summary:")
    remaining_uploads = list(uploads_dir.glob("*.jpg"))
    print(f"  Uploads folder status   : {'CLEAN (0 images)' if not remaining_uploads else f'{len(remaining_uploads)} unhandled image(s)'}")
    print(f"  Screenshots processed   : {processed_count}")
    print(f"  Date archives updated   :")
    for d, total_orders in sorted(date_summaries.items()):
        print(f"    - archives/{d}/ (consolidated.json: {total_orders} unique orders)")
    print("=" * 115)


if __name__ == "__main__":
    main()
