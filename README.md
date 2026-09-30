# stocks-trans-extract

Turn Interactive Brokers (TWS) order-table screenshots into a clean, deduplicated
per-day trade ledger — 100% local OCR for the heavy lifting, Gemini Vision as an
always-on validator.

Drop a TWS screenshot into `uploads/`, run the pipeline, and get structured
orders (symbol, account, BUY/SELL, quantity, limit/fill price, status) merged into
`archives/{US trade date}/full.json`.

> **Windows only** — uses WinRT OCR (`winocr`) for the taskbar clock.

## Workflow (important)

**Feed screenshots one at a time, in the order you captured them:**

1. Drop **one** screenshot into `uploads/`
2. Run the pipeline (see below)
3. Check the console (`Successfully audited and verified N orders`) and the JSON
4. Repeat with the next screenshot

Screenshots are merged cumulatively into `archive/{trade_date}/full.json`, so
sequential runs build the day's ledger in your capture order. Do not batch-drop
screenshots that show the same orders in different states (e.g. `WORKING` then
`FILLED`) — the ledger merges last-write-wins per order, so upload order matters.

## How it works

1. **Date authentication** — OCRs the Windows taskbar clock inside the screenshot
   (never file metadata or filenames, which are download/copy times) to get the
   real capture moment, converts local time → US Eastern, and assigns the US trade
   date. Captures before 09:30 ET roll over to the previous trading day.
2. **Table extraction** — crops the orders table (fixed pixel box), runs RapidOCR
   locally, clusters rows, and parses each into a `Transaction`. Gemini Vision
   (`gemini-3.5-flash`, with fallback chain) then audits **every** row against the
   image — fixing gaps, wrong values (misread digits, clipped quantities), and
   adding rows local OCR missed. Falls back to local heuristics if the API is down.
3. **Consolidation** — saves per-image JSON/CSV, moves the screenshot into
   `archive/{US trade date}/`, and merges into a master ledger.

## Usage

```bash
uv run python src/main.py              # full pipeline (OCR + Gemini + consolidate)
uv run python src/main.py --consolidate  # rebuild consolidated.json from full.json only (no OCR/Gemini)
uv run python src/main.py --csv          # rebuild full.csv from full.json only (no OCR/Gemini)
```

### Fixing errors

`full.json` is the editable source of truth: it stores every image's records
indexed by processing order (`index: 1` = first image processed, then 2, 3, ...).
If a record is wrong (e.g. a model misread), open `full.json`, fix or delete the
record, then re-run `--consolidate` and/or `--csv`. The consolidated ledger is
rebuilt from scratch from those records every time — nothing incremental, so
manual corrections and deletions always take effect.

## Setup

```bash
uv sync
```

- `.env` — `GEMINI_API_KEY=...` (optional; without it the pipeline uses local
  heuristics only)
- `config.json`:

| Key | Purpose | Default |
|---|---|---|
| `local_timezone` | Your timezone (taskbar clock) | `Asia/Jakarta` |
| `date_format` | Taskbar date format | `MM/DD/YYYY` |
| `time_format` | `auto` / `12h` / `24h` | `auto` |
| `table_top/bottom/left/right` | Orders-table crop box | tuned for 1280×719 TWS screenshots |
| `gemini_model` | Gemini Vision model | `gemini-3.5-flash` |

## Output layout

```
archives/
└── 2026-09-25/                      # US trade date
    ├── photo_....jpg                # archived screenshot
    ├── photo_....json / .csv        # raw extraction for that screenshot
    ├── full.json                    # per-image records indexed by processing order (editable source of truth)
    ├── consolidated.json            # deduplicated master ledger (derived from full.json)
    └── full.csv                     # CSV of the consolidated ledger
```

## Design notes

- **Dedup key**: `(us_trade_date, account, symbol, action, price)`. Same key with
  a changed quantity = lifecycle update (`WORKING 0/7` → `FILLED 7`), tracked in
  the record's `history`. Status is always derived from quantity, never used as a
  detector.
- **Two same-price orders for the same symbol/account** (e.g. a filled order plus
  a re-placed working one) share a key — that's why screenshots are fed one at a
  time, in capture order.
- **Priceless orders** (MID / "Price Cap N/A" with no fill yet) record `price: 0.0`
  and get their price from the fill once executed.
