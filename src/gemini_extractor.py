import json
import os
import time
from pathlib import Path
from typing import Sequence
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

try:
    from .table_extractor import Transaction
except (ImportError, ValueError):
    from table_extractor import Transaction


class AuditedOrder(BaseModel):
    symbol: str = Field(
        description="Clean uppercase stock ticker symbol (e.g. ULTA, AAPL, ODFL, XLB, SOXX, STX). "
        "Verify that no letters are clipped or misread, and strip any leading UI dashes or bullets."
    )
    account: str = Field(
        description="Interactive Brokers account ID matching U followed by 8 digits (e.g. U14984242, U18210401)."
    )
    action: str = Field(
        description="Order action: strictly either 'BUY' or 'SELL'. "
        "NOTE: Badges are not always present! BUY may appear as plain text 'BUY' or with a blue badge. "
        "SELL may appear as 'SELL', truncated 'S...', 'S.', or 'S', with or without a red badge. "
        "Every stock order in this table is strictly BUY or SELL."
    )
    details: str = Field(
        default="",
        description="The literal order parameter shown in the Details column. "
        "For limit orders, provide the numeric limit (e.g. '1086.90', '220.43', '164.60'). "
        "For midpoint or price cap orders, provide 'Price Cap'. "
        "DO NOT copy the Fill Price here!"
    )
    quantity: str = Field(
        description="Order quantity string: '0/N' for working/unfilled, 'M/N' for partially filled, "
        "or integer 'N' for fully filled. Check carefully for any digits clipped by column boundaries."
    )
    fill_price: float | None = Field(
        default=None,
        description="Executed average fill price from the Fill Px column, or null if unexecuted / 0.00 / '-'."
    )


class AuditResult(BaseModel):
    orders: list[AuditedOrder] = Field(
        description="Complete list of audited and corrected orders matching visual order from top to bottom."
    )


def get_gemini_client() -> genai.Client | None:
    """Returns an initialized GenAI client if GEMINI_API_KEY is available."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(client_args={"timeout": 45.0}),
    )


def validate_and_fill_with_gemini(
    crop_image: Image.Image,
    draft_transactions: Sequence[Transaction],
    image_name: str,
    us_trade_date: str,
    model_name: str = "gemini-3.5-flash",
) -> list[Transaction]:
    """Gemini Vision validator and gap/wrong/missing filler.

    Runs on every image: feeds the preliminary JSON draft extracted by local OCR plus
    the cropped table image into Gemini Vision, which audits the table row-by-row and
    returns a corrected list. This catches not only explicit gaps (UNKNOWN action,
    missing symbol) but also silently wrong values (misread price digits, clipped
    quantities, misread accounts) and rows local OCR dropped entirely.
    """
    client = get_gemini_client()
    if not client:
        print("  [Gemini Vision] GEMINI_API_KEY not configured. Falling back to local heuristics.")
        return _apply_local_sell_fallback(draft_transactions)

    print(f"  [Gemini Vision] Auditing draft ({len(draft_transactions)} rows) against image to validate and fix gap/wrong/missing values...")

    # Serialize local OCR draft into clean JSON for Gemini
    draft_data = [
        {
            "row": idx,
            "symbol": t.symbol,
            "account": t.account,
            "action": t.action,
            "details": getattr(t, "details", t.price),
            "quantity": t.quantity,
            "fill_price": t.fill_price,
        }
        for idx, t in enumerate(draft_transactions, 1)
    ]
    draft_json_str = json.dumps(draft_data, indent=2)

    prompt = (
        "You are an expert financial OCR auditor for Interactive Brokers (TWS) desktop screenshots.\n"
        "Attached is:\n"
        "1. The cropped screenshot of the transaction orders table.\n"
        "2. The preliminary JSON draft extracted by local OCR:\n\n"
        f"```json\n{draft_json_str}\n```\n\n"
        "Your task: Carefully compare the JSON draft row-by-row against the table in the image from top to bottom.\n\n"
        "FIELD-BY-FIELD AUDIT GUIDELINES:\n"
        "- Action (BUY vs SELL):\n"
        "  * IMPORTANT: Rows DO NOT always have a red or blue badge! Badges are optional.\n"
        "  * BUY orders appear as 'BUY' (plain white text or with a blue badge).\n"
        "  * SELL orders appear as 'SELL', truncated 'S...', 'S.', or 'S' (plain text or with a red badge).\n"
        "  * Every stock order in this table is strictly either BUY or SELL. Resolve any row where action is 'UNKNOWN'.\n"
        "- Symbol:\n"
        "  * Verify the full ticker symbol. Ensure no letters were clipped or misread (e.g. STX, SOXX, ODFL, XLB).\n"
        "  * Strip any leading UI bullets, play icons, or dashes (e.g. '- HD' -> 'HD').\n"
        "- Account:\n"
        "  * Must match 'U' followed by 8 digits (e.g. U14984242, U18210401). Correct any misread digits.\n"
        "- Quantity:\n"
        "  * Verify the exact quantity string. Check if any digits were clipped at column dividers (e.g. '11/12' vs '11/1').\n"
        "  * Format: '0/N' for working, 'M/N' for partial, or integer 'N' for filled.\n"
        "- Details & Fill Price:\n"
        "  * Details: The order instruction / limit or stop parameter shown in the Details column.\n"
        "    - If it is a limit or stop order (e.g. 'LMT 1086.90', 'STP 868.95'), record the numeric price as a string (e.g. '1086.90', '868.95').\n"
        "    - If it shows non-numeric text like 'Price Cap N...' or 'MKT', record 'Price Cap' or 'MKT'. DO NOT put the fill price in Details!\n"
        "  * Fill Price: float from Fill Px column (the rightmost price column; the actual execution price), or null if unexecuted (shown as '-' or '0.00').\n"
        "    - NOTE: Columns on the right are: Quantity | Aux. Px | Fill Px. DO NOT take numbers from Aux. Px!\n"
        "    - For WORKING orders ('0/N'), Fill Price is ALWAYS null.\n"
        "  * If a fill price is truncated with trailing dots (e.g. '1832....', '1085....'), record the visible number (e.g. 1832.0, 1085.0).\n"
        "- Table edges (IMPORTANT):\n"
        "  * This image is a CROP of a larger TWS window. The table does NOT necessarily start or end at the image edges.\n"
        "  * A row touching the top or bottom edge may be CUT OFF: half-visible text, clipped digits, missing decimals (e.g. '77.5' shown when the real value is '77.25').\n"
        "  * If any record is cut off or only partially visible at the top or bottom edge, DO NOT include it and DO NOT guess its values. Exclude it entirely.\n"
        "- Completeness:\n"
        "  * The draft may be incomplete: if local OCR missed entire rows, add them — but ONLY rows that are fully visible with every column readable.\n"
        "  * If the draft contains a row that does not exist in the image, remove it.\n\n"
        "Return the complete, corrected list of orders in visual order from top to bottom."
    )

    candidate_models = [model_name, "gemini-3.8-flash", "gemini-3-flash-preview"]
    seen = set()
    candidate_models = [m for m in candidate_models if not (m in seen or seen.add(m))]

    for model in candidate_models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[crop_image, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AuditResult,
                        temperature=0.0,
                        # This pipeline does no function calling; disable AFC so
                        # the SDK skips its AFC path (and its deprecation warning)
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                audit: AuditResult = AuditResult.model_validate_json(response.text)
                print(f"  [Gemini Vision] Successfully audited and verified {len(audit.orders)} orders using {model}!")

                # Convert to standard Transaction dataclasses with quantity-driven status
                audited_transactions = []
                for o in audit.orders:
                    # Determine status strictly from quantity (never from status icons)
                    if "/" in o.quantity:
                        parts = o.quantity.split("/")
                        status = "WORKING" if int(parts[0]) == 0 else "PARTIAL"
                    else:
                        status = "FILLED"

                    details_str = str(o.details or "").strip()
                    if "Price Cap" in details_str or "Cap" in details_str:
                        details_str = "Price Cap"
                    elif details_str.startswith("LMT"):
                        details_str = details_str.replace("LMT", "").strip()
                    elif details_str.startswith("STP"):
                        details_str = details_str.replace("STP", "").strip()

                    try:
                        numeric_px = float(details_str)
                    except ValueError:
                        numeric_px = o.fill_price if (o.fill_price and o.fill_price > 0) else 0.0

                    # For working orders, fill_price is strictly None
                    final_fill = None if status == "WORKING" else (o.fill_price if (o.fill_price and o.fill_price > 0) else None)

                    audited_transactions.append(
                        Transaction(
                            image_name=image_name,
                            us_trade_date=us_trade_date,
                            account=o.account,
                            symbol=o.symbol.upper(),
                            action=o.action.upper(),
                            quantity=o.quantity,
                            details=details_str,
                            price=numeric_px,
                            fill_price=final_fill,
                            status=status,
                            raw_text=f"Audited: {o.symbol} | {o.account} | {o.action} | {details_str} | {o.quantity}",
                        )
                    )
                return audited_transactions

            except Exception as e:
                err_msg = str(e)
                if (
                    "503" in err_msg
                    or "UNAVAILABLE" in err_msg
                    or "429" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                    or "timeout" in err_msg.lower()
                    or "timed out" in err_msg.lower()
                ):
                    sleep_time = 2.0 * (attempt + 1)
                    print(f"  [Gemini Vision] {model} busy/timeout (attempt {attempt+1}/3). Retrying in {sleep_time:g}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"  [Gemini Vision] {model} notice: {err_msg[:80]}...")
                    break  # Try next candidate model

    print("  [Gemini Vision] API temporarily unavailable; applying local deterministic SELL fallback.")
    return _apply_local_sell_fallback(draft_transactions)


def _apply_local_sell_fallback(transactions: Sequence[Transaction]) -> list[Transaction]:
    """Local fallback: In stock trading, orders that do not say BUY are SELL."""
    fixed = []
    for t in transactions:
        if t.action == "UNKNOWN":
            t.action = "SELL"
        fixed.append(t)
    return fixed
