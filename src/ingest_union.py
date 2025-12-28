#!/usr/bin/env python3
"""
ingest_union.py 
Reads all Excel files from data/raw/, standardizes column names into a canonical schema
that does NOT rely on business_id, unions them into one table, and writes a parquet file.

Run:
  python src/ingest_union.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PARQUET = PROCESSED_DIR / "reviews.parquet"


# ----------------------------
# Canonical schema (NO business_id)
# ----------------------------
CANONICAL_COLS = [
    "business_name",
    "address",             # optional
    "location",            # city/state or displayLocation
    "date",
    "review_text",
    "review_rating",       # optional
    "review_count_business",  # optional (business-level total review count if present)
    "outlet_key",          # derived identifier from name + location (+address)
    "source_file",
]


# ----------------------------
# Helpers
# ----------------------------
def _normalize_colname(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def _find_first_present(df_cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df_cols:
            return c
    return None


def _build_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map raw column names -> canonical names.
    We intentionally allow many variants because each Excel file can differ.
    """
    cols = list(df.columns)
    mapping: Dict[str, str] = {}

    # Candidate lists (already normalized in df.columns)
    cand_business_name = ["business_name", "name", "business", "restaurant_name", "biz_name", "outlet_name"]
    cand_address = ["address", "street_address", "full_address", "addr"]
    cand_location = ["displaylocation", "display_location", "location", "city_state", "city", "state"]
    cand_date = ["date", "review_date", "created_at", "time", "timestamp"]
    cand_text = ["comment", "review_text", "review", "text", "content", "body"]
    cand_rating = ["reviewer_rating", "rating", "stars", "review_stars", "user_rating"]
    cand_review_count_business = ["number_of_reviews", "num_reviews", "review_count", "reviewcount", "total_reviews"]

    def set_map(canonical: str, found: Optional[str]) -> None:
        if found is not None:
            mapping[found] = canonical

    set_map("business_name", _find_first_present(cols, cand_business_name))
    set_map("address", _find_first_present(cols, cand_address))
    set_map("location", _find_first_present(cols, cand_location))
    set_map("date", _find_first_present(cols, cand_date))
    set_map("review_text", _find_first_present(cols, cand_text))
    set_map("review_rating", _find_first_present(cols, cand_rating))
    set_map("review_count_business", _find_first_present(cols, cand_review_count_business))

    return mapping


def _clean_str_series(s: pd.Series) -> pd.Series:
    # normalize whitespace and lowercase for stable keys; keep original columns unchanged elsewhere
    return (
        s.astype("string")
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _make_outlet_key(df: pd.DataFrame) -> pd.Series:
    """
    Create a stable outlet identifier from name + location (+ address if present).
    This replaces business_id in downstream logic.
    """
    name = _clean_str_series(df.get("business_name", pd.Series([""] * len(df))))
    loc = _clean_str_series(df.get("location", pd.Series([""] * len(df))))
    addr = _clean_str_series(df.get("address", pd.Series([""] * len(df))))

    # Use address only if it exists and has signal; otherwise name+location is fine
    has_addr_signal = addr.str.len() > 0
    key = name.str.lower() + " | " + loc.str.lower()
    key = key.where(~has_addr_signal, name.str.lower() + " | " + loc.str.lower() + " | " + addr.str.lower())

    # If name is missing, fall back to location+address (last resort)
    name_missing = name.str.len() == 0
    key = key.where(~name_missing, loc.str.lower() + " | " + addr.str.lower())

    return key


def _standardize_one_file(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [_normalize_colname(c) for c in df.columns]

    mapping = _build_column_mapping(df)
    df = df.rename(columns=mapping)

    # Ensure canonical columns exist
    for col in CANONICAL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # Add source file
    df["source_file"] = path.name

    # Parse types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    df["review_count_business"] = pd.to_numeric(df["review_count_business"], errors="coerce")
    df["review_text"] = df["review_text"].astype("string")
    df["business_name"] = df["business_name"].astype("string")
    df["location"] = df["location"].astype("string")
    df["address"] = df["address"].astype("string")

    # Create outlet_key
    df["outlet_key"] = _make_outlet_key(df)

    # Keep only canonical columns in stable order
    df = df[CANONICAL_COLS].copy()

    return df, mapping


def _print_mapping_report(file_name: str, mapping: Dict[str, str]) -> None:
    if not mapping:
        print(f"  - {file_name}: No columns mapped (all canonical cols will be NA).")
        return

    inv: Dict[str, List[str]] = {}
    for raw, canon in mapping.items():
        inv.setdefault(canon, []).append(raw)

    print(f"  - {file_name} column mapping:")
    for canon in ["business_name", "address", "location", "date", "review_text", "review_rating", "review_count_business"]:
        raw_cols = inv.get(canon, [])
        print(f"      {canon:>20} <= {raw_cols if raw_cols else 'MISSING'}")


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

    excel_files = sorted(list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.xls")))
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in: {RAW_DIR} (expected .xlsx/.xls)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs: List[pd.DataFrame] = []
    print(f"Found {len(excel_files)} raw Excel file(s) in {RAW_DIR}:\n")

    for p in excel_files:
        df_std, mapping = _standardize_one_file(p)
        _print_mapping_report(p.name, mapping)
        print(f"      rows loaded: {len(df_std):,}\n")
        all_dfs.append(df_std)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Drop empty review_text
    before = len(combined)
    combined = combined.dropna(subset=["review_text"])
    combined = combined[combined["review_text"].astype("string").str.strip() != ""]
    after = len(combined)

    # Dedupe using outlet_key + date + review_text (robust without business_id)
    if combined["outlet_key"].notna().any():
        combined["_dedupe_key"] = (
            combined["outlet_key"].fillna("")
            + "|"
            + combined["date"].astype("string").fillna("")
            + "|"
            + combined["review_text"].fillna("")
        )
        combined = combined.drop_duplicates(subset=["_dedupe_key"]).drop(columns=["_dedupe_key"])

    print(f"Combined rows: {before:,} -> {after:,} after dropping empty review_text")
    print(f"Final rows after dedupe (if applicable): {len(combined):,}")

    combined.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote: {OUT_PARQUET}")

    # Quick preview
    print("\nPreview (first 5 rows):")
    with pd.option_context("display.max_columns", 50, "display.width", 120):
        print(combined.head(5))


if __name__ == "__main__":
    main()
