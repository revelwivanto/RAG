"""Hapus duplikasi listing dari hasil CSV tanpa menggabungkan marketplace berbeda."""
import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

from schema import LaptopListing


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def dedupe_key(row: dict[str, str]) -> tuple[str, ...]:
    source = _normalise(row.get("source", ""))
    source_id = _normalise(row.get("source_id", ""))
    product_fields = (
        "brand", "model", "cpu", "ram_gb", "storage_gb", "gpu",
        "screen_size_in", "condition",
    )
    fingerprint = tuple(_normalise(row.get(field, "")) for field in product_fields)
    if all(fingerprint[:2]):
        # Jangan gabungkan listing identik dari marketplace yang berbeda.
        return ("product", source, *fingerprint)
    if source_id:
        return ("id", source, source_id)
    return (
        "fallback",
        source,
        _normalise(row.get("title", "")),
        _normalise(row.get("price_idr", "")),
        _normalise(row.get("seller_name", "")),
    )


def deduplicate_listings(listings: Iterable[LaptopListing]) -> list[LaptopListing]:
    """Versi in-memory dari deduplikasi CSV untuk pipeline scraping utama."""
    items = list(listings)
    unique = []
    seen = set()
    for listing in items:
        key = dedupe_key({
            "source": listing.source,
            "source_id": listing.source_id,
            "title": listing.title,
            "price_idr": str(listing.price_idr or ""),
            "seller_name": listing.seller_name or "",
            "brand": listing.brand or "",
            "model": listing.model or "",
            "cpu": listing.cpu or "",
            "ram_gb": str(listing.ram_gb or ""),
            "storage_gb": str(listing.storage_gb or ""),
            "gpu": listing.gpu or "",
            "screen_size_in": str(listing.screen_size_in or ""),
            "condition": listing.condition,
        })
        if key not in seen:
            seen.add(key)
            unique.append(listing)
    print(f"deduplicated: removed={len(items) - len(unique)} unique={len(unique)}")
    return unique


def process(input_path: Path, output_path: Path) -> tuple[int, int]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if not reader.fieldnames:
            raise ValueError("CSV tidak memiliki header.")
        seen = set()
        unique_rows = []
        total = 0
        for row in reader:
            total += 1
            key = dedupe_key(row)
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    return total, len(unique_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hapus duplikasi listing dalam CSV.")
    parser.add_argument("--input", default="laptop_listings_all_cleaned_normalized.csv")
    parser.add_argument("--output", default="laptop_listings_all_cleaned_deduplicated.csv")
    args = parser.parse_args()

    total, unique = process(Path(args.input), Path(args.output))
    print(f"processed={total} removed_duplicates={total - unique} unique={unique}")
