"""Normalisasi atribut laptop dari CSV sebelum proses deduplikasi."""
import argparse
import csv
import re
from pathlib import Path

import spec_parser
from schema import LaptopListing


TEXT_FIELDS = ("brand", "model", "cpu", "gpu")
NUMBER_FIELDS = ("ram_gb", "storage_gb", "screen_size_in")


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("–", "-").strip())


def _normalise_number(value: str, decimal: bool = False) -> str:
    try:
        number = float(_clean_text(value).replace(",", "."))
    except ValueError:
        return ""
    return str(number) if decimal and not number.is_integer() else str(int(number))


def normalise_row(row: dict[str, str]) -> None:
    title = _clean_text(row.get("title", ""))
    row["title"] = title
    extracted = {
        "brand": spec_parser.extract_brand(title),
        "model": spec_parser.extract_model(title),
        "cpu": spec_parser.extract_cpu(title),
        "ram_gb": spec_parser.extract_ram_gb(title),
        "storage_gb": spec_parser.extract_storage_gb(title),
        "gpu": spec_parser.extract_gpu(title),
        "screen_size_in": spec_parser.extract_screen_size(title),
    }
    for field, value in extracted.items():
        row[field] = str(value) if value is not None else _clean_text(row.get(field, ""))
    for field in TEXT_FIELDS:
        row[field] = _clean_text(row.get(field, ""))
    for field in NUMBER_FIELDS:
        row[field] = _normalise_number(row.get(field, ""), decimal=field == "screen_size_in")


def normalise_listing(listing: LaptopListing) -> LaptopListing:
    """Normalisasi listing yang baru discrape tanpa serialisasi CSV terlebih dulu."""
    listing.title = _clean_text(listing.title)
    extracted = {
        "brand": spec_parser.extract_brand(listing.title),
        "model": spec_parser.extract_model(listing.title),
        "cpu": spec_parser.extract_cpu(listing.title),
        "ram_gb": spec_parser.extract_ram_gb(listing.title),
        "storage_gb": spec_parser.extract_storage_gb(listing.title),
        "gpu": spec_parser.extract_gpu(listing.title),
        "screen_size_in": spec_parser.extract_screen_size(listing.title),
    }
    for field, value in extracted.items():
        if value is not None:
            setattr(listing, field, value)
    return listing


def normalise_listings(listings: list[LaptopListing]) -> list[LaptopListing]:
    return [normalise_listing(listing) for listing in listings]


def process(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "title" not in reader.fieldnames:
            raise ValueError("CSV harus memiliki kolom 'title'.")
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        normalise_row(row)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalisasi brand, model, dan spesifikasi laptop dalam CSV.")
    parser.add_argument("--input", default="laptop_listings_all_cleaned.csv")
    parser.add_argument("--output", default="laptop_listings_all_cleaned_normalized.csv")
    args = parser.parse_args()
    print(f"normalized={process(Path(args.input), Path(args.output))}")
