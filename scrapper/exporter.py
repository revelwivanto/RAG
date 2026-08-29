import csv
import json
from typing import List
from schema import LaptopListing


def export_json(listings: List[LaptopListing], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([l.to_dict() for l in listings], f, ensure_ascii=False, indent=2)


def export_csv(listings: List[LaptopListing], path: str):
    if not listings:
        return
    fieldnames = list(listings[0].to_dict().keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for l in listings:
            row = l.to_dict()
            row["scam_reasons"] = ";".join(row["scam_reasons"])
            writer.writerow(row)
