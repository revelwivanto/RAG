from typing import List
from schema import LaptopListing

# Kata kunci yang menandakan listing itu bukan unit laptop utuh.
REJECT_KEYWORDS = [
    "keyboard", "adaptor", "adapter", "charger", "casing", "case ", "tas laptop",
    "sarung laptop", "skin laptop", "stiker laptop", "sticker", "cooling pad", "coolpad",
    "mouse", "mousepad", "stand laptop", "engsel", "hinge", "baterai laptop",
    "battery laptop", "lcd laptop", "layar laptop", "kabel laptop", "speaker laptop",
    "cooler laptop", "kipas laptop", "fan laptop", "backpack", "sleeve laptop",
    "cover laptop", "hardcase", "softcase", "docking", "hub usb", "adaptor charger",
    "kabel charger", "kabel power", "sparepart", "motherboard", "mainboard",
    "ram sodimm", "ssd m.2", "hard disk", "processor", "monitor", "tv ",
    "televisi", "printer", "proyektor", "router", "modem", "switch", "mini pc",
    "desktop", "all in one", "aiox", "tablet", "ipad", "handphone", "smartphone",
]
LAPTOP_SIGNALS = [
    "laptop", "notebook", "macbook", "chromebook", "thinkpad", "ideapad", "vivobook",
    "zenbook", "rog", "tuf", "legion", "yoga", "aspire", "swift", "nitro", "predator",
    "inspiron", "latitude", "elitebook", "probook", "pavilion", "victus", "omen", "xps",
    "katana", "cyborg", "axioo",
]


def is_non_laptop(title: str) -> bool:
    text = (title or "").lower()
    if any(keyword in text for keyword in REJECT_KEYWORDS):
        return True
    return not any(signal in text for signal in LAPTOP_SIGNALS)


def filter_non_laptops(listings: List[LaptopListing]) -> List[LaptopListing]:
    kept = [listing for listing in listings if not is_non_laptop(listing.title)]
    dropped = len(listings) - len(kept)
    print(f"filtered out {dropped} non-laptop listings")
    return kept


# Kompatibilitas untuk pemanggil lama.
filter_accessories = filter_non_laptops
