from typing import List
from schema import LaptopListing

# Harga pasar wajar (median kasar, IDR) per brand -- kalibrasi ulang berkala.
MIN_PLAUSIBLE_PRICE_IDR = {
    "Apple": 8_000_000,
    "Asus": 3_500_000,
    "Lenovo": 3_000_000,
    "Hp": 3_000_000,
    "Acer": 2_800_000,
    "Dell": 3_500_000,
    "Msi": 5_000_000,
}
DEFAULT_MIN_PRICE = 2_500_000

MIN_SELLER_REVIEWS = 5
MIN_SELLER_RATING = 4.0


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def evaluate(listing: LaptopListing) -> LaptopListing:
    reasons = []

    price = _to_int(listing.price_idr)
    original_price = _to_int(listing.original_price_idr)
    rating = _to_float(listing.seller_rating)
    num_reviews = _to_int(listing.seller_num_reviews)
    sold = _to_int(listing.sold_count)

    floor = MIN_PLAUSIBLE_PRICE_IDR.get(listing.brand, DEFAULT_MIN_PRICE)
    if price and price < floor * 0.4:
        reasons.append(f"price_far_below_market(<{floor})")

    if num_reviews is not None and num_reviews < MIN_SELLER_REVIEWS \
            and not listing.seller_is_official:
        reasons.append("new_seller_low_reviews")

    if rating is not None and rating < MIN_SELLER_RATING \
            and num_reviews and num_reviews > 10:
        reasons.append("low_seller_rating")

    if original_price and price:
        discount_pct = 1 - (price / original_price)
        if discount_pct > 0.75:
            reasons.append("unrealistic_discount")

    if sold is not None and sold > 50 \
            and num_reviews is not None and num_reviews < 3:
        reasons.append("sold_review_mismatch")

    listing.is_suspected_scam = len(reasons) > 0
    listing.scam_reasons = reasons
    return listing


def filter_listings(listings: List[LaptopListing], drop_scams: bool = False) -> List[LaptopListing]:
    evaluated = [evaluate(l) for l in listings]
    if drop_scams:
        return [l for l in evaluated if not l.is_suspected_scam]
    return evaluated
