from typing import List
import re
from base_adapter import BaseMarketplaceAdapter
from schema import LaptopListing
from playwright_fetcher import intercept_json, rendered_page_texts
import spec_parser

# Nama operation GQL Tokopedia berubah dari waktu ke waktu (v4 -> v5 -> ...).
# Match longgar di "SearchProduct" biar tahan ganti versi minor.
MATCH_SUBSTR = "SearchProduct"

# Halaman detail produk (PDP) Tokopedia memuat spesifikasi lewat query GQL
# yang beda dari hasil search -- match longgar di "PDP" biar tahan ganti versi.
DETAIL_MATCH_SUBSTR = "PDP"

# Field yang mau dilengkapi dari halaman detail kalau masih kosong dari card.
_DETAIL_FILLABLE_FIELDS = ("storage_gb", "gpu", "screen_size_in", "model")

# Kandidat nama field yang menandakan sebuah dict adalah "kartu produk".
_PRODUCT_MARKER_KEYS = {"name", "product_name", "productName", "title"}
_PRICE_MARKER_KEYS = {"price", "price_int", "priceFmt", "price_str"}


def _find_product_list(node, _depth=0):
    """Cari list of dict yang 'mirip' kartu produk di mana pun posisinya di JSON."""
    if _depth > 12:
        return None
    if isinstance(node, list):
        if node and isinstance(node[0], dict):
            keys = set(node[0].keys())
            if keys & _PRODUCT_MARKER_KEYS and (keys & _PRICE_MARKER_KEYS or "price" in str(keys).lower()):
                return node
        for item in node:
            found = _find_product_list(item, _depth + 1)
            if found:
                return found
    elif isinstance(node, dict):
        for v in node.values():
            found = _find_product_list(v, _depth + 1)
            if found:
                return found
    return None


# Kandidat pasangan key yang menandakan sebuah dict adalah "baris spesifikasi"
# (label + value) di halaman PDP -- nama field GQL-nya belum dikonfirmasi,
# jadi dicoba beberapa pola umum sebelum menyerah dan nge-print debug.
_SPEC_ROW_KEY_PAIRS = [
    ("title", "subtitle"),
    ("label", "value"),
    ("name", "value"),
    ("key", "value"),
]


def _find_spec_list(node, _depth=0):
    """Cari list of dict yang mirip baris spesifikasi produk di JSON PDP."""
    if _depth > 12:
        return None
    if isinstance(node, list):
        if node and isinstance(node[0], dict):
            keys = set(node[0].keys())
            if any({key_a, key_b} <= keys for key_a, key_b in _SPEC_ROW_KEY_PAIRS):
                return node
        for item in node:
            found = _find_spec_list(item, _depth + 1)
            if found:
                return found
    elif isinstance(node, dict):
        for v in node.values():
            found = _find_spec_list(v, _depth + 1)
            if found:
                return found
    return None


def _spec_row_text(row: dict) -> str:
    """Gabungkan label+value 1 baris spesifikasi jadi teks buat di-regex spec_parser."""
    for key_a, key_b in _SPEC_ROW_KEY_PAIRS:
        if key_a in row and key_b in row:
            return f"{row.get(key_a)} {row.get(key_b)}"
    return " ".join(str(v) for v in row.values() if isinstance(v, (str, int, float)))


def _dig(d, *keys, default=None):
    """Ambil nested value dgn aman; kalau bukan dict di tengah jalan, return default."""
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


def _parse_sold(value):
    """Ubah variasi countSold Tokopedia menjadi jumlah unit terjual.

    API Tokopedia dapat memberi integer (`countSold`) atau teks seperti
    `1,2 rb+ terjual`; format terakhir sebelumnya selalu gagal diparse.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)

    text = str(value).lower().replace("terjual", "").replace("+", "").strip()
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(rb|ribu|k)?", text)
    if not match:
        return None
    try:
        number = float(match.group(1).replace(",", "."))
        multiplier = 1000 if match.group(2) in ("rb", "ribu", "k") else 1
        return int(number * multiplier)
    except ValueError:
        return None


def _extract_sold(raw_card: dict):
    """Ambil jumlah terjual dari beberapa versi field respons pencarian."""
    containers = (
        raw_card,
        raw_card.get("stats") or {},
        raw_card.get("productStats") or {},
        raw_card.get("product_stats") or {},
    )
    fields = (
        "countSold", "count_sold", "countSoldFmt", "count_sold_fmt",
        "soldCount", "sold_count", "sold", "historical_sold",
    )
    for container in containers:
        if not isinstance(container, dict):
            continue
        for field in fields:
            parsed = _parse_sold(container.get(field))
            if parsed is not None:
                return parsed

    # SearchProductV5 tidak punya field angka terpisah untuk ini -- "X terjual"
    # muncul sebagai salah satu entry teks di labelGroups (biasanya
    # position="ri_product_credibility").
    for label in raw_card.get("labelGroups") or []:
        title = label.get("title") if isinstance(label, dict) else None
        if title and "terjual" in title.lower():
            parsed = _parse_sold(title)
            if parsed is not None:
                return parsed
    return None


def _sold_from_page_text(text: str):
    """Ambil angka yang secara eksplisit ditampilkan dekat kata 'terjual'."""
    body = text or ""
    match = re.search(r"(\d+(?:[.,]\d+)?\s*(?:rb|ribu|k)?\s*\+?)\s*terjual\b", body, re.I)
    if match:
        return _parse_sold(match.group(1))
    match = re.search(r"\bterjual\s*(\d+(?:[.,]\d+)?\s*(?:rb|ribu|k)?\s*\+?)", body, re.I)
    return _parse_sold(match.group(1)) if match else None


def enrich_sold_counts(listings: List[LaptopListing], headless: bool = True) -> List[LaptopListing]:
    """Isi sold_count kosong dari halaman detail listing Tokopedia.

    Tokopedia sering tidak mengirim statistik ini di API pencarian. Fallback
    ini membaca teks yang terlihat pada halaman detail, bukan menebak angka.
    """
    candidates = [
        listing for listing in listings
        if listing.source == "tokopedia" and listing.sold_count is None and listing.url
    ]
    if not candidates:
        print("[tokopedia] tidak ada sold_count kosong untuk dilengkapi.")
        return listings

    texts = rendered_page_texts([listing.url for listing in candidates], headless=headless)
    filled = 0
    for listing, text in zip(candidates, texts):
        sold_count = _sold_from_page_text(text)
        if sold_count is not None:
            listing.sold_count = sold_count
            filled += 1
    print(f"[tokopedia] sold_count halaman detail: filled={filled}/{len(candidates)}")
    return listings


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


class TokopediaAdapter(BaseMarketplaceAdapter):
    source_name = "tokopedia"

    def fetch_search_page(self, keyword: str, page: int) -> dict:
        url = f"https://www.tokopedia.com/search?q={keyword.replace(' ', '+')}&page={page}"
        return intercept_json(url, MATCH_SUBSTR) or {}

    def fetch_detail_page(self, url: str) -> dict:
        """Ambil spesifikasi dari halaman detail produk (PDP) Tokopedia.

        Beda dari fetch_search_page: ini buka URL produk langsung, bukan URL
        pencarian, lalu intercept response GQL yang memuat blok "Spesifikasi".
        """
        if not url:
            return {}

        raw = intercept_json(url, DETAIL_MATCH_SUBSTR)
        body = raw[0] if isinstance(raw, list) else raw
        if not isinstance(body, dict):
            return {}

        spec_rows = _find_spec_list(body.get("data", body))
        if spec_rows is None:
            print("[tokopedia][detail] tidak nemu list spesifikasi. top-level keys:", list(body.keys()))
            if isinstance(body.get("data"), dict):
                print("[tokopedia][detail] data keys:", list(body["data"].keys()))
            return {}

        if not getattr(self, "_dumped_detail_sample", False):
            print("[tokopedia][detail][debug] contoh spec row keys:", list(spec_rows[0].keys()))
            print("[tokopedia][detail][debug] contoh spec rows (maks 5):", spec_rows[:5])
            self._dumped_detail_sample = True

        spec_text = " | ".join(_spec_row_text(row) for row in spec_rows if isinstance(row, dict))
        return {
            "storage_gb": spec_parser.extract_storage_gb(spec_text),
            "gpu": spec_parser.extract_gpu(spec_text),
            "screen_size_in": spec_parser.extract_screen_size(spec_text),
            "model": spec_parser.extract_model(spec_text),
        }

    def parse_listing_cards(self, raw_page) -> List[dict]:
        body = raw_page[0] if isinstance(raw_page, list) else raw_page
        if not isinstance(body, dict):
            return []

        products = _find_product_list(body.get("data", body))
        if products is None:
            print("[tokopedia] tidak nemu list produk. top-level keys:", list(body.keys()))
            if isinstance(body.get("data"), dict):
                print("[tokopedia] data keys:", list(body["data"].keys()))
            return []
        return products

    def normalize(self, raw_card: dict) -> LaptopListing:
        if not getattr(self, "_dumped_sample", False):
            print("[tokopedia][debug] contoh raw_card keys:", list(raw_card.keys()))
            self._dumped_sample = True

        title = raw_card.get("name") or raw_card.get("product_name") or raw_card.get("title") or ""
        shop = raw_card.get("shop") or {}

        if not getattr(self, "_dumped_shop_sample", False) and shop:
            print("[tokopedia][debug] contoh shop keys:", list(shop.keys()))
            self._dumped_shop_sample = True

        # harga: coba beberapa bentuk umum -- dict {"number": ...}, int langsung, atau string berformat
        price_raw = raw_card.get("price")
        if isinstance(price_raw, dict):
            price_idr = _to_int(price_raw.get("number") or price_raw.get("price_int"))
        else:
            price_idr = _to_int(price_raw)

        listing = LaptopListing(
            source=self.source_name,
            source_id=str(raw_card.get("id") or raw_card.get("product_id") or ""),
            url=raw_card.get("url") or raw_card.get("product_url"),
            title=title,
            brand=spec_parser.extract_brand(title),
            model=None,
            cpu=spec_parser.extract_cpu(title),
            ram_gb=spec_parser.extract_ram_gb(title),
            storage_gb=spec_parser.extract_storage_gb(title),
            gpu=None,
            screen_size_in=spec_parser.extract_screen_size(title),
            price_idr=price_idr,
            original_price_idr=_to_int(raw_card.get("original_price")),
            condition="used" if "bekas" in title.lower() or "second" in title.lower() else "new",
            seller_name=_dig(shop, "name"),
            seller_rating=_to_float(raw_card.get("rating")),
            seller_num_reviews=_to_int(raw_card.get("countReview") or raw_card.get("count_review")),
            seller_is_official=bool(_dig(shop, "isOfficial") or _dig(shop, "goldmerchant")),
            sold_count=_extract_sold(raw_card),
            location=_dig(shop, "city"),
        )

        if getattr(self, "_enrich_detail", False) and listing.url:
            self._enrich_from_detail_page(listing)

        return listing

    def _enrich_from_detail_page(self, listing: LaptopListing) -> None:
        """Isi field yang masih kosong dari halaman detail produk, tanpa nimpa yang sudah ada."""
        missing = [field for field in _DETAIL_FILLABLE_FIELDS if getattr(listing, field) is None]
        if not missing:
            return
        self._throttle()
        detail = self.fetch_detail_page(listing.url)
        for field in missing:
            value = detail.get(field)
            if value is not None:
                setattr(listing, field, value)
