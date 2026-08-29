from typing import List
from base_adapter import BaseMarketplaceAdapter
from schema import LaptopListing
from playwright_fetcher import intercept_json
from json_utils import find_product_list, dig, to_float, to_int, parse_sold_text
import spec_parser

# Blibli SPA -- endpoint internal belum diverifikasi (belum ada akses live
# buat inspect). Match longgar di "search" dulu; kalau kebanyakan noise
# (banyak request lain ikut match), persempit ke substring lebih spesifik
# setelah lihat hasil debug_capture.py di URL "https://www.blibli.com/cari/laptop%20asus".
MATCH_SUBSTR = "search"


class BlibliAdapter(BaseMarketplaceAdapter):
    source_name = "blibli"

    def fetch_search_page(self, keyword: str, page: int) -> dict:
        url = f"https://www.blibli.com/cari/{keyword.replace(' ', '%20')}?page={page}"
        return intercept_json(url, MATCH_SUBSTR) or {}

    def parse_listing_cards(self, raw_page) -> List[dict]:
        body = raw_page[0] if isinstance(raw_page, list) else raw_page
        if not isinstance(body, dict):
            return []

        products = find_product_list(body.get("data", body))
        if products is None:
            print("[blibli] tidak nemu list produk. top-level keys:", list(body.keys()))
            if isinstance(body.get("data"), dict):
                print("[blibli] data keys:", list(body["data"].keys()))
            return []
        return products

    def normalize(self, raw_card: dict) -> LaptopListing:
        if not getattr(self, "_dumped_sample", False):
            print("[blibli][debug] contoh raw_card keys:", list(raw_card.keys()))
            self._dumped_sample = True

        title = (raw_card.get("name") or raw_card.get("product_name")
                 or raw_card.get("title") or raw_card.get("itemName") or "")

        price_raw = (raw_card.get("price") or raw_card.get("priceValue")
                     or raw_card.get("salePrice"))
        if isinstance(price_raw, dict):
            price_idr = to_int(price_raw.get("value") or price_raw.get("amount") or price_raw.get("number"))
        else:
            price_idr = to_int(price_raw)

        merchant = raw_card.get("merchant") or raw_card.get("shop") or {}

        return LaptopListing(
            source=self.source_name,
            source_id=str(raw_card.get("id") or raw_card.get("sku") or raw_card.get("itemSku") or ""),
            url=raw_card.get("url") or raw_card.get("productUrl"),
            title=title,
            brand=spec_parser.extract_brand(title),
            model=None,
            cpu=spec_parser.extract_cpu(title),
            ram_gb=spec_parser.extract_ram_gb(title),
            storage_gb=spec_parser.extract_storage_gb(title),
            gpu=None,
            screen_size_in=spec_parser.extract_screen_size(title),
            price_idr=price_idr,
            original_price_idr=to_int(raw_card.get("originalPrice") or raw_card.get("strikethroughPrice")),
            condition="used" if "bekas" in title.lower() or "second" in title.lower() else "new",
            seller_name=dig(merchant, "name") or raw_card.get("merchantName"),
            seller_rating=to_float(raw_card.get("rating") or raw_card.get("productRating")),
            seller_num_reviews=to_int(raw_card.get("reviewCount") or raw_card.get("countReview")),
            seller_is_official=bool(dig(merchant, "isOfficial") or raw_card.get("isOfficialStore")),
            sold_count=parse_sold_text(raw_card.get("soldCount") or raw_card.get("countSoldFmt")),
            location=dig(merchant, "location") or raw_card.get("merchantCity"),
        )
