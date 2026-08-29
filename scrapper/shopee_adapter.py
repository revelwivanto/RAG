from typing import List
from base_adapter import BaseMarketplaceAdapter
from schema import LaptopListing
from playwright_fetcher import intercept_json
import spec_parser


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


class ShopeeAdapter(BaseMarketplaceAdapter):
    source_name = "shopee"

    SHOPEE_COOKIES = [
        {"name": "language", "value": "id", "domain": ".shopee.co.id", "path": "/"},
        {"name": "shopee_webUnique_ccd", "value": "%7B%7D", "domain": ".shopee.co.id", "path": "/"},
    ]

    def fetch_search_page(self, keyword: str, page: int) -> dict:
        url = f"https://shopee.co.id/search?keyword={keyword.replace(' ', '%20')}&page={page - 1}"
        return intercept_json(
            url,
            "search_items",
            pre_goto_cookies=self.SHOPEE_COOKIES,
            post_goto_click_text="Bahasa Indonesia",
        ) or {}

    def parse_listing_cards(self, raw_page) -> List[dict]:
        items = raw_page.get("items", []) if raw_page else []
        return [it.get("item_basic", {}) for it in items if it.get("item_basic")]

    def normalize(self, raw_card: dict) -> LaptopListing:
        title = raw_card.get("name", "")
        item_rating = raw_card.get("item_rating") or {}
        rating_count = item_rating.get("rating_count") or []
        price = _to_int(raw_card.get("price"))
        price_before = _to_int(raw_card.get("price_before_discount"))

        return LaptopListing(
            source=self.source_name,
            source_id=str(raw_card.get("itemid")),
            url=_build_shopee_url(raw_card),
            title=title,
            brand=spec_parser.extract_brand(title),
            model=None,
            cpu=spec_parser.extract_cpu(title),
            ram_gb=spec_parser.extract_ram_gb(title),
            storage_gb=spec_parser.extract_storage_gb(title),
            gpu=None,
            screen_size_in=spec_parser.extract_screen_size(title),
            price_idr=price // 100000 if price else None,
            original_price_idr=price_before // 100000 if price_before else None,
            condition="used" if "bekas" in title.lower() or "second" in title.lower() else "new",
            seller_name=str(raw_card.get("shopid")) if raw_card.get("shopid") else None,
            seller_rating=_to_float(item_rating.get("rating_star")),
            seller_num_reviews=_to_int(rating_count[0]) if rating_count else None,
            seller_is_official=bool(raw_card.get("shopee_verified") or raw_card.get("is_official_shop")),
            sold_count=_to_int(raw_card.get("historical_sold") or raw_card.get("sold")),
            location=raw_card.get("shop_location"),
        )


def _build_shopee_url(raw_card: dict) -> str:
    shopid = raw_card.get("shopid")
    itemid = raw_card.get("itemid")
    name_slug = raw_card.get("name", "").replace(" ", "-")
    return f"https://shopee.co.id/{name_slug}-i.{shopid}.{itemid}"
