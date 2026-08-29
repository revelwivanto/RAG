from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from base_adapter import BaseMarketplaceAdapter
from schema import LaptopListing
import spec_parser

# PENTING: robots.txt Bhinneka melarang automated fetch di sejumlah path.
# Cek Terms of Service / hubungi Bhinneka soal akses data resmi (API/B2B feed)
# sebelum pakai ini untuk scraping produksi -- terutama untuk konteks
# procurement bank yang butuh jejak kepatuhan bersih.
#
# Selector CSS di bawah ini PLACEHOLDER -- isi dari hasil inspect element
# manual di browser kamu (klik kanan produk -> Inspect), karena situsnya
# server-rendered (bukan API JSON kayak Tokopedia/Shopee).


import re
import time
from urllib.parse import urlencode


def _to_int(v):
    if v is None:
        return None
    digits = re.sub(r"[^\d]", "", str(v))
    return int(digits) if digits else None


def _to_float(v):
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _extract_name(card):
    """Coba beberapa sumber judul SESUAI URUTAN PRIORITAS -- select_one dgn
    selector gabungan (",") gak menjamin urutan ini, jadi dicoba manual."""
    for sel in (".product-item__title", ".product-name"):
        el = card.select_one(sel)
        if el and el.get_text(strip=True):
            return el.get_text(strip=True)
    a = card.select_one("a[title]")
    if a and a.get("title", "").strip():
        return a["title"].strip()
    a2 = card.select_one("a")
    if a2 and a2.get_text(strip=True):
        return a2.get_text(strip=True)
    return None


class BhinnekaAdapter(BaseMarketplaceAdapter):
    source_name = "bhinneka"

    # Bhinneka kadang lambat mengirim body halaman pencarian. Pakai timeout
    # connect/read terpisah agar koneksi yang macet tetap cepat gagal, tetapi
    # halaman yang sedang diproses server diberi waktu cukup untuk selesai.
    FETCH_TIMEOUT = (10, 45)
    FETCH_ATTEMPTS = 3

    def fetch_search_page(self, keyword: str, page: int) -> Optional[str]:
        query = urlencode({"q": keyword, "page": page})
        url = f"https://www.bhinneka.com/catalog-search/products?{query}"
        self._last_fetch_error = None

        for attempt in range(1, self.FETCH_ATTEMPTS + 1):
            try:
                resp = self.session.get(url, timeout=self.FETCH_TIMEOUT)
                resp.raise_for_status()
                if resp.text.strip():
                    return resp.text
                raise requests.exceptions.RequestException("respons kosong")
            except requests.exceptions.RequestException as e:
                self._last_fetch_error = e
                if attempt == self.FETCH_ATTEMPTS:
                    break
                wait_seconds = attempt
                print(f"[bhinneka] fetch gagal (percobaan {attempt}/{self.FETCH_ATTEMPTS}): "
                      f"{e}. Coba lagi dalam {wait_seconds} dtk...")
                time.sleep(wait_seconds)

        print(f"[bhinneka] gagal fetch setelah {self.FETCH_ATTEMPTS} percobaan {url}: "
              f"{self._last_fetch_error}")
        return None

    def parse_listing_cards(self, raw_page: Optional[str]) -> List[dict]:
        # Jangan tampilkan error selector ketika request-nya sendiri gagal.
        # Ini sebelumnya membuat timeout terlihat seperti masalah selector.
        if raw_page is None:
            print("[bhinneka] halaman pencarian tidak tersedia; parsing selector dilewati.")
            return []

        soup = BeautifulSoup(raw_page, "html.parser")

        # TODO: ganti ".product-item" dengan class asli card produk Bhinneka
        cards = soup.select(".product-item")

        if not cards:
            print(f"[bhinneka] 0 card ketemu dgn selector '.product-item' "
                  f"(HTML {len(raw_page)} bytes). Cek manual class produk via "
                  f"Inspect Element lalu update selector di bhinneka_adapter.py.")
            return []

        results = []
        for card in cards:
            # TODO: sesuaikan semua selector di bawah dengan struktur asli
            name = _extract_name(card)
            price_el = card.select_one(".product-item__price, .price")
            link_el = card.select_one("a[href]")
            img_el = card.select_one("img")

            results.append({
                "name": name,
                "url": link_el["href"] if link_el and link_el.has_attr("href") else None,
                "price_text": price_el.get_text(strip=True) if price_el else None,
                "sku": card.get("data-sku") or card.get("data-product-id"),
                "img": img_el["src"] if img_el and img_el.has_attr("src") else None,
            })
        return [r for r in results if r["name"]]

    def normalize(self, raw_card: dict) -> LaptopListing:
        title = raw_card.get("name") or ""
        url = raw_card.get("url") or ""
        if url and not url.startswith("http"):
            url = "https://www.bhinneka.com" + url

        return LaptopListing(
            source=self.source_name,
            source_id=str(raw_card.get("sku") or hash(url)),
            url=url,
            title=title,
            brand=spec_parser.extract_brand(title),
            model=None,
            cpu=spec_parser.extract_cpu(title),
            ram_gb=spec_parser.extract_ram_gb(title),
            storage_gb=spec_parser.extract_storage_gb(title),
            gpu=None,
            screen_size_in=spec_parser.extract_screen_size(title),
            price_idr=_to_int(raw_card.get("price_text")),
            original_price_idr=None,
            condition="used" if "bekas" in title.lower() or "second" in title.lower() else "new",
            seller_name="Bhinneka (retailer resmi)",
            seller_rating=None,
            seller_num_reviews=None,
            seller_is_official=True,  # Bhinneka jual sendiri/first-party, bukan marketplace C2C
            sold_count=None,
            location=None,
        )
