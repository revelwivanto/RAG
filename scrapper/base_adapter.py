from abc import ABC, abstractmethod
from typing import List
import time
import random
import requests
from schema import LaptopListing


class BaseMarketplaceAdapter(ABC):
    """
    Kontrak yang wajib dipenuhi tiap adapter situs.
    Subclass hanya perlu implement 4 method ini; orchestrator tidak
    peduli detail HTML masing-masing situs.
    """

    source_name: str = "unknown"
    base_search_url: str = ""

    def __init__(self, session: requests.Session = None, delay_range=(1.5, 3.5)):
        self.session = session or requests.Session()
        self.session.headers.update(self._default_headers())
        self.delay_range = delay_range

    def _default_headers(self):
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
        }

    def _throttle(self):
        time.sleep(random.uniform(*self.delay_range))

    @abstractmethod
    def fetch_search_page(self, keyword: str, page: int) -> str:
        """Ambil raw HTML/JSON dari halaman hasil pencarian."""
        raise NotImplementedError

    @abstractmethod
    def parse_listing_cards(self, raw_page) -> List[dict]:
        """Ekstrak list card produk mentah (belum di-normalize) dari 1 halaman."""
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw_card: dict) -> LaptopListing:
        """Ubah 1 card mentah -> LaptopListing (skema seragam)."""
        raise NotImplementedError

    def fetch_detail_page(self, url: str) -> dict:
        """Ambil data tambahan dari halaman detail produk (opsional per-adapter).

        Default: tidak didukung -> return dict kosong. Adapter yang mau mengisi
        field yang kosong dari halaman detail (bukan cuma card hasil search)
        boleh override method ini; adapter lain tetap jalan tanpa perubahan.
        """
        return {}

    def scrape(self, keyword: str, max_pages: int = 3, enrich_detail: bool = False) -> List[LaptopListing]:
        # Disimpan sebagai atribut instance (bukan argumen normalize()) supaya
        # signature normalize(raw_card) tidak berubah untuk adapter lain yang
        # belum/tidak mendukung enrich_detail.
        self._enrich_detail = enrich_detail
        results = []
        for page in range(1, max_pages + 1):
            raw_page = self.fetch_search_page(keyword, page)
            cards = self.parse_listing_cards(raw_page)
            if not cards:
                break
            for card in cards:
                try:
                    results.append(self.normalize(card))
                except Exception as e:
                    print(f"[{self.source_name}] skip 1 card, parse error: {e}")
            self._throttle()
        return results
