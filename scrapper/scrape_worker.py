"""Dijalankan sebagai proses Python terpisah (bukan di-import).

Alasan: Playwright sync API butuh event loop asyncio ber-tipe Proactor
(supaya bisa spawn subprocess browser). Kernel Jupyter di Windows memaksa
event loop ber-tipe Selector demi kompatibilitas zmq, dan policy asyncio
itu berlaku process-wide -- bahkan dari thread lain di proses yang sama.
Satu-satunya cara aman untuk memanggil adapter (Playwright) dari dalam
notebook adalah menjalankannya di proses OS yang benar-benar terpisah.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokopedia_adapter import TokopediaAdapter
from shopee_adapter import ShopeeAdapter
from blibli_adapter import BlibliAdapter
from bhinneka_adapter import BhinnekaAdapter

ADAPTERS = {
    "tokopedia": TokopediaAdapter,
    "shopee": ShopeeAdapter,
    "blibli": BlibliAdapter,
    "bhinneka": BhinnekaAdapter,
}


def main():
    site_name, keyword, max_pages, out_path = sys.argv[1:5]
    adapter = ADAPTERS[site_name]()
    listings = adapter.scrape(keyword, max_pages=int(max_pages))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([l.to_dict() for l in listings], f, ensure_ascii=False)


if __name__ == "__main__":
    main()
