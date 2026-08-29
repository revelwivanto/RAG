import argparse
from tokopedia_adapter import TokopediaAdapter
from shopee_adapter import ShopeeAdapter
from blibli_adapter import BlibliAdapter
from bhinneka_adapter import BhinnekaAdapter
from category_filter import filter_accessories
from scam_filter import filter_listings
from spec_enricher import enrich_listings
from normalize_listings import normalise_listings
from deduplicate_listings import deduplicate_listings
from exporter import export_csv, export_json

ADAPTERS = {
    "tokopedia": TokopediaAdapter,
    "shopee": ShopeeAdapter,
    "blibli": BlibliAdapter,
    "bhinneka": BhinnekaAdapter,
    # tambah situs baru: cukup daftarkan class adapter baru di sini
}


def run(keyword: str, sites: list, max_pages: int, drop_scams: bool, out_prefix: str,
        enrich_from_web: bool = False):
    all_listings = []
    for site in sites:
        adapter = ADAPTERS[site]()
        print(f"[{site}] scraping keyword='{keyword}' ({max_pages} pages)...")
        listings = adapter.scrape(keyword, max_pages=max_pages)
        print(f"[{site}] got {len(listings)} listings")
        all_listings.extend(listings)

    all_listings = filter_accessories(all_listings)
    all_listings = enrich_listings(all_listings, fetch_product_page=enrich_from_web)
    all_listings = normalise_listings(all_listings)
    all_listings = deduplicate_listings(all_listings)
    all_listings = filter_listings(all_listings, drop_scams=drop_scams)
    n_scam = sum(1 for l in all_listings if l.is_suspected_scam)
    print(f"total={len(all_listings)} suspected_scam={n_scam}")

    export_csv(all_listings, f"{out_prefix}.csv")
    export_json(all_listings, f"{out_prefix}.json")
    print(f"saved: {out_prefix}.csv, {out_prefix}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--sites", nargs="+", default=list(ADAPTERS.keys()))
    ap.add_argument("--max-pages", type=int, default=2)
    ap.add_argument("--drop-scams", action="store_true")
    ap.add_argument(
        "--enrich-from-web",
        action="store_true",
        help="lengkapi spesifikasi kosong dari halaman produk (lebih lambat)",
    )
    ap.add_argument("--out", default="laptop_listings")
    args = ap.parse_args()

    run(args.keyword, args.sites, args.max_pages, args.drop_scams, args.out,
        enrich_from_web=args.enrich_from_web)
