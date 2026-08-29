"""Unit test untuk fetch_detail_page() dan wiring enrich_detail di TokopediaAdapter.

Semua test di sini mock intercept_json / fetch_detail_page -- tidak ada yang
benar-benar membuka browser atau menghubungi Tokopedia.
"""
import tokopedia_adapter as tk


def make_adapter():
    # delay_range=(0, 0) supaya _throttle() tidak benar-benar nge-sleep saat test.
    return tk.TokopediaAdapter(delay_range=(0, 0))


def make_card(title, **overrides):
    card = {
        "id": "1",
        "url": "https://www.tokopedia.com/toko/produk-1",
        "name": title,
        "price": {"number": 5_000_000},
        "shop": {"name": "Toko A", "city": "Jakarta", "isOfficial": True},
    }
    card.update(overrides)
    return card


# ---------------------------------------------------------------------------
# fetch_detail_page(): parsing JSON halaman detail
# ---------------------------------------------------------------------------

def test_fetch_detail_page_extracts_spec_rows_from_nested_json(monkeypatch):
    adapter = make_adapter()

    fake_body = {
        "data": {
            "pdpGetLayout": {
                "components": [
                    {
                        "name": "product_detail",
                        "data": [
                            {
                                "content": [
                                    {"title": "Sistem Operasi", "subtitle": "Windows 11"},
                                    {"title": "Kartu Grafis", "subtitle": "RTX 3050"},
                                    {"title": "Ukuran Layar", "subtitle": "14 inch"},
                                ]
                            }
                        ],
                    }
                ]
            }
        }
    }
    monkeypatch.setattr(tk, "intercept_json", lambda url, match_substr: fake_body)

    result = adapter.fetch_detail_page("https://www.tokopedia.com/toko/produk-1")

    assert result["gpu"] == "RTX 3050"
    assert result["screen_size_in"] == 14.0


def test_fetch_detail_page_returns_empty_dict_and_prints_debug_when_shape_unknown(monkeypatch, capsys):
    adapter = make_adapter()
    monkeypatch.setattr(tk, "intercept_json", lambda url, match_substr: {"data": {"unexpected": True}})

    result = adapter.fetch_detail_page("https://www.tokopedia.com/toko/produk-1")

    assert result == {}
    # Field API belum ketahuan -> harus print struktur, bukan nebak diam-diam.
    assert "tidak nemu list spesifikasi" in capsys.readouterr().out


def test_fetch_detail_page_returns_empty_dict_for_blank_url():
    adapter = make_adapter()
    assert adapter.fetch_detail_page("") == {}


# ---------------------------------------------------------------------------
# normalize(): wiring enrich_detail, tanpa nimpa field yang sudah ada
# ---------------------------------------------------------------------------

def test_normalize_default_does_not_call_detail_page(monkeypatch):
    """_enrich_detail belum pernah di-set (scrape() belum dipanggil) -> default False."""
    adapter = make_adapter()
    calls = []
    monkeypatch.setattr(adapter, "fetch_detail_page", lambda url: calls.append(url) or {})

    card = make_card("Asus Vivobook Intel i5 8GB 512GB SSD")
    listing = adapter.normalize(card)

    assert listing.storage_gb == 512
    assert listing.gpu is None
    assert calls == []


def test_normalize_enrich_detail_fills_missing_without_overwriting_existing(monkeypatch):
    adapter = make_adapter()
    adapter._enrich_detail = True  # seperti yang di-set oleh scrape(enrich_detail=True)

    calls = []

    def fake_fetch_detail_page(url):
        calls.append(url)
        return {
            "storage_gb": 256,  # beda dari nilai di title -> HARUS diabaikan
            "gpu": "RTX 3050",
            "screen_size_in": 14.0,
            "model": "G614",
        }

    monkeypatch.setattr(adapter, "fetch_detail_page", fake_fetch_detail_page)

    card = make_card("Asus ROG Strix Intel i7 16GB 512GB SSD")
    listing = adapter.normalize(card)

    # (a) storage_gb sudah keisi dari title (512) -> tidak boleh ketiban 256 dari detail page
    assert listing.storage_gb == 512
    # (b) gpu/screen_size_in/model kosong dari card -> diisi dari detail page
    assert listing.gpu == "RTX 3050"
    assert listing.screen_size_in == 14.0
    assert listing.model == "G614"
    assert calls == [card["url"]]


def test_scrape_enrich_detail_false_makes_no_extra_request(monkeypatch):
    adapter = make_adapter()

    detail_calls = []
    monkeypatch.setattr(adapter, "fetch_detail_page", lambda url: detail_calls.append(url) or {})
    monkeypatch.setattr(adapter, "fetch_search_page", lambda keyword, page: {"raw": True})

    card = make_card("Asus Vivobook Intel i5 8GB")  # storage/gpu/model/screen semua kosong di title
    monkeypatch.setattr(adapter, "parse_listing_cards", lambda raw_page: [card] if raw_page else [])

    listings = adapter.scrape("laptop asus", max_pages=1, enrich_detail=False)

    assert len(listings) == 1
    assert listings[0].storage_gb is None
    # (c) enrich_detail=False -> fetch_detail_page tidak boleh terpanggil sama sekali
    assert detail_calls == []


def test_scrape_enrich_detail_true_calls_detail_page_for_missing_fields(monkeypatch):
    adapter = make_adapter()

    detail_calls = []

    def fake_fetch_detail_page(url):
        detail_calls.append(url)
        return {"gpu": "RTX 3050"}

    monkeypatch.setattr(adapter, "fetch_detail_page", fake_fetch_detail_page)
    monkeypatch.setattr(adapter, "fetch_search_page", lambda keyword, page: {"raw": True})

    card = make_card("Asus Vivobook Intel i5 8GB")
    monkeypatch.setattr(adapter, "parse_listing_cards", lambda raw_page: [card] if raw_page else [])

    listings = adapter.scrape("laptop asus", max_pages=1, enrich_detail=True)

    assert len(listings) == 1
    assert listings[0].gpu == "RTX 3050"
    assert detail_calls == [card["url"]]
