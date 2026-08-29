import time
from playwright.sync_api import sync_playwright

DEFAULT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

LAUNCH_ARGS = [
    "--disable-http2",
    "--disable-blink-features=AutomationControlled",
]

STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['id-ID','id','en-US','en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
window.chrome = { runtime: {} };
Object.defineProperty(navigator, 'permissions', {
    get: () => ({ query: async () => ({ state: 'granted' }) })
});
"""


def _launch(p, headless: bool):
    try:
        return p.chromium.launch(channel="chrome", headless=headless, args=LAUNCH_ARGS)
    except Exception:
        return p.chromium.launch(headless=headless, args=LAUNCH_ARGS)


def intercept_json(url: str, match_substr: str, timeout_ms: int = 30000, headless: bool = True,
                    pre_goto_cookies: list = None, post_goto_click_text: str = None):
    """
    Buka `url` pakai browser asli, tunggu response JSON pertama yang URL-nya
    mengandung `match_substr`, lalu return body-nya sebagai dict.

    PENTING: body di-extract SYNCHRONOUS di dalam response callback, saat
    event itu juga -- bukan ditunda sampai setelah kode lain jalan (mis.
    klik tombol yang men-trigger navigasi baru). Kalau ditunda, Chrome bisa
    keburu buang buffer response-nya -> "navigated away" error.
    """
    with sync_playwright() as p:
        browser = _launch(p, headless)
        context = browser.new_context(
            user_agent=DEFAULT_UA,
            locale="id-ID",
            viewport={"width": 1366, "height": 768},
        )
        context.add_init_script(STEALTH_JS)

        if pre_goto_cookies:
            context.add_cookies(pre_goto_cookies)

        page = context.new_page()
        captured = {}

        def on_response(response):
            if "data" in captured:
                return
            if match_substr in response.url and response.status == 200:
                try:
                    captured["data"] = response.json()
                    captured["url"] = response.url
                except Exception as e:
                    print(f"[intercept_json] response ketangkep tapi gagal parse JSON: {e}")

        page.on("response", on_response)

        try:
            page.goto(url, wait_until="load", timeout=45000)
        except Exception as e:
            print(f"[intercept_json] goto error ke {url}: {e}")

        if post_goto_click_text:
            try:
                page.get_by_text(post_goto_click_text, exact=False).first.click(timeout=3000)
            except Exception:
                pass

        deadline = time.time() + (timeout_ms / 1000)
        while "data" not in captured and time.time() < deadline:
            page.wait_for_timeout(200)

        browser.close()

        if "data" not in captured:
            print(f"[intercept_json] gagal tangkap '{match_substr}' di {url}: timeout {timeout_ms}ms")
            return None

        data = captured["data"]
        n_items = (
            len(data.get("items", []))
            if isinstance(data, dict) and "items" in data
            else "?"
        )
        print(f"[intercept_json] tangkap '{match_substr}', items={n_items}")
        print(f"[intercept_json][debug] url respons: {captured.get('url')}")
        if n_items == "?":
            print(f"[intercept_json][debug] top-level keys respons: "
                  f"{list(data.keys()) if isinstance(data, dict) else type(data)}")
        return data


def rendered_page_texts(urls: list[str], headless: bool = True,
                        timeout_ms: int = 45000) -> list[str]:
    """Ambil teks yang terlihat setelah halaman SPA selesai dirender.

    Digunakan untuk data yang tidak ada pada respons pencarian, misalnya
    jumlah terjual di halaman detail produk. Satu browser dipakai untuk semua
    URL agar tidak meluncurkan Chrome untuk setiap listing.
    """
    if not urls:
        return []

    with sync_playwright() as p:
        browser = _launch(p, headless)
        context = browser.new_context(
            user_agent=DEFAULT_UA,
            locale="id-ID",
            viewport={"width": 1366, "height": 768},
        )
        context.add_init_script(STEALTH_JS)
        page = context.new_page()
        texts = []
        for url in urls:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(1_500)
                texts.append(page.locator("body").inner_text(timeout=5_000))
            except Exception as e:
                print(f"[rendered_page_texts] gagal buka {url}: {e}")
                texts.append("")
        browser.close()
    return texts
