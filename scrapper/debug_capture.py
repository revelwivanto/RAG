"""
Jalankan: python debug_capture.py "https://www.tokopedia.com/search?q=laptop+asus"
Simpan screenshot ke debug_screenshot.png dan print semua URL response
yang lewat, biar ketahuan: (a) diblokir/interstitial, (b) request API-nya
namanya beda dari yang kita tebak.
"""
import sys
from playwright.sync_api import sync_playwright

LAUNCH_ARGS = ["--disable-http2", "--disable-blink-features=AutomationControlled"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['id-ID','id','en-US','en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
window.chrome = { runtime: {} };
"""


def main(url):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=False, args=LAUNCH_ARGS)
        except Exception:
            browser = p.chromium.launch(headless=False, args=LAUNCH_ARGS)

        context = browser.new_context(user_agent=UA, locale="id-ID",
                                       viewport={"width": 1366, "height": 768})
        context.add_init_script(STEALTH_JS)
        page = context.new_page()

        seen = []
        page.on("response", lambda r: seen.append((r.status, r.url)))

        print(f"membuka: {url}")
        try:
            page.goto(url, wait_until="load", timeout=45000)
        except Exception as e:
            print(f"goto error: {e}")

        page.wait_for_timeout(8000)  # kasih waktu XHR nyusul + kamu lihat manual

        page.screenshot(path="debug_screenshot.png", full_page=True)
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(page.content())

        print(f"\n=== {len(seen)} response tertangkap ===")
        for status, u in seen:
            if any(k in u for k in ["api", "graphql", "gql", "search", "json"]):
                print(status, u)

        print("\nscreenshot -> debug_screenshot.png")
        print("html dump  -> debug_page.html")
        print("\nTekan Enter buat nutup browser (silakan cek manual dulu window-nya)...")
        input()
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1])
