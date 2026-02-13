from playwright.sync_api import sync_playwright
import time

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto("https://www.freelancermap.de", wait_until="domcontentloaded")
        time.sleep(3)

        # First pass: grab visible title-like headings
        headings = page.locator("h1, h2, h3").all_inner_texts()
        headings = [h.strip() for h in headings if h.strip()]

        print("First headings found (up to 20):")
        for i, h in enumerate(headings[:20], start=1):
            print(f"{i}. {h}")

        browser.close()

if __name__ == "__main__":
    main()
