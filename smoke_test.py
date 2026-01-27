from playwright.sync_api import sync_playwright
import time

def main():
    print("SMOKE: starting")
    with sync_playwright() as p:
        print("SMOKE: launching chromium")
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        print("SMOKE: going to site")
        page.goto("https://www.freelancermap.de", wait_until="domcontentloaded")
        title = page.title()
        print("SMOKE: TITLE =", title)
        time.sleep(2)
        print("SMOKE: closing browser")
        browser.close()
    print("SMOKE: done")

if __name__ == "__main__":
    main()
