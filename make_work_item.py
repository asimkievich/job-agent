import asyncio
import json
from pathlib import Path
from datetime import datetime

from playwright.async_api import async_playwright


AUTH_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"
OUT_DIR = Path("work_items")
OUT_DIR.mkdir(exist_ok=True)

# THIS is the URL you actually have (board + popup)
BOARD_URL = "https://www.freelancermap.de/projektboerse.html?id=2962307"


async def main() -> None:
    if not AUTH_STATE_PATH.exists():
        raise RuntimeError("Auth state not found. Run login_and_save_state.py first.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=str(AUTH_STATE_PATH), locale="de-DE")
        page = await context.new_page()

        # 1) Open the board URL (popup will appear)
        await page.goto(BOARD_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)

        # 2) Inside the popup, find the real project link
        # Freelancermap always includes at least one /projekt/... anchor
        project_link = page.locator("a[href*='/projekt/']").first
        href = await project_link.get_attribute("href")

        if not href:
            await page.screenshot(path="no_project_link.png", full_page=True)
            raise RuntimeError(
                "Could not find /projekt/ link inside popup. "
                "Saved screenshot no_project_link.png"
            )

        if href.startswith("/"):
            href = "https://www.freelancermap.de" + href

        print(f"Resolved real job URL: {href}")

        # 3) Load the REAL job page
        await page.goto(href, wait_until="domcontentloaded")
        await page.wait_for_timeout(1200)

        title = (await page.title()).strip()
        body_text = (await page.locator("body").inner_text()).strip()

        await browser.close()

    item = {
        "meta": {
            "created_at": datetime.now().astimezone().isoformat(),
            "source": "freelancermap",
            "resolved_from": BOARD_URL
        },
        "job": {
            "url": href,
            "title": title,
            "body_text": body_text
        },
        "decision": {
            "selected_resume_profile_id": "resume_product_owner_ai_pm",
            "notes": "initial capture"
        }
    }

    out_path = OUT_DIR / "job_001.json"
    out_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
