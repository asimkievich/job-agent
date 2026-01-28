import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from playwright.async_api import async_playwright


AUTH_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"
LISTING_URL = "https://www.freelancermap.de/projektboerse.html"

OUT_PATH = Path("data/latest_jobs.json")
OUT_PATH.parent.mkdir(exist_ok=True)


async def fetch_jobs_async(max_jobs: int = 60) -> List[Dict[str, str]]:
    if not AUTH_STATE_PATH.exists():
        raise RuntimeError(
            f"Auth state not found at {AUTH_STATE_PATH}. Run login_and_save_state.py first."
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=str(AUTH_STATE_PATH), locale="de-DE")
        page = await context.new_page()

        await page.goto(LISTING_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1200)

        anchors = page.locator('a[href*="/projekt/"]')
        n = await anchors.count()

        jobs: List[Dict[str, str]] = []
        seen = set()

        for i in range(min(n, max_jobs * 3)):
            a = anchors.nth(i)
            href = await a.get_attribute("href")
            if not href:
                continue

            if href.startswith("/"):
                href = "https://www.freelancermap.de" + href

            if "/projekt/" not in href:
                continue
            if href in seen:
                continue
            seen.add(href)

            text = (await a.inner_text()).strip()
            if not text:
                continue

            jobs.append({"text": text, "href": href})
            if len(jobs) >= max_jobs:
                break

        await browser.close()

    # also write debug file
    payload = {
        "count": len(jobs),
        "jobs": jobs,
        "meta": {
            "fetched_at": datetime.now().astimezone().isoformat(),
            "source": "freelancermap",
            "listing_url": LISTING_URL,
        },
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return jobs


def main():
    jobs = asyncio.run(fetch_jobs_async(max_jobs=60))
    payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nOK: wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()


