import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright


AUTH_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"
PROJECTS_URL = "https://www.freelancermap.de/projektboerse.html"

MAX_JOBS = 15  # start small; increase later


@dataclass
class Job:
    title: str
    url: str
    location: Optional[str] = None
    remote_hint: Optional[str] = None
    rate_hint: Optional[str] = None
    posted_hint: Optional[str] = None
    customer_hint: Optional[str] = None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _first_match(patterns: list[str], text: str) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return _clean(m.group(1))
    return None


async def extract_job_from_detail(page, title: str, url: str) -> Job:
    # Use visible text as a robust baseline (selectors vary)
    body_text = _clean(await page.locator("body").inner_text())

    # Heuristics: try to pick out common fields from German listings
    location = _first_match(
        [
            r"(?:Einsatzort|Ort|Standort)\s*[:\-]\s*([^\n\r|•]+)",
            r"(?:Location)\s*[:\-]\s*([^\n\r|•]+)",
        ],
        body_text,
    )

    posted = _first_match(
        [
            r"(?:Eingestellt|Veröffentlicht|Online seit)\s*[:\-]\s*([^\n\r|•]+)",
            r"(?:Starttermin)\s*[:\-]\s*([^\n\r|•]+)",
        ],
        body_text,
    )

    rate = _first_match(
        [
            r"(?:Stundensatz|Tagessatz|Honorar)\s*[:\-]\s*([^\n\r|•]+)",
            r"(?:Rate)\s*[:\-]\s*([^\n\r|•]+)",
        ],
        body_text,
    )

    customer = _first_match(
        [
            r"(?:Kunde|Auftraggeber)\s*[:\-]\s*([^\n\r|•]+)",
        ],
        body_text,
    )

    # Remote: just detect hints (we’ll refine later with better selectors)
    remote_hint = None
    if re.search(r"\bremote\b", body_text, flags=re.IGNORECASE):
        remote_hint = "remote mentioned"
    elif re.search(r"\bvor ort\b", body_text, flags=re.IGNORECASE):
        remote_hint = "vor Ort mentioned"
    elif re.search(r"\bonsite\b", body_text, flags=re.IGNORECASE):
        remote_hint = "onsite mentioned"

    return Job(
        title=title,
        url=url,
        location=location,
        remote_hint=remote_hint,
        rate_hint=rate,
        posted_hint=posted,
        customer_hint=customer,
    )


async def main() -> None:
    if not AUTH_STATE_PATH.exists():
        raise RuntimeError("Auth state not found. Run login_and_save_state.py first.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=str(AUTH_STATE_PATH), locale="de-DE")
        page = await context.new_page()

        # 1) Load project board
        await page.goto(PROJECTS_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)

        # 2) Collect project links
        links = await page.eval_on_selector_all(
            "a[href]",
            """
            els => els
              .map(a => ({ text: (a.textContent || '').trim(), href: a.href }))
              .filter(x =>
                  x.href.includes('freelancermap.de') &&
                  (x.href.includes('/projekt/') || x.href.includes('/projektboerse/projekte/')) &&
                  x.text.length > 10
              )
            """
        )

        # De-duplicate
        seen = set()
        jobs_seed = []
        for x in links:
            href = x.get("href")
            text = x.get("text")
            if not href or href in seen:
                continue
            seen.add(href)
            jobs_seed.append({"title": text, "url": href})
            if len(jobs_seed) >= MAX_JOBS:
                break

        results: list[Job] = []

        # 3) Visit each detail page and extract best-effort fields
        for i, item in enumerate(jobs_seed, start=1):
            url = item["url"]
            title = item["title"]
            print(f"[{i}/{len(jobs_seed)}] {title}")

            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(1000)

            job = await extract_job_from_detail(page, title=title, url=url)
            results.append(job)

        await browser.close()

    print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
