import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from playwright.async_api import async_playwright

AUTH_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"
LISTING_URL = "https://www.freelancermap.de/projektboerse.html"

OUT_PATH = Path("data/latest_jobs.json")
OUT_PATH.parent.mkdir(exist_ok=True)


def normalize_url(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("/"):
        href = "https://www.freelancermap.de" + href
    return href


async def _extract_job_links_from_page(page, limit: int) -> List[Dict[str, str]]:
    """
    Extract job links from the current listing page.
    We dedupe later across pages; here we just read what's visible.
    """
    anchors = page.locator('a[href*="/projekt/"]')
    n = await anchors.count()

    jobs: List[Dict[str, str]] = []
    max_scan = min(n, max(50, limit * 5))  # scan a bit more than needed

    for i in range(max_scan):
        a = anchors.nth(i)
        href = normalize_url(await a.get_attribute("href"))
        if not href or "/projekt/" not in href:
            continue

        text = (await a.inner_text() or "").strip()
        if not text:
            continue

        jobs.append({"text": text, "href": href})

        if len(jobs) >= limit:
            break

    return jobs


async def _go_to_next_page(page) -> bool:
    """
    Try to navigate to the next listing page.
    Returns True if navigation likely succeeded, False if no next page found.

    Implementation uses resilient heuristics:
      1) link rel=next
      2) anchor with common 'next' labels
      3) pagination element with arrow/chevron
    """
    # Strategy 1: rel=next
    rel_next = page.locator('a[rel="next"]')
    if await rel_next.count() > 0:
        await rel_next.first.click()
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(800)
        return True

    # Strategy 2: common German/English labels
    candidates = [
        'a:has-text("Weiter")',
        'a:has-text("Nächste")',
        'a:has-text("Next")',
        'a:has-text(">")',
        'a[aria-label*="next" i]',
        'a[aria-label*="weiter" i]',
        'button:has-text("Weiter")',
        'button:has-text("Next")',
    ]
    for sel in candidates:
        loc = page.locator(sel)
        if await loc.count() > 0:
            try:
                await loc.first.click()
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(800)
                return True
            except Exception:
                continue

    return False


def _count_new_vs_seen(urls: List[str], seen_urls: Set[str]) -> Tuple[int, int, int]:
    """
    Returns: (new_count, seen_count, seen_streak_from_end?)
    We mainly use a consecutive-seen streak while iterating in order.
    """
    new_count = 0
    seen_count = 0
    for u in urls:
        if u in seen_urls:
            seen_count += 1
        else:
            new_count += 1
    return new_count, seen_count, 0


async def fetch_jobs_async(
    max_jobs: int = 200,
    seen_urls: Optional[Set[str]] = None,
    max_pages: int = 10,
    stop_after_seen_streak: int = 50,
    per_page_extract_limit: int = 120,
    debug: bool = True,
) -> List[Dict[str, str]]:
    """
    Fetch job links from freelancermap listing pages with pagination.

    Goals:
      - Avoid missing postings on high-volume days by reading multiple pages.
      - Optionally stop early when we hit many already-seen URLs.

    Parameters:
      max_jobs: max number of unique job links to return (safety cap)
      seen_urls: set of normalized hrefs already processed; used for early stop
      max_pages: safety cap on number of listing pages to scan
      stop_after_seen_streak: stop when we encounter this many seen links consecutively
      per_page_extract_limit: how many links to scan per page (upper bound)
      debug: write latest_jobs.json for inspection

    Returns:
      List of {"text": ..., "href": ...}
    """
    if not AUTH_STATE_PATH.exists():
        raise RuntimeError(f"Auth state not found at {AUTH_STATE_PATH}. Run login_and_save_state.py first.")

    seen_urls = seen_urls or set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=str(AUTH_STATE_PATH), locale="de-DE")
        page = await context.new_page()

        await page.goto(LISTING_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1200)

        unique_jobs: List[Dict[str, str]] = []
        unique_hrefs: Set[str] = set()

        pages_scanned = 0
        consecutive_seen = 0

        while pages_scanned < max_pages and len(unique_jobs) < max_jobs:
            pages_scanned += 1

            batch = await _extract_job_links_from_page(page, limit=per_page_extract_limit)

            # Process in listing order: helps the seen-streak stop condition.
            for item in batch:
                href = item.get("href", "")
                if not href or "/projekt/" not in href:
                    continue
                if href in unique_hrefs:
                    continue

                unique_hrefs.add(href)
                unique_jobs.append(item)

                if href in seen_urls:
                    consecutive_seen += 1
                else:
                    consecutive_seen = 0  # reset streak when we find something new

                if len(unique_jobs) >= max_jobs:
                    break
                if seen_urls and consecutive_seen >= stop_after_seen_streak:
                    break

            if len(unique_jobs) >= max_jobs:
                break
            if seen_urls and consecutive_seen >= stop_after_seen_streak:
                break

            # Attempt next page; stop if no navigation.
            moved = await _go_to_next_page(page)
            if not moved:
                break

        await browser.close()

    # Trim to max_jobs (since we may have over-collected slightly)
    jobs = unique_jobs[:max_jobs]

    if debug:
        payload = {
            "count": len(jobs),
            "jobs": jobs,
            "meta": {
                "fetched_at": datetime.now().astimezone().isoformat(),
                "source": "freelancermap",
                "listing_url": LISTING_URL,
                "max_jobs": max_jobs,
                "max_pages": max_pages,
                "stop_after_seen_streak": stop_after_seen_streak,
                "pages_scanned": pages_scanned,
                "unique_hrefs_found": len(unique_hrefs),
                "used_seen_urls_early_stop": bool(seen_urls),
            },
        }
        OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return jobs


def main() -> None:
    # For manual testing we fetch without seen_urls, but allow more than 60.
    jobs = asyncio.run(fetch_jobs_async(max_jobs=200, seen_urls=None, max_pages=5, debug=True))
    payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nOK: wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()


