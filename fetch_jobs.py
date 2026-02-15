# fetch_jobs.py
import asyncio
import json
import re
import queue_db
from pathlib import Path
from typing import Set
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from time_utils import parse_published_at




AUTH_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"
LISTING_URL = "https://www.freelancermap.de/projektboerse.html"

OUT_PATH = Path("data/latest_jobs.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Debug artifacts (written when next-page navigation fails OR page does not change)
DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# If we have seen_urls, we allow early-stop checks only after scanning at least this many pages
MIN_PAGES_BEFORE_EARLY_STOP = 2

# If we scan a whole page and find <= this many new items, we treat it as "nothing new" and stop.
# (Set to 0 for strict "must be zero new items".)
STOP_IF_PAGE_NEW_ITEMS_LEQ = 0


def normalize_url(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("/"):
        href = "https://www.freelancermap.de" + href
    return href

async def ensure_sort_neueste(page) -> None:
    """
    Best-effort: ensure listing is sorted by "Neueste".
    We try a few common patterns:
      - a select/dropdown with option text "Neueste"
      - a button/menu item "Neueste"
    If already selected, do nothing.
    """
    try:
        # If "Neueste" is already visibly selected, we can skip.
        # (Heuristic: an element that contains "Neueste" and has aria-selected=true,
        # or a selected <option> with that label.)
        already = page.locator(':is([aria-selected="true"], option:checked):has-text("Neueste")')
        if await already.count() > 0:
            return

        # Strategy 1: Click an element that *is* "Neueste" (might be a tab/button/menu item).
        neueste_btn = page.locator(':is(a,button,div,span):visible:has-text("Neueste")').first
        if await neueste_btn.count() > 0:
            try:
                await neueste_btn.click()
                await page.wait_for_timeout(800)
                return
            except Exception:
                pass

        # Strategy 2: Open a "Sortieren" dropdown then click "Neueste".
        # (Text varies; we try a few likely labels.)
        sort_triggers = [
            ':is(button,a,div,span):visible:has-text("Sort")',
            ':is(button,a,div,span):visible:has-text("Sortieren")',
            ':is(button,a,div,span):visible:has-text("Relevanz")',
            ':is(button,a,div,span):visible:has-text("Neueste")',  # sometimes current label is shown
        ]
        for sel in sort_triggers:
            trig = page.locator(sel).first
            if await trig.count() == 0:
                continue
            try:
                await trig.click()
                await page.wait_for_timeout(300)
                neu = page.locator(':is(a,button,div,span,li):visible:has-text("Neueste")').first
                if await neu.count() > 0:
                    await neu.click()
                    await page.wait_for_timeout(800)
                    return
            except Exception:
                continue

    except Exception:
        # Best-effort only — never block fetching if sorting UI changes.
        return



def load_seen_urls_from_queue(
    queue_path: Path = Path("queue") / "queue.json",
    *,
    queue_db_path: Optional[Path] = None,
) -> Set[str]:
    # Prefer DB if provided
    if queue_db_path is not None and queue_db_path.exists():
        return queue_db.get_seen_urls(queue_db_path)

    # Fallback to JSON (transition period)
    if not queue_path.exists():
        return set()

    try:
        doc = json.loads(queue_path.read_text(encoding="utf-8"))
        items = doc.get("items_by_job_id") or {}
        out: Set[str] = set()
        if isinstance(items, dict):
            for _, it in items.items():
                if isinstance(it, dict):
                    url = (it.get("url") or "").strip()
                    if url:
                        out.add(url)
        return out
    except Exception:
        return set()

async def ensure_vertragsart_freiberuflich(page, debug: bool = True) -> None:
    """
    Best-effort: ensure the left filter 'Vertragsart' has 'Freiberuflich' checked.
    This keeps the listing page focused on freelance roles (excluding AÜ/Festanstellung).
    """

    # 1) Expand the "Vertragsart" filter section if it is collapsible.
    #    (On some UIs it’s always open; then this does nothing.)
    try:
        toggle = page.get_by_role("button", name=re.compile(r"\bVertragsart\b", re.IGNORECASE))
        if await toggle.count() > 0:
            await toggle.first.click()
            await page.wait_for_timeout(150)
    except Exception:
        pass

    # 2) Find the checkbox for "Freiberuflich" and ensure it is checked.
    # Prefer stable id/for; fall back to text.
    cb = page.locator('input#contract-type-contracting[type="checkbox"]').first
    lbl = page.locator('label[for="contract-type-contracting"]').first

    if await cb.count() == 0:
        cb = page.locator('label:has-text("Freiberuflich") input[type="checkbox"]').first
    if await lbl.count() == 0:
        lbl = page.locator('label:has-text("Freiberuflich")').first


    if await lbl.count() == 0 or await cb.count() == 0:
        if debug:
            print("[fetch_jobs] WARNING: could not locate 'Freiberuflich' label/checkbox; continuing unfiltered.")
        return

    try:
        if not await cb.is_checked():
            if debug:
                print("[fetch_jobs] Applying filter: Vertragsart = Freiberuflich")
            await lbl.scroll_into_view_if_needed()
            await lbl.click()
            await page.wait_for_timeout(600)
            try:
                await page.wait_for_load_state("networkidle", timeout=4000)
            except Exception:
                pass
        else:
            if debug:
                print("[fetch_jobs] Filter already active: Vertragsart = Freiberuflich")
    except Exception as e:
        if debug:
            print(f"[fetch_jobs] WARNING: failed to apply 'Freiberuflich' filter ({e}); continuing.")

    
async def _extract_job_links_from_page(page, limit: int) -> List[Dict[str, str]]:
    """
    Extract job links from *listing cards*, not from arbitrary /projekt/ links anywhere on the page.

    Heuristic definition of a "card":
      - contains a /projekt/ link
      - contains "Freiberuflich" (common on result cards)
    This filters out nav/footer/random links that happen to include /projekt/.
    """
    jobs: List[Dict[str, str]] = []
    seen_hrefs: Set[str] = set()

    # Card candidates: visible containers that look like listing entries.
    # Using :has-text("Freiberuflich") is a strong signal for real project cards.
    cards = page.locator(
        ':is(article, li, div):visible'
        ':has(a[href*="/projekt/"])'
    )

    card_count = await cards.count()

    # If for any reason the heuristic finds nothing (markup change),
    # fall back to a safer version of the old approach: only visible anchors
    # and exclude common non-card texts.
    if card_count == 0:
        anchors = page.locator('a[href*="/projekt/"]:visible')
        n = await anchors.count()
        max_scan = min(n, max(80, limit * 8))  # scan a bit more defensively

        for i in range(max_scan):
            a = anchors.nth(i)
            href = normalize_url(await a.get_attribute("href"))
            if not href or "/projekt/" not in href:
                continue
            if href in seen_hrefs:
                continue

            text = (await a.inner_text() or "").strip()
            if not text:
                continue

            # Filter obvious non-card links
            bad = ["Mehr Jobs", "Ähnliche Projekte", "Impressum", "Datenschutz", "AGB"]
            if any(b.lower() in text.lower() for b in bad):
                continue

            seen_hrefs.add(href)
            jobs.append({"text": text, "href": href})
            if len(jobs) >= limit:
                break

        return jobs

    # Normal path: extract from cards
    max_cards_to_scan = min(card_count, max(40, limit * 2))
    for i in range(max_cards_to_scan):
        card = cards.nth(i)

        # Prefer a prominent link: longest non-empty text among project links in the card.
        links = card.locator('a[href*="/projekt/"]:visible')
        ln = await links.count()
        best_href = ""
        best_text = ""

        # Cap per-card scan so it stays fast
        for j in range(min(ln, 12)):
            a = links.nth(j)
            href = normalize_url(await a.get_attribute("href"))
            if not href or "/projekt/" not in href:
                continue

            text = (await a.inner_text() or "").strip()
            if not text:
                continue

            # Skip obvious button-like texts
            if text.lower() in {"bewerben", "merken", "anschreiben", "text generieren"}:
                continue

            # Choose "best" as the longest label (usually the title)
            if len(text) > len(best_text):
                best_text = text
                best_href = href

        if not best_href or best_href in seen_hrefs:
            continue

        seen_hrefs.add(best_href)
        jobs.append({"text": best_text, "href": best_href})

        if len(jobs) >= limit:
            break

    return jobs


async def _go_to_next_page(page, previous_first_href: str) -> bool:
    """
    Click next and wait until the first job changes.
    This handles AJAX pagination where no real navigation occurs.
    """

    async def get_first_href() -> Optional[str]:
        # Use the same “card-like” heuristic as extraction:
        cards = page.locator(
            ':is(article, li, div):visible'
            ':has(a[href*="/projekt/"])'
        )
        if await cards.count() == 0:
            anchors = page.locator('a[href*="/projekt/"]:visible')
            if await anchors.count() == 0:
                return None
            href = await anchors.first.get_attribute("href")
            return normalize_url(href)

        first_card = cards.first
        a = first_card.locator('a[href*="/projekt/"]:visible').first
        href = await a.get_attribute("href")
        return normalize_url(href)


    # Try rel=next link first (canonical pagination)
    
    next_href = await page.locator('link[rel="next"]').get_attribute("href")
    next_href = normalize_url(next_href) if next_href else None
    if next_href:
        last_err = None
        for attempt in range(3):
            try:
                await page.goto(
                    next_href,
                    wait_until="domcontentloaded",
                    timeout=60000,
            )
                await page.wait_for_timeout(800)
                new_first = await get_first_href()
                return bool(new_first and new_first != previous_first_href)
            
            except Exception as e:
                last_err = e
                await asyncio.sleep(1.5 + attempt)

        return False

    # ---------------------
    # Scroll down so pagination renders
    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(800)
    except Exception:
        pass

    candidates = [
        'a[rel="next"]',
        'a:has-text("Weiter")',
        'a:has-text("Nächste")',
        'a:has-text("Next")',
        'button:has-text("Weiter")',
        'button:has-text("Nächste")',
        'button:has-text("Next")',
        'a[aria-label*="next" i]',
        'button[aria-label*="next" i]',
        'a[aria-label*="weiter" i]',
        'button[aria-label*="weiter" i]',
        'a[aria-label*="nächste" i]',
        'button[aria-label*="nächste" i]',
        # common chevrons/arrows
        'a:has-text("›")',
        'a:has-text("→")',
        'a:has-text(">")',
    ]

    old_url = page.url
   
    clicked = False
    for sel in candidates:
        loc = page.locator(sel)
        if await loc.count() > 0:
            try:
                await loc.first.scroll_into_view_if_needed()
                await loc.first.click()
                clicked = True
                break
            except Exception:
                continue

    if not clicked:
        return False

    await page.wait_for_timeout(1000)
    new_first = await get_first_href()
   

    # Wait for content change (max ~6s)
        # Wait for *something* to change: URL, DOM, or first job.
    old_url = page.url

    for _ in range(20):  # up to ~10s
        await page.wait_for_timeout(500)

        # 1) URL changed (some sites use query params for pagination)
        if page.url != old_url:
            return True

        # 2) First job changed (your existing signal)
        new_href = await get_first_href()
        if new_href and new_href != previous_first_href:
            return True

        # 3) Fallback: scroll height changed (new content appended/replaced)
        try:
            old_h = await page.evaluate("() => window.__jobagent_prev_h || 0")
            h = await page.evaluate("() => document.body.scrollHeight")
            await page.evaluate("(h) => { window.__jobagent_prev_h = h; }", h)
            if old_h and h and h != old_h:
                return True
        except Exception:
            pass

    return False


async def _extract_published_from_detail(context, href: str) -> Dict[str, str]:
    """
    Open job detail in a new tab, extract 'veröffentlicht am ...', return:
      {"published_at": iso, "published_at_raw": raw}
    Returns {} if not found.
    """
    detail = await context.new_page()
    try:
        await detail.goto(href, wait_until="domcontentloaded")
        await detail.wait_for_timeout(800)

        # Get whole page text and parse from it (robust across markup changes)
        body_text = await detail.locator("body").inner_text()
        parsed = parse_published_at(body_text)
        if not parsed:
            return {}

        return {
            "published_at_raw": parsed.raw,
            "published_at": parsed.dt.isoformat(),
        }
    except Exception:
        return {}
    finally:
        try:
            await detail.close()
        except Exception:
            pass


async def _enrich_jobs_with_published_at(
    context,
    jobs: List[Dict[str, str]],
    seen_urls: Set[str],
    only_for_new: bool = True,
    max_enrich: int = 80,
) -> None:
    enriched = 0
    for j in jobs:
        href = j.get("href") or ""
        if not href:
            continue
        if only_for_new and href in seen_urls:
            continue
        if "published_at" in j or "published_at_raw" in j:
            continue

        meta = await _extract_published_from_detail(context, href)

        if meta:
            j.update(meta)

        enriched += 1
        if enriched >= max_enrich:
            break

async def accept_cookies_if_present(page, debug: bool = True) -> None:
    try:
        btn = page.locator(
            ':is(button,a):visible:has-text("Alle Cookies akzeptieren")'
        ).first
        if await btn.count() > 0:
            await btn.click()
            await page.wait_for_timeout(800)
    except Exception as e:
        if debug:
            pass


async def fetch_jobs_async(
    max_jobs: int = 200,
    seen_urls: Optional[Set[str]] = None,
    max_pages: int = 10,
    stop_after_seen_streak: int = 50,
    per_page_extract_limit: int = 120,
    stop_if_page_new_items_leq: int = STOP_IF_PAGE_NEW_ITEMS_LEQ,
    debug: bool = True,
) -> List[Dict[str, str]]:

    """
    Fetch job links from freelancermap listing pages with pagination.

    EARLY EXIT (what you asked for):
      If we have seen_urls and after MIN_PAGES_BEFORE_EARLY_STOP:
        - we stop when a page adds <= STOP_IF_PAGE_NEW_ITEMS_LEQ brand-new items
        - OR we hit a consecutive-seen streak >= stop_after_seen_streak
      This avoids scanning many pages when there are no new posts.

    Also:
      If caller passes seen_urls=None, we auto-load seen_urls from queue/queue.json (if present),
      so you still get early exit in practice.
    """
    if not AUTH_STATE_PATH.exists():
        raise RuntimeError(f"Auth state not found at {AUTH_STATE_PATH}. Run login_and_save_state.py first.")

    if seen_urls is None:
        # Prefer SQLite queue if present; fallback to JSON if not.
        db_default = Path("queue") / "queue.db"
        seen_urls = load_seen_urls_from_queue(queue_db_path=db_default)


    async with async_playwright() as p:
        HEADLESS = False
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(storage_state=str(AUTH_STATE_PATH), locale="de-DE")
        page = await context.new_page()

        if debug:
            ua = await page.evaluate("() => navigator.userAgent")
            lang = await page.evaluate("() => navigator.language")
            tz = await page.evaluate("() => Intl.DateTimeFormat().resolvedOptions().timeZone")
            vp = page.viewport_size



        await page.goto(LISTING_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1200)    
        await accept_cookies_if_present(page, debug=debug)

        if debug:
            print(f"[fetch_jobs][DEBUG] landed_on={page.url}")

            html = await page.content()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            await page.screenshot(path=str(DEBUG_DIR / f"landing_{ts}.png"), full_page=True)
            (DEBUG_DIR / f"landing_{ts}.html").write_text(html, encoding="utf-8")
            print(f"[fetch_jobs][DEBUG] saved {DEBUG_DIR / f'landing_{ts}.png'} and .html")
    

        # Ensure correct ordering (newest-first) before extracting links.
        await ensure_sort_neueste(page)
        await ensure_vertragsart_freiberuflich(page, debug=debug)

        if debug:
            # after UI interactions, re-check counts
            job_links2 = await page.locator('a[href*="/projekt/"]').count()
            job_links_vis2 = await page.locator('a[href*="/projekt/"]:visible').count()
            print(f"[fetch_jobs][DEBUG] after_sort_filter job_links_total={job_links2} job_links_visible={job_links_vis2}")



        unique_jobs: List[Dict[str, str]] = []
        unique_hrefs: Set[str] = set()

        pages_scanned = 0
        consecutive_seen = 0
        consecutive_seen_max = 0

        early_stop_triggered = False
        early_stop_reason = ""
        seen_streak_at_stop = 0

        last_first_href: str = ""
        stagnant_pages: int = 0

        while pages_scanned < max_pages and len(unique_jobs) < max_jobs:
            pages_scanned += 1

            batch = await _extract_job_links_from_page(page, limit=per_page_extract_limit)
            first_href = batch[0]["href"] if batch else ""

            # log (useful when debugging pagination)
            first5 = [item.get("href", "") for item in batch[:5]]

            hrefs_only = [item.get("href", "") for item in batch if item.get("href")]
            Path("data/debug").mkdir(parents=True, exist_ok=True)
            out = Path("data/debug") / f"hrefs_page_{pages_scanned}.json"
            out.write_text(
                json.dumps(hrefs_only, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            # ===== ADD OPTION A ENDS HERE =====

            # detect "next doesn't change content"
            if last_first_href and first_href and first_href == last_first_href:
                stagnant_pages += 1
            else:
                stagnant_pages = 0
            last_first_href = first_href

            if stagnant_pages >= 1 and pages_scanned >= 2:
                # capture debug artifacts
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    await page.screenshot(path=str(DEBUG_DIR / f"pagination_stuck_{ts}.png"), full_page=True)
                    html = await page.content()
                    (DEBUG_DIR / f"pagination_stuck_{ts}.html").write_text(html, encoding="utf-8")
                except Exception:
                    pass
                early_stop_triggered = True
                early_stop_reason = "pagination_stuck_same_first_job"
                seen_streak_at_stop = consecutive_seen
                break

            # Track how many NEW items this page contributes
            page_new_items = 0

            for item in batch:
                href = item.get("href", "")
                if not href or "/projekt/" not in href:
                    continue
                if href in unique_hrefs:
                    continue

                unique_hrefs.add(href)
                unique_jobs.append(item)

                is_seen = href in seen_urls
                if is_seen:
                    consecutive_seen += 1
                    consecutive_seen_max = max(consecutive_seen_max, consecutive_seen)
                else:
                    page_new_items += 1
                    consecutive_seen = 0

                if len(unique_jobs) >= max_jobs:
                    break

                if (
                    seen_urls
                    and pages_scanned >= MIN_PAGES_BEFORE_EARLY_STOP
                    and consecutive_seen >= stop_after_seen_streak
                ):
                    early_stop_triggered = True
                    early_stop_reason = "seen_streak"
                    seen_streak_at_stop = consecutive_seen
                    break

            if len(unique_jobs) >= max_jobs:
                break
            if early_stop_triggered:
                break

            if debug:
                print(
        f"[fetch_jobs][DEBUG] pages_scanned={pages_scanned} "
        f"page_new_items={page_new_items} consecutive_seen={consecutive_seen} "
        f"unique_jobs={len(unique_jobs)}"
    )
                if batch:
                    print(f"[fetch_jobs][DEBUG] page={pages_scanned} first={batch[0].get('href')}")

        
            # -------- EARLY EXIT YOU ASKED FOR --------
            if seen_urls and pages_scanned >= MIN_PAGES_BEFORE_EARLY_STOP:
                if page_new_items <= stop_if_page_new_items_leq:
                    early_stop_triggered = True
                    early_stop_reason = f"no_new_items_on_page<= {stop_if_page_new_items_leq}"
                    seen_streak_at_stop = consecutive_seen
                    break
            # ----------------------------------------

            prev_first = batch[0]["href"] if batch else ""
            moved = await _go_to_next_page(page, prev_first)
            if not moved:
                early_stop_triggered = True
                early_stop_reason = "no_next_page"
                seen_streak_at_stop = consecutive_seen
                break

        # Enrich jobs with published_at for easier manual verification.
        # Default: only enrich NEW (unseen) jobs to keep it fast.
        try:
            await _enrich_jobs_with_published_at(
                context,
                unique_jobs,
                seen_urls=seen_urls or set(),
                only_for_new=False,
                max_enrich=120,
            )
        except Exception:
            pass
        
        if debug and early_stop_triggered:
            print(
                f"[fetch_jobs][DEBUG] early_stop_reason={early_stop_reason} "
                f"seen_streak_at_stop={seen_streak_at_stop}"
            )

        await browser.close()

    jobs = unique_jobs[:max_jobs]

    if debug:
        urls = [j.get("href", "") for j in jobs]
        new_hits = sum(1 for u in urls if u and u not in seen_urls)
        seen_hits = sum(1 for u in urls if u and u in seen_urls)

        pub_dts = []
        for j in jobs:
            pub = j.get("published_at")
            if pub:
                try:
                    pub_dts.append(datetime.fromisoformat(pub))
                except Exception:
                    pass

        published_at_min = min(pub_dts).isoformat() if pub_dts else None
        published_at_max = max(pub_dts).isoformat() if pub_dts else None
        published_at_count = len(pub_dts)

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
                "per_page_extract_limit": per_page_extract_limit,
                "used_seen_urls_early_stop": bool(seen_urls),
                "pages_scanned": pages_scanned,
                "unique_hrefs_found": len(unique_hrefs),
                "seen_hits": seen_hits,
                "new_hits": new_hits,
                "consecutive_seen_max": consecutive_seen_max,
                "early_stop_triggered": early_stop_triggered,
                "early_stop_reason": early_stop_reason,
                "seen_streak_at_stop": seen_streak_at_stop,
                "min_pages_before_early_stop": MIN_PAGES_BEFORE_EARLY_STOP,
                "stop_if_page_new_items_leq": stop_if_page_new_items_leq,
                "published_at_min": published_at_min,
                "published_at_max": published_at_max,
                "published_at_count": published_at_count,
            },
        }
        OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return jobs


def main() -> None:
    jobs = asyncio.run(fetch_jobs_async(max_jobs=300, seen_urls=None, max_pages=15, debug=True))
    payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nOK: wrote {OUT_PATH.resolve()}")
    if DEBUG_DIR.exists():
        print(f"Debug dir: {DEBUG_DIR.resolve()}")


if __name__ == "__main__":
    main()