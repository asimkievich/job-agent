import asyncio
import re
from pathlib import Path

from playwright.async_api import async_playwright

from config import load_secrets, require_env


# Auth state is stored OUTSIDE the repo (never commit this file)
AUTH_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"
LOGIN_URL = "https://www.freelancermap.de/login"


async def _extract_visible_error_text(page) -> str | None:
    """
    Best-effort: scrape common error containers without relying on exact markup.
    Returns first non-empty visible error text if found.
    """
    candidates = [
        # common alert/error patterns
        ".alert",
        ".fm-alert",
        ".error",
        ".errors",
        ".invalid-feedback",
        "[role='alert']",
        "[data-testid*='error']",
        "[class*='error']",
    ]
    for sel in candidates:
        loc = page.locator(sel).filter(has_text=re.compile(r".+"))
        try:
            if await loc.count() > 0:
                # pick first visible with some text
                for i in range(min(await loc.count(), 5)):
                    item = loc.nth(i)
                    if await item.is_visible():
                        txt = (await item.inner_text()).strip()
                        if txt:
                            return txt
        except Exception:
            continue

    # Also look for typical German login error wording anywhere on page
    keywords = [
        "Benutzername",
        "Passwort",
        "ungültig",
        "falsch",
        "Fehler",
        "nicht korrekt",
        "versuchen Sie es erneut",
    ]
    try:
        body = (await page.locator("body").inner_text()).strip()
        if any(k.lower() in body.lower() for k in keywords):
            # don't dump the whole body; just signal that something is present
            return "Possible inline error present on page (German login-related text detected)."
    except Exception:
        pass

    return None


async def main() -> None:
    load_secrets()
    username = require_env("FREELANCERMAP_USERNAME")
    password = require_env("FREELANCERMAP_PASSWORD")

    async with async_playwright() as p:
        # locale helps avoid weird language flows (and can reduce translation prompts)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(locale="de-DE")
        page = await context.new_page()

        await page.goto(LOGIN_URL, wait_until="domcontentloaded")

        # Accept cookie banner if present
        try:
            await page.get_by_role("button", name="Alle Cookies akzeptieren").click(timeout=4000)
            await page.wait_for_timeout(300)
        except Exception:
            pass

        # Fill fields by label (stable with the German UI)
        await page.get_by_label("E-Mail-Adresse oder Benutzername").fill(username)
        await page.get_by_label("Passwort").fill(password)

        # Ensure "Angemeldet bleiben" is checked (nice for persistence)
        try:
            checkbox = page.get_by_label("Angemeldet bleiben")
            if await checkbox.count() > 0:
                if not await checkbox.is_checked():
                    await checkbox.check()
        except Exception:
            pass

        # Click the form submit button.
        # There may be duplicates, so scope to <main> and choose first.
        await page.get_by_role("main").get_by_test_id("next-button").first.click()

        # Do NOT wait for "networkidle" (often never happens).
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(800)

        # Determine login success by the presence of logged-in indicators,
        # not just URL changes (some apps keep /login in history or use XHR).
        logged_in_indicators = [
            re.compile(r"\bAbmelden\b", re.I),
            re.compile(r"\bLogout\b", re.I),
            re.compile(r"\bMein Profil\b", re.I),
            re.compile(r"\bProfil\b", re.I),
            re.compile(r"\bEinstellungen\b", re.I),
        ]

        success = False
        for _ in range(30):  # ~15 seconds total
            # 1) URL changed away from /login
            if "login" not in page.url.lower():
                success = True
                break

            # 2) Logged-in UI indicators present
            for rx in logged_in_indicators:
                try:
                    if await page.get_by_text(rx).count() > 0:
                        success = True
                        break
                except Exception:
                    continue
            if success:
                break

            # 3) If an obvious error appears, stop early
            err = await _extract_visible_error_text(page)
            if err:
                await page.screenshot(path="login_failed.png", full_page=True)
                raise RuntimeError(
                    f"Login appears to have failed (inline error detected). "
                    f"URL: {page.url}. Error hint: {err}. Screenshot: login_failed.png"
                )

            await page.wait_for_timeout(500)

        if not success:
            await page.screenshot(path="login_failed.png", full_page=True)
            raise RuntimeError(
                f"Login likely failed or is blocked/stuck; still at: {page.url}. "
                "Screenshot saved as login_failed.png"
            )

        # Save authenticated browser state
        await context.storage_state(path=str(AUTH_STATE_PATH))
        await browser.close()

    print(f"OK: auth state saved to {AUTH_STATE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
