# batch_run.py
import asyncio
import json
import os
import traceback
import re
from typing import Any, Dict, Optional

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from playwright.async_api import async_playwright



import extract_job_profile
import fetch_jobs
import prepare_application_draft
import score_job
import queue_utils
import config


# ----------------------------
# MODE SWITCH (prod vs injected)
# ----------------------------
USE_INJECTED = os.getenv("USE_INJECTED_JOBS", "0") == "1"

INJECTED_PATH = Path("data/injected_jobs.json")

SEEN_PATH = Path("data/seen_jobs_injected.json" if USE_INJECTED else "data/seen_jobs.json")
QUEUE_PATH = Path("queue/queue_injected.json" if USE_INJECTED else "queue/queue.json")
RUNS_DIR = Path("runs_injected" if USE_INJECTED else "runs")

AUTH_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"

Path("data").mkdir(exist_ok=True)
QUEUE_PATH.parent.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

WORK_ITEMS_DIR = Path("work_items")
WORK_ITEMS_DIR.mkdir(exist_ok=True)

# ----------------------------
# Config
# ----------------------------
def _read_config_int(name: str, fallback: int) -> int:
    cfg = Path("config.py")
    if not cfg.exists():
        return fallback
    txt = cfg.read_text(encoding="utf-8")
    if name not in txt:
        return fallback
    try:
        return int(txt.split(name, 1)[1].split("=", 1)[1].splitlines()[0].strip())
    except Exception:
        return fallback


FETCH_MAX_JOBS = _read_config_int("FETCH_MAX_JOBS", 80)
FETCH_MAX_PAGES = _read_config_int("FETCH_MAX_PAGES", 5)
FETCH_STOP_AFTER_SEEN_STREAK = _read_config_int("FETCH_STOP_AFTER_SEEN_STREAK", 30)

# one-offs

FETCH_PER_PAGE_EXTRACT_LIMIT = _read_config_int("FETCH_PER_PAGE_EXTRACT_LIMIT", 120)
FETCH_STOP_IF_PAGE_NEW_ITEMS_LEQ = _read_config_int("FETCH_STOP_IF_PAGE_NEW_ITEMS_LEQ", 5)



@dataclass
class RunSummary:
    run_id: str
    started_at: str
    finished_at: str
    jobs_input: int
    jobs_enqueued: int
    jobs_skipped_seen: int
    jobs_processed: int
    jobs_failed: int
    notes: List[str]
    mode: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_url(url: str) -> str:
    return fetch_jobs.normalize_url(url)

def _norm_key(k: str) -> str:
    # "Projekt-ID" -> "projektid", "project_id" -> "projectid"
    return re.sub(r"[^a-z0-9]", "", (k or "").lower())

def get_project_id(profile: Dict[str, Any]) -> Optional[str]:
    if not isinstance(profile, dict):
        return None

    wanted = {"projectid", "projektid"}  # normalized targets
    for k, v in profile.items():
        if _norm_key(k) in wanted and v:
            # return as string, keep digits if present
            return str(v).strip()
    return None

def _fallback_job_id_from_href(href: str) -> Optional[str]:
    """Best-effort numeric id from URL if extractor didn't return a project_id."""
    if not href:
        return None
    # pick last long-ish digit chunk if present
    nums = re.findall(r"\d{5,}", href)
    return nums[-1] if nums else None


def persist_work_items(
    *,
    job_id: str,
    href: str,
    title: str,
    captured_at: str,
    body_text: str,
    profile: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Persist raw and derived artifacts once (do not overwrite existing).
    Returns (work_item_path, profile_path) as strings.
    """
    WORK_ITEMS_DIR.mkdir(exist_ok=True)

    work_item_path = WORK_ITEMS_DIR / f"{job_id}.json"
    profile_path = WORK_ITEMS_DIR / f"{job_id}.profile.json"

    if not work_item_path.exists():
        payload = {
            "job_id": job_id,
            "url": href,
            "title": title,
            "captured_at": captured_at,
            "body_text": body_text,
        }
        work_item_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not profile_path.exists():
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(work_item_path), str(profile_path)

# ----------------------------
# Seen store
# ----------------------------
def load_seen() -> Dict[str, Any]:
    if SEEN_PATH.exists():
        try:
            return json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"seen_urls": {}, "last_run": None}


def save_seen(seen: Dict[str, Any]) -> None:
    SEEN_PATH.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------
# Queue IO (compat shim)
# ----------------------------
def load_queue_compat() -> Dict[str, Any]:
    try:
        return queue_utils.load_queue(path=QUEUE_PATH)  # type: ignore[arg-type]
    except TypeError:
        if QUEUE_PATH.exists():
            return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        return {"items": []}
    except Exception:
        if QUEUE_PATH.exists():
            return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        return {"items": []}


def save_queue_compat(queue: Dict[str, Any]) -> None:
    try:
        queue_utils.save_queue(queue, path=QUEUE_PATH)  # type: ignore[arg-type]
    except TypeError:
        QUEUE_PATH.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        QUEUE_PATH.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")


def enqueue_item_compat(queue: Dict[str, Any], item: Dict[str, Any]) -> None:
    try:
        queue_utils.enqueue_item(queue, item)
        return
    except Exception:
        pass
    queue.setdefault("items", [])
    queue["items"].append(item)


def discover_resume_paths() -> List[Path]:
    resumes_dir = Path("profiles") / "resumes"
    if not resumes_dir.exists():
        return []

    return sorted(
        p
        for p in resumes_dir.glob("*.json")
        if p.is_file() and "resume" in p.name.lower()
    )




async def fetch_body_text(page) -> str:
    try:
        return await page.inner_text("body")
    except Exception:
        try:
            return await page.content()
        except Exception:
            return ""


# ----------------------------
# Extractor resolver (fixes your AttributeError)
# ----------------------------
def resolve_profile_extractor() -> Callable[..., Dict[str, Any]]:
    """
    Your code was calling extract_job_profile.extract_job_profile(...), but that function
    doesn't exist in your module. This resolver tries common function names and returns one.
    """
    candidates = [
        "extract_job_profile",
        "extract",
        "parse_job_profile",
        "parse",
        "build_job_profile",
        "make_job_profile",
        "job_profile_from_text",
    ]

    for name in candidates:
        fn = getattr(extract_job_profile, name, None)
        if callable(fn):
            return fn  # type: ignore[return-value]

    # If none matched, provide a helpful error showing what's available
    public_callables = sorted(
        [
            n
            for n in dir(extract_job_profile)
            if not n.startswith("_") and callable(getattr(extract_job_profile, n, None))
        ]
    )
    raise AttributeError(
        "No known extractor function found in extract_job_profile.py. "
        "Expected one of: "
        + ", ".join(candidates)
        + ". Found callables: "
        + ", ".join(public_callables)
    )


async def process_and_enqueue(
    jobs: List[Dict[str, Any]],
    run_id: str,
    run_tags: Optional[List[str]] = None,
    job_pub: Optional[Dict[str, str]] = None,
) -> Tuple[int, int, int, int]:
    run_tags = run_tags or []
    job_pub = job_pub or {}

    queue = load_queue_compat()
    seen = load_seen()

    enqueued = 0
    skipped_seen = 0
    processed = 0
    failed = 0

    resume_paths = discover_resume_paths()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    extract_fn = resolve_profile_extractor()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            storage_state=str(AUTH_STATE_PATH) if AUTH_STATE_PATH.exists() else None
        )
        page = await context.new_page()

        for j in jobs:
            href = j.get("href")
            title = j.get("text", "") or ""
            if not href:
                continue

            u = normalize_url(href)
            if not u:
                continue

            if u in seen.get("seen_urls", {}):
                skipped_seen += 1
                continue

            try:
                # Robust navigation with retries + page reset on failure
                last_err = None
                for attempt in range(3):
                    try:
                        await page.goto(href, wait_until="domcontentloaded", timeout=45000)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        # reset the page to avoid "interrupted by another navigation" cascades
                        try:
                            await page.close()
                        except Exception:
                            pass
                        page = await context.new_page()
                        # small backoff (1s, 2s, 3s)
                        await page.wait_for_timeout(1000 * (attempt + 1))

                if last_err is not None:
                    print(f"[batch_run] SKIP href={href} after retries err={last_err}")
                    continue

                body_text = await fetch_body_text(page)


                # Call resolved extractor. Most implementations accept (title, body_text).
                # If yours differs, the error will now be explicit.
                profile = extract_fn(title=title, body_text=body_text)

                project_id = (
                    profile.get("project_id")
                    or profile.get("Projekt-ID")
                    or profile.get("Projekt_ID")
                    or profile.get("projekt_id")
                    or profile.get("ProjektId")
                    or profile.get("projectId")
                )

                # Persist work items (raw + profile) for offline rescoring/debugging
                stable_job_id = str(project_id) if project_id else (_fallback_job_id_from_href(href) or u or "unknown")
                work_item_path, profile_path = persist_work_items(
                    job_id=stable_job_id,
                    href=href,
                    title=title,
                    captured_at=now_iso(),
                    body_text=body_text,
                    profile=profile,
                )


                scoring = score_job.score_job(
                    job_profile=profile,
                    resume_paths=resume_paths,
                    model=model,
                )

                published_at = (j.get("published_at") or "").strip()
                if not published_at and u in job_pub:
                    published_at = (job_pub[u] or "").strip()

                item = {
                    "href": href,
                    "project_id": str(project_id) if project_id else None,
                    "job_id": str(project_id) if project_id else None,
                    "normalized_href": u,
                    "title": title,
                    "published_at": published_at or None,
                    "profile": profile,
                    "scoring": scoring,
                    "work_item_path": work_item_path,
                    "profile_path": profile_path,
                    "created_at": now_iso(),
                    "run_id": run_id,
                    "run_tags": run_tags,
                    "mode": "injected" if USE_INJECTED else "prod",
                    "selected_resume_id": scoring.get("selected_resume_id") or "default",
                    "top_resume_scores": scoring.get("top_resume_scores") or [],
                }

                enqueue_item_compat(queue, item)
                save_queue_compat(queue)

                seen["seen_urls"][u] = {"seen_at": now_iso(), "run_id": run_id}
                enqueued += 1
                processed += 1

                # Draft generation failures should NOT count as job failures
                try:
                    prepare_application_draft.maybe_prepare_application_draft(item)
                except Exception as e:
                    print(f"[batch_run] DRAFT_FAIL href={href} err={type(e).__name__}: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)

            except Exception as e:
                failed += 1
                print(f"[batch_run] FAIL href={href} err={type(e).__name__}: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                # Do not mark seen on failure (allows retry)

        await context.close()
        await browser.close()

    save_seen(seen)
    return enqueued, skipped_seen, processed, failed


async def main() -> None:
    run_id = make_run_id()
    started_at = now_iso()
    notes: List[str] = []
    mode = "injected" if USE_INJECTED else "prod"

    seen = load_seen()

    if USE_INJECTED:
        if not INJECTED_PATH.exists():
            raise FileNotFoundError(f"USE_INJECTED_JOBS=1 but {INJECTED_PATH} does not exist.")
        payload = json.loads(INJECTED_PATH.read_text(encoding="utf-8"))
        jobs = payload.get("jobs", [])
        print(f"[batch_run] MODE=injected Using injected jobs: {len(jobs)} from {INJECTED_PATH}")
        notes.append(f"Using injected jobs from {INJECTED_PATH}")
    else:
        jobs = await fetch_jobs.fetch_jobs_async(
            max_jobs=FETCH_MAX_JOBS,
            seen_urls=set(seen["seen_urls"].keys()),
            max_pages=FETCH_MAX_PAGES,
            stop_after_seen_streak=FETCH_STOP_AFTER_SEEN_STREAK,
            per_page_extract_limit=FETCH_PER_PAGE_EXTRACT_LIMIT,
            stop_if_page_new_items_leq=FETCH_STOP_IF_PAGE_NEW_ITEMS_LEQ,
        )

    job_pub: Dict[str, str] = {}
    for j in jobs:
        href = j.get("href")
        if not href:
            continue
        u = normalize_url(href)
        pub = (j.get("published_at") or "").strip()
        if u:
            job_pub[u] = pub

    queue = load_queue_compat()
    updated = 0
    for item in queue.get("items", []):
        if item.get("published_at"):
            continue
        u = item.get("normalized_href") or normalize_url(item.get("href", ""))
        if u and u in job_pub and job_pub[u]:
            item["published_at"] = job_pub[u]
            updated += 1
    if updated:
        save_queue_compat(queue)
        notes.append(f"Backfilled published_at for {updated} queued items")

    enq, skipped_seen, processed, failed = await process_and_enqueue(
        jobs=jobs,
        run_id=run_id,
        run_tags=["batch", mode],
        job_pub=job_pub,
    )

    finished_at = now_iso()

    summary = RunSummary(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        jobs_input=len(jobs),
        jobs_enqueued=enq,
        jobs_skipped_seen=skipped_seen,
        jobs_processed=processed,
        jobs_failed=failed,
        notes=notes,
        mode=mode,
    )

    summary_path = RUNS_DIR / f"summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    seen["last_run"] = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "jobs_input": len(jobs),
        "jobs_enqueued": enq,
        "jobs_failed": failed,
        "mode": mode,
    }
    save_seen(seen)

    print(
        f"[batch_run] MODE={mode} run_id={run_id} jobs={len(jobs)} "
        f"enqueued={enq} failed={failed} queue={QUEUE_PATH} seen={SEEN_PATH} summary={summary_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())