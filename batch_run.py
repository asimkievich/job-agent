# batch_run.py
import asyncio
import json
import os
import traceback
import re
from typing import Any, Dict, Optional
import hashlib
import job_graph

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
# Discovery (prod vs injected)
# ----------------------------

async def discover_jobs(*, seen: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns (jobs, notes_to_append).
    Keeps logic identical to what used to be in main().
    """
    notes: List[str] = []

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

    return jobs, notes


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

def resolve_project_id(
    profile: Dict[str, Any],
    body_text: str,
    last_job_id: Optional[str],
) -> Optional[str]:

    # 1️⃣ Try profile first (highest confidence)
    project_id = get_project_id(profile)
    if project_id and project_id.isdigit():
        return project_id

    # 2️⃣ Try label-based extraction
    if body_text:
        lines = [l.strip() for l in body_text.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            key = line.lower().replace("-", "").replace(" ", "")
            if key == "projektid":
                if i + 1 < len(lines):
                    candidate = lines[i + 1]
                    if candidate.isdigit() and 6 <= len(candidate) <= 8:
                        return candidate

    # 3️⃣ Try ±10 heuristic (lowest confidence)
    if body_text and last_job_id and last_job_id.isdigit():
        last_id_int = int(last_job_id)
        candidates = re.findall(r"\b\d{6,8}\b", body_text)

        closest = None
        min_diff = None

        for c in candidates:
            val = int(c)
            diff = abs(val - last_id_int)
            if diff <= 10:
                if min_diff is None or diff < min_diff:
                    closest = c
                    min_diff = diff

        if closest:
            print(f"[heuristic] Using near-ID fallback: {closest}")
            return closest

    return None


def build_job_pub(jobs: List[Dict[str, Any]]) -> Dict[str, str]:
    job_pub: Dict[str, str] = {}
    for j in jobs:
        href = j.get("href")
        if not href:
            continue
        u = normalize_url(href)
        pub = (j.get("published_at") or "").strip()
        if u:
            job_pub[u] = pub
    return job_pub


def backfill_published_at(job_pub: Dict[str, str]) -> int:
    """
    Minimal-diff: keep your legacy loop over queue.get("items", []) exactly as before.
    Returns updated count.
    """
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

    return updated

async def process_and_enqueue(
    jobs: List[Dict[str, Any]],
    run_id: str,
    run_tags: Optional[List[str]] = None,
    job_pub: Optional[Dict[str, str]] = None,
) -> Tuple[int, int, int, int]:
    run_tags = run_tags or []
    job_pub = job_pub or {}

    # NOTE: we intentionally keep "seen" loaded once and only saved at the end.
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
        last_job_id: Optional[str] = None

                # ---- LangGraph nodes (v1) ----

        async def node_capture(state: job_graph.JobState) -> Dict[str, Any]:
            href = state["href"]

            last_err = None
            for attempt in range(3):
                try:
                    await page.goto(href, wait_until="domcontentloaded", timeout=45000)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    await page.wait_for_timeout(1000 * (attempt + 1))

            if last_err is not None:
                return {"error": f"nav_failed: {type(last_err).__name__}: {last_err}"}

            body_text = await fetch_body_text(page)
            return {"body_text": body_text}

        def node_extract_profile(state: job_graph.JobState) -> Dict[str, Any]:
            if state.get("error"):
                return {}
            profile = extract_fn(title=state.get("title", ""), body_text=state.get("body_text", ""))
            return {"profile": profile}

        def node_resolve_ids(state: job_graph.JobState) -> Dict[str, Any]:
            if state.get("error"):
                return {}

            nonlocal last_job_id

            body_text = state.get("body_text", "")
            profile = state.get("profile") or {}

            project_id = resolve_project_id(profile, body_text, last_job_id)
            if project_id and project_id.isdigit():
                last_job_id = project_id

            stable_job_id = str(project_id).strip() if project_id else None
            if not stable_job_id:
                stable_job_id = _fallback_job_id_from_href(state.get("href", ""))

            if not stable_job_id:
                u = state.get("normalized_href", "")
                h = hashlib.sha1(u.encode("utf-8")).hexdigest()[:10] if u else "nohref"
                stable_job_id = f"unknown_{h}"

            return {"project_id": project_id, "stable_job_id": stable_job_id}

        def node_persist(state: job_graph.JobState) -> Dict[str, Any]:
            if state.get("error"):
                return {}

            work_item_path, profile_path = persist_work_items(
                job_id=state["stable_job_id"],
                href=state["href"],
                title=state.get("title", ""),
                captured_at=now_iso(),
                body_text=state.get("body_text", ""),
                profile=state.get("profile") or {},
            )
            return {"work_item_path": work_item_path, "profile_path": profile_path}

        def node_score(state: job_graph.JobState) -> Dict[str, Any]:
            if state.get("error"):
                return {}
            scoring = score_job.score_job(
                job_profile=state.get("profile") or {},
                resume_paths=resume_paths,
                model=model,
            )
            return {"scoring": scoring}

        def node_decide(state: job_graph.JobState) -> Dict[str, Any]:
            if state.get("error"):
                return {"final_lifecycle": "failed", "decision": "skip"}

            scoring = state.get("scoring") or {}
            decision = (scoring.get("decision") or "").strip().lower()

            if decision == "skip":
                final_lifecycle = "skipped"
            elif decision in {"review", "pursue"}:
                final_lifecycle = "awaiting_human_decision"
            else:
                final_lifecycle = "scored"

            return {"decision": decision, "final_lifecycle": final_lifecycle}

        def node_write(state: job_graph.JobState) -> Dict[str, Any]:
            if state.get("error"):
                return {}

            item = {
                "href": state["href"],
                "normalized_href": state["normalized_href"],
                "title": state.get("title", ""),
                "published_at": state.get("published_at"),
                "project_id": state.get("project_id"),
                "job_id": state["stable_job_id"],
                "profile": state.get("profile") or {},
                "scoring": state.get("scoring") or {},
                "work_item_path": state.get("work_item_path") or "",
                "profile_path": state.get("profile_path") or "",
                "created_at": now_iso(),
                "run_id": state.get("run_id"),
                "run_tags": state.get("run_tags") or [],
                "mode": state.get("mode"),
                "source": "freelancermap",
                "lifecycle": state.get("final_lifecycle") or "scored",
                "selected_resume_id": (state.get("scoring") or {}).get("selected_resume_id") or "default",
                "top_resume_scores": (state.get("scoring") or {}).get("top_resume_scores") or [],
            }

            queue = load_queue_compat()
            queue_utils.enqueue_item(queue, item)   # canonical schema converter
            save_queue_compat(queue)

            seen["seen_urls"][state["normalized_href"]] = {"seen_at": now_iso(), "run_id": state.get("run_id")}
            return {}

        graph = job_graph.build_job_graph({
            "capture": node_capture,
            "extract_profile": node_extract_profile,
            "resolve_ids": node_resolve_ids,
            "persist": node_persist,
            "score": node_score,
            "decide": node_decide,
            "write": node_write,
        })

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
            
            if u in seen.get("seen_urls", {}):
                skipped_seen += 1
                continue

            published_at = (j.get("published_at") or "").strip() or None
            if (not published_at) and (u in job_pub):
                published_at = (job_pub[u] or "").strip() or None

            state_in: job_graph.JobState = {
                "href": href,
                "normalized_href": u,
                "title": title,
                "published_at": published_at,
                "run_id": run_id,
                "run_tags": run_tags,
                "mode": "injected" if USE_INJECTED else "prod",
            }

            try:
                out = await graph.ainvoke(
                    state_in,
                    config={"configurable": {"thread_id": f"{run_id}:{u}"}},
                )
                if out.get("error"):
                    failed += 1
                    print(f"[batch_run] FAIL href={href} err={out['error']}", flush=True)
                else:
                    enqueued += 1
                    processed += 1

                    # keep your draft step (optional; we’ll convert it to a node later)
                    try:
                        # you can reconstruct item from out if needed
                        pass
                    except Exception:
                        pass

            except Exception as e:
                failed += 1
                print(f"[batch_run] FAIL href={href} err={type(e).__name__}: {e}", flush=True)
                print(traceback.format_exc(), flush=True)

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
    # 1) Discover jobs (prod or injected)
    jobs, discover_notes = await discover_jobs(seen=seen)
    notes.extend(discover_notes)

    # 2) Build published_at lookup + backfill legacy queue items
    job_pub = build_job_pub(jobs)
    updated = backfill_published_at(job_pub)
    if updated:
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