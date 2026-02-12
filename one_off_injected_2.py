#!/usr/bin/env python3
"""
one_off_inject_and_score.py  (self-contained utility)

Purpose
- Maintain a small, curated evaluation set in queue/queue_injected.json.
- For each JOB_ID in JOB_IDS:
  1) If present in main queue (queue/queue.json): deep-copy the full item into injected queue.
     (Keeps original published_at, captured_at, url, etc.)
  2) Ensure a work-item profile exists (work_items/<job_id>.profile.json or artifacts.profile_path)
  3) Score via score_job.score_job(...) and write results back into injected queue item.

Fallback (optional)
- If job_id is NOT in main queue AND no profile exists:
  - Best-effort resolve a /projekt/<slug> URL via Playwright search by job_id.
  - This script intentionally does NOT guess how to build work_items automatically unless you wire it.
    (You can later wire fetch_job_details/extract_job_profile entrypoints safely.)

How to use
- Edit JOB_IDS below.
- Run:
    python one_off_inject_and_score.py
- Optional overrides:
    python one_off_inject_and_score.py --no-playwright
    python one_off_inject_and_score.py --queue queue/queue.json --injected queue/queue_injected.json
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer  # type: ignore

import score_job  # type: ignore


# ============================================================
# EDIT HERE: job ids to inject/copy + score
# ============================================================
JOB_IDS: List[str] = [
    "2967427"
]


# ============================================================
# Paths / defaults (match your repo layout)
# ============================================================
QUEUE_PATH = Path("queue/queue.json")
INJECTED_QUEUE_PATH = Path("queue/queue_injected.json")
WORK_ITEMS_DIR = Path("work_items")
RESUMES_DIR = Path("profiles/resumes")

# Optional: Playwright auth (only used if fallback enabled and file exists)
STORAGE_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"

# Toggle fallback behavior (can also override via CLI)
ENABLE_PLAYWRIGHT_FALLBACK = True


# ============================================================
# Helpers
# ============================================================
def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_queue_shape(queue_obj: Dict[str, Any]) -> None:
    if "items_by_job_id" not in queue_obj or not isinstance(queue_obj["items_by_job_id"], dict):
        queue_obj["items_by_job_id"] = {}
    if "meta" not in queue_obj or not isinstance(queue_obj.get("meta"), dict):
        queue_obj["meta"] = queue_obj.get("meta") if isinstance(queue_obj.get("meta"), dict) else {}


def _queue_items(queue_obj: Dict[str, Any]) -> Dict[str, Any]:
    items = queue_obj.get("items_by_job_id")
    if not isinstance(items, dict):
        raise ValueError("Queue JSON missing 'items_by_job_id' dict")
    return items


def _norm_job_id(x: str) -> str:
    s = str(x).strip()
    if not s.isdigit():
        raise ValueError(f"JOB_ID must be numeric, got: {x!r}")
    return s


def _discover_resumes(resumes_dir: Path) -> List[Path]:
    if not resumes_dir.exists():
        return []
    # Your resumes are JSON; keep search_profile.json out.
    return sorted([p for p in resumes_dir.glob("*.json") if p.name != "search_profile.json"])


def _load_job_profile_from_item(item: Dict[str, Any], work_items_dir: Path) -> Tuple[Dict[str, Any], Path]:
    """
    Return (job_profile_dict, profile_path_used).
    Priority:
      1) item['artifacts']['profile_path']
      2) work_items/{job_id}.profile.json
    """
    job_id = str(item.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("Queue item missing job_id")

    artifacts = item.get("artifacts") or {}
    profile_path_str = artifacts.get("profile_path")
    if isinstance(profile_path_str, str) and profile_path_str.strip():
        p = Path(profile_path_str)
        if not p.exists():
            # tolerate backslashes / relative paths
            p = Path(profile_path_str.replace("\\", "/"))
        if p.exists():
            return _read_json(p), p

    p2 = work_items_dir / f"{job_id}.profile.json"
    if p2.exists():
        return _read_json(p2), p2

    raise FileNotFoundError(
        f"Missing job profile for job_id={job_id}. Tried artifacts.profile_path and {p2}"
    )


def _apply_scoring_to_item(item: Dict[str, Any], scoring: Dict[str, Any]) -> None:
    """
    Mutates `item` in-place.
    Keeps all original fields; only sets/updates scoring-related keys.
    """
    item["decision"] = scoring.get("decision", item.get("decision", "skip"))
    item["selected_resume_id"] = scoring.get("selected_resume_id", item.get("selected_resume_id", "default"))

    item["scores"] = {
        "hybrid_score": scoring.get("hybrid_score", 0.0),
        "semantic_similarity": scoring.get("semantic_similarity", 0.0),
        "penalty": scoring.get("penalty", 0.0),
        "reward": scoring.get("reward", 0.0),
        "rule_score": scoring.get("rule_score", 0.0),
        "domain_distance": scoring.get("domain_distance", 0),
    }

    item["reasons"] = scoring.get("reasons", []) or []
    item["top_resume_scores"] = scoring.get("top_resume_scores", []) or []
    item["scoring"] = scoring  # full blob


# ============================================================
# Optional Playwright fallback
# ============================================================
async def _resolve_url_via_playwright(job_id: str, storage_state: Optional[Path]) -> Optional[str]:
    """
    Best-effort URL resolution:
    - load projektboerse with query=<job_id>
    - pick a /projekt/ href containing -<job_id> or job_id
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except Exception:
        print("[inject][WARN] Playwright not available; cannot resolve URL.", file=sys.stderr)
        return None

    search_url = f"https://www.freelancermap.de/projektboerse.html?query={job_id}&sort=1"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        ctx_kwargs: Dict[str, Any] = {"locale": "de-DE"}
        if storage_state and storage_state.exists():
            ctx_kwargs["storage_state"] = str(storage_state)
        context = await browser.new_context(**ctx_kwargs)
        page = await context.new_page()

        await page.goto(search_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)

        hrefs = await page.eval_on_selector_all(
            "a[href*='/projekt/']",
            "els => els.map(e => e.href).filter(Boolean)"
        )

        await context.close()
        await browser.close()

    if not isinstance(hrefs, list):
        return None

    for h in hrefs:
        if isinstance(h, str) and (f"-{job_id}" in h or h.endswith(job_id)):
            return h

    for h in hrefs:
        if isinstance(h, str) and job_id in h:
            return h

    return None


def _fallback_note(job_id: str, url: str) -> None:
    print(
        f"[inject][WARN] job_id={job_id} not found in main queue and no profile exists.\n"
        f"[inject][WARN] Resolved URL: {url}\n"
        f"[inject][WARN] This script does not auto-build work_items yet.\n"
        f"[inject][WARN] Next step: either run your normal pipeline to create work_items/{job_id}.profile.json,\n"
        f"[inject][WARN] or wire explicit callable entrypoints from fetch_job_details/extract_job_profile here.",
        file=sys.stderr
    )


# ============================================================
# Main
# ============================================================
@dataclass
class InjectResult:
    job_id: str
    injected: bool
    scored: bool
    reason: str


def run(
    job_ids: List[str],
    *,
    queue_path: Path,
    injected_path: Path,
    work_items_dir: Path,
    resumes_dir: Path,
    enable_playwright_fallback: bool,
    storage_state: Optional[Path],
) -> List[InjectResult]:
    main_q = _read_json(queue_path)
    inj_q = _read_json(injected_path) if injected_path.exists() else {"items_by_job_id": {}, "meta": {}}

    _ensure_queue_shape(main_q)
    _ensure_queue_shape(inj_q)

    main_items = _queue_items(main_q)
    inj_items = _queue_items(inj_q)

    resume_paths = _discover_resumes(resumes_dir)
    if not resume_paths:
        print(f"[inject][WARN] No resume jsons found in: {resumes_dir.resolve()}", file=sys.stderr)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    results: List[InjectResult] = []

    for raw in job_ids:
        job_id = _norm_job_id(raw)

        # 1) Copy from main queue if present
        src_item = main_items.get(job_id)
        if src_item is not None:
            inj_items[job_id] = copy.deepcopy(src_item)
            injected = True
            inject_reason = "copied_from_main_queue"
        else:
            # if already in injected, keep it; otherwise placeholder
            if job_id in inj_items:
                injected = True
                inject_reason = "already_in_injected_queue"
            else:
                inj_items[job_id] = {
                    "job_id": job_id,
                    "source": "freelancermap",
                    "status": "injected",
                    "artifacts": {
                        "work_item_path": str((work_items_dir / f"{job_id}.json").as_posix()),
                        "profile_path": str((work_items_dir / f"{job_id}.profile.json").as_posix()),
                        "draft_path": None,
                        "run_log_path": "",
                    },
                }
                injected = True
                inject_reason = "placeholder_not_found_in_main_queue"

        item = inj_items[job_id]

        # 2) Load profile for scoring
        try:
            job_profile, profile_path = _load_job_profile_from_item(item, work_items_dir)
        except FileNotFoundError as e:
            if enable_playwright_fallback:
                try:
                    import asyncio

                    url = asyncio.run(_resolve_url_via_playwright(job_id, storage_state))
                except Exception as ex:
                    results.append(InjectResult(job_id, injected, False, f"profile_missing_playwright_failed: {ex}"))
                    continue

                if url:
                    _fallback_note(job_id, url)
                    results.append(InjectResult(job_id, injected, False, "profile_missing_needs_pipeline_or_wiring"))
                else:
                    results.append(InjectResult(job_id, injected, False, "profile_missing_and_url_not_resolved"))
            else:
                results.append(InjectResult(job_id, injected, False, f"profile_missing: {e}"))
            continue

        # 3) Score and apply results
        try:
            scoring = score_job.score_job(job_profile, resume_paths=resume_paths, model=model)
            _apply_scoring_to_item(item, scoring)
            results.append(InjectResult(job_id, injected, True, f"{inject_reason}; scored_using={profile_path}"))
        except Exception as e:
            results.append(InjectResult(job_id, injected, False, f"{inject_reason}; scoring_failed: {e}"))

    # Persist injected queue
    _write_json(injected_path, inj_q)
    return results


def _parse_cli(argv: List[str]) -> Dict[str, Any]:
    """
    Minimal CLI parsing (kept self-contained without argparse).
    Supported:
      --queue <path>
      --injected <path>
      --work-items <path>
      --resumes-dir <path>
      --no-playwright
    """
    out: Dict[str, Any] = {
        "queue_path": QUEUE_PATH,
        "injected_path": INJECTED_QUEUE_PATH,
        "work_items_dir": WORK_ITEMS_DIR,
        "resumes_dir": RESUMES_DIR,
        "enable_playwright_fallback": ENABLE_PLAYWRIGHT_FALLBACK,
    }

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--queue" and i + 1 < len(argv):
            out["queue_path"] = Path(argv[i + 1])
            i += 2
        elif a == "--injected" and i + 1 < len(argv):
            out["injected_path"] = Path(argv[i + 1])
            i += 2
        elif a == "--work-items" and i + 1 < len(argv):
            out["work_items_dir"] = Path(argv[i + 1])
            i += 2
        elif a == "--resumes-dir" and i + 1 < len(argv):
            out["resumes_dir"] = Path(argv[i + 1])
            i += 2
        elif a == "--no-playwright":
            out["enable_playwright_fallback"] = False
            i += 1
        else:
            print(f"[inject][WARN] Unknown arg ignored: {a}", file=sys.stderr)
            i += 1

    return out


def main() -> int:
    if not JOB_IDS:
        print("[inject] JOB_IDS is empty. Add job ids at the top of the file.", file=sys.stderr)
        return 2

    cfg = _parse_cli(sys.argv[1:])

    results = run(
        job_ids=JOB_IDS,
        queue_path=cfg["queue_path"],
        injected_path=cfg["injected_path"],
        work_items_dir=cfg["work_items_dir"],
        resumes_dir=cfg["resumes_dir"],
        enable_playwright_fallback=cfg["enable_playwright_fallback"],
        storage_state=STORAGE_STATE_PATH if STORAGE_STATE_PATH.exists() else None,
    )

    ok = 0
    for r in results:
        print(f"[inject] job_id={r.job_id} injected={r.injected} scored={r.scored} :: {r.reason}")
        if r.scored:
            ok += 1

    print(f"[inject] done. scored {ok}/{len(results)}. updated: {cfg['injected_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())