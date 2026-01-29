# batch_run.py
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from playwright.async_api import async_playwright

import extract_job_profile
import fetch_jobs
import prepare_application_draft
import score_job
import queue_utils

AUTH_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"

SEEN_PATH = Path("data/seen_jobs.json")
Path("data").mkdir(exist_ok=True)

WORK_DIR = Path("work_items")
WORK_DIR.mkdir(exist_ok=True)

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

PROFILES_DIR = Path("profiles/resumes")
SEARCH_PROFILE_PATH = PROFILES_DIR / "search_profile.json"

# --- AUTO DECISION THRESHOLDS (tune later) ---
HYBRID_AUTO_THRESHOLD = 0.60
MAX_DOMAIN_DISTANCE = 2
TOP_N_SUMMARY = 10

# For coverage: allow many, but stop early once we hit mostly-seen jobs.
FETCH_MAX_JOBS = 300
FETCH_MAX_PAGES = 15
FETCH_STOP_AFTER_SEEN_STREAK = 50


def short(s: str, n: int = 90) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def print_summary(results: List[Dict[str, Any]], top_n: int = TOP_N_SUMMARY) -> None:
    if not results:
        print("\nNo new jobs processed.")
        return

    ranked = sorted(results, key=lambda r: float(r.get("hybrid_score", 0.0)), reverse=True)
    top = ranked[:top_n]

    print("\n" + "=" * 100)
    print(f"TOP {len(top)} JOBS (by hybrid_score)")
    print("=" * 100)

    for i, r in enumerate(top, start=1):
        print(
            f"{i:>2}. {float(r['hybrid_score']):.3f} | dom={r['domain_distance']} | {r['decision']:<6} | "
            f"{short(r.get('title', ''), 55)}"
        )
        print(f"    {r.get('url', '')}")
        if r.get("selected_resume_id"):
            print(f"    resume: {r['selected_resume_id']}")
        reasons = r.get("reasons") or []
        if reasons:
            print(f"    reasons: {short('; '.join(reasons), 160)}")
        top3 = r.get("top_resume_scores") or []
        if top3:
            top3_str = ", ".join([f"{x['resume_id']}={x['hybrid_score']:.3f}" for x in top3])
            print(f"    top3: {top3_str}")

    counts: Dict[str, int] = {}
    for r in results:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1

    print("\nCounts:", ", ".join([f"{k}={v}" for k, v in sorted(counts.items())]))
    print("=" * 100)


def load_seen() -> Dict[str, Any]:
    if not SEEN_PATH.exists():
        return {"seen_urls": {}, "last_run_at": None}
    return json.loads(SEEN_PATH.read_text(encoding="utf-8"))


def save_seen(seen: Dict[str, Any]) -> None:
    seen["last_run_at"] = datetime.now().astimezone().isoformat()
    SEEN_PATH.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_url(url: str) -> str:
    if url.startswith("/"):
        return "https://www.freelancermap.de" + url
    return url.strip()


def resume_id_from_path(p: Path) -> str:
    return p.stem  # resume_product_owner_ai_pm.json -> resume_product_owner_ai_pm


def discover_resume_paths() -> List[Path]:
    if not PROFILES_DIR.exists():
        raise RuntimeError(
            f"Profiles dir not found: {PROFILES_DIR}. "
            "Since profiles are private and gitignored, ensure they exist locally."
        )

    resume_paths = sorted(PROFILES_DIR.glob("resume_*.json"))
    if not resume_paths:
        raise RuntimeError(f"No resume profiles found in {PROFILES_DIR} (expected resume_*.json).")

    if not SEARCH_PROFILE_PATH.exists():
        raise RuntimeError(f"Search profile not found: {SEARCH_PROFILE_PATH}")

    return resume_paths


async def capture_work_item(url: str, selected_resume_id: str) -> Dict[str, Any]:
    if not AUTH_STATE_PATH.exists():
        raise RuntimeError("Auth state not found. Run login_and_save_state.py first.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=str(AUTH_STATE_PATH), locale="de-DE")
        page = await context.new_page()

        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(800)

        title = (await page.title()).strip()
        body_text = (await page.locator("body").inner_text()).strip()

        await browser.close()

    return {
        "meta": {
            "created_at": datetime.now().astimezone().isoformat(),
            "source": "freelancermap",
        },
        "job": {
            "url": url,
            "title": title,
            "body_text": body_text,
        },
        "decision": {
            "selected_resume_profile_id": selected_resume_id,
            "notes": "batch_capture_auto",
        },
    }


def run_extract_profile(work_item_path: Path, profile_out_path: Path) -> Dict[str, Any]:
    extract_job_profile.IN_PATH = work_item_path
    extract_job_profile.OUT_PATH = profile_out_path
    extract_job_profile.main()
    return json.loads(profile_out_path.read_text(encoding="utf-8"))


def auto_decide(scoring: Dict[str, Any]) -> str:
    if int(scoring["domain_distance"]) > MAX_DOMAIN_DISTANCE:
        return "skip"
    return "pursue" if float(scoring["hybrid_score"]) >= HYBRID_AUTO_THRESHOLD else "skip"


def run_draft(work_item_path: Path, resume_path: Path, draft_out_path: Path) -> Dict[str, Any]:
    prepare_application_draft.WORK_ITEM = work_item_path
    prepare_application_draft.RESUME_PROFILE = resume_path
    prepare_application_draft.OUT_PATH = draft_out_path
    prepare_application_draft.main()
    return json.loads(draft_out_path.read_text(encoding="utf-8"))


def run_score_with_model(model: Any, profile_path: Path, resume_path: Path, search_path: Path) -> Dict[str, Any]:
    score_job.JOB_PROFILE = profile_path
    score_job.RESUME = resume_path
    score_job.SEARCH = search_path

    job = score_job.load(profile_path)
    resume = score_job.load(resume_path)
    search = score_job.load(search_path)

    rs = score_job.rule_score(job, search)

    from sentence_transformers import util

    emb_resume = model.encode(score_job.build_resume_text(resume), convert_to_tensor=True, normalize_embeddings=True)
    emb_job = model.encode(score_job.build_job_text(job), convert_to_tensor=True, normalize_embeddings=True)
    sim = float(util.cos_sim(emb_resume, emb_job).item())

    rule_norm = max(0.0, min(1.0, (rs["rule_score"] / 0.8)))
    hybrid = 0.6 * rule_norm + 0.4 * max(0.0, min(1.0, sim))

    return {
        "rule_score": rs["rule_score"],
        "semantic_similarity": round(sim, 3),
        "hybrid_score": round(hybrid, 3),
        "domain_distance": rs["domain_distance"],
        "reasons": rs["reasons"],
    }


@dataclass
class ResumeScore:
    resume_path: Path
    resume_id: str
    scoring: Dict[str, Any]


def score_against_all_resumes(
    model: Any, profile_path: Path, resume_paths: List[Path], search_path: Path
) -> Tuple[ResumeScore, List[ResumeScore]]:
    all_scores: List[ResumeScore] = []
    for rp in resume_paths:
        s = run_score_with_model(model, profile_path, rp, search_path)
        all_scores.append(ResumeScore(resume_path=rp, resume_id=resume_id_from_path(rp), scoring=s))

    all_scores_sorted = sorted(all_scores, key=lambda x: float(x.scoring["hybrid_score"]), reverse=True)
    best = all_scores_sorted[0]
    return best, all_scores_sorted


def top_resume_scores_payload(all_scores_sorted: List[ResumeScore], k: int = 3) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for rs in all_scores_sorted[:k]:
        payload.append(
            {
                "resume_id": rs.resume_id,
                "hybrid_score": float(rs.scoring["hybrid_score"]),
                "semantic_similarity": float(rs.scoring["semantic_similarity"]),
                "rule_score": float(rs.scoring["rule_score"]),
                "domain_distance": int(rs.scoring["domain_distance"]),
            }
        )
    return payload


async def main() -> None:
    run_started_at = datetime.now().astimezone()
    results: List[Dict[str, Any]] = []

    seen = load_seen()
    resume_paths = discover_resume_paths()

    queue = queue_utils.load_queue()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    jobs = await fetch_jobs.fetch_jobs_async(
        max_jobs=FETCH_MAX_JOBS,
        seen_urls=set(seen["seen_urls"].keys()),
        max_pages=FETCH_MAX_PAGES,
        stop_after_seen_streak=FETCH_STOP_AFTER_SEEN_STREAK,
    )
    urls = [normalize_url(j["href"]) for j in jobs if "href" in j]

    new_urls = [u for u in urls if u not in seen["seen_urls"]]
    print(f"Found {len(urls)} jobs, {len(new_urls)} new/unseen.")

    for url in new_urls:
        print("\n---")
        print(f"Processing: {url}")

        placeholder_resume_id = resume_id_from_path(resume_paths[0])
        work_item = await capture_work_item(url, selected_resume_id=placeholder_resume_id)

        tmp_path = WORK_DIR / "tmp_job.json"
        tmp_path.write_text(json.dumps(work_item, ensure_ascii=False, indent=2), encoding="utf-8")

        tmp_profile_path = WORK_DIR / "tmp_job.profile.json"
        profile = run_extract_profile(tmp_path, tmp_profile_path)

        job_id = str(profile.get("job_id") or "unknown_job")
        work_item_path = WORK_DIR / f"{job_id}.json"
        profile_path = WORK_DIR / f"{job_id}.profile.json"

        work_item_path.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")
        profile_path.write_text(tmp_profile_path.read_text(encoding="utf-8"), encoding="utf-8")
        tmp_path.unlink(missing_ok=True)
        tmp_profile_path.unlink(missing_ok=True)

        best, all_scores_sorted = score_against_all_resumes(model, profile_path, resume_paths, SEARCH_PROFILE_PATH)
        scoring = best.scoring
        selected_resume_path = best.resume_path
        selected_resume_id = best.resume_id

        decision = auto_decide(scoring)

        print(
            f"Best resume: {selected_resume_id} | Hybrid: {scoring['hybrid_score']} | "
            f"DomainDist: {scoring['domain_distance']} | Auto: {decision}"
        )

        draft_path: Optional[Path] = None
        if decision == "pursue":
            draft_path = WORK_DIR / f"{job_id}.application_draft.json"
            run_draft(work_item_path, selected_resume_path, draft_path)
            print(f"Draft saved: {draft_path}")

        top_scores = top_resume_scores_payload(all_scores_sorted, k=3)

        run_log = {
            "job_id": job_id,
            "job_url": url,
            "captured_at": work_item["meta"]["created_at"],
            "auto_threshold": HYBRID_AUTO_THRESHOLD,
            "max_domain_distance": MAX_DOMAIN_DISTANCE,
            "selected_resume_id": selected_resume_id,
            "selected_resume_path": str(selected_resume_path),
            "scoring": scoring,
            "top_resume_scores": top_scores,
            "decision": decision,
            "draft_path": str(draft_path) if draft_path else None,
        }
        run_path = RUNS_DIR / f"run_{job_id}.json"
        run_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Run log saved: {run_path}")

        results.append(
            {
                "job_id": job_id,
                "url": url,
                "title": work_item["job"]["title"],
                "hybrid_score": float(scoring["hybrid_score"]),
                "domain_distance": int(scoring["domain_distance"]),
                "decision": decision,
                "reasons": scoring.get("reasons", []),
                "selected_resume_id": selected_resume_id,
                "top_resume_scores": top_scores,
                "draft_path": str(draft_path) if draft_path else None,
            }
        )

        queue_utils.upsert_queue_item(
            queue=queue,
            job_id=job_id,
            source=work_item["meta"]["source"],
            url=url,
            title=work_item["job"]["title"],
            captured_at=work_item["meta"]["created_at"],
            decision=decision,
            selected_resume_id=selected_resume_id,
            scoring=scoring,
            reasons=scoring.get("reasons", []),
            top_resume_scores=top_scores,
            work_item_path=str(work_item_path),
            profile_path=str(profile_path),
            draft_path=str(draft_path) if draft_path else None,
            run_log_path=str(run_path),
        )

        seen["seen_urls"][url] = {
            "job_id": job_id,
            "first_seen_at": datetime.now().astimezone().isoformat(),
            "decision": decision,
            "hybrid_score": float(scoring["hybrid_score"]),
            "selected_resume_id": selected_resume_id,
        }
        save_seen(seen)

    # Persist queue and summary artifacts
    queue_utils.save_queue(queue)

    print_summary(results, top_n=TOP_N_SUMMARY)

    summary = {
        "run_started_at": run_started_at.isoformat(),
        "run_finished_at": datetime.now().astimezone().isoformat(),
        "total_found": len(urls),
        "new_processed": len(results),
        "top_n": TOP_N_SUMMARY,
        "results": sorted(results, key=lambda r: float(r.get("hybrid_score", 0.0)), reverse=True),
    }
    summary_path = RUNS_DIR / f"summary_{run_started_at.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")
    print("Queue saved: queue/queue.json")

    print("\nOK: batch run complete.")


if __name__ == "__main__":
    asyncio.run(main())
