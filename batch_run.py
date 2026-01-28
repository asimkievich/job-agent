import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from playwright.async_api import async_playwright

import fetch_jobs
import extract_job_profile
import score_job
import prepare_application_draft


AUTH_STATE_PATH = Path.home() / "secrets" / "job-agent.storage_state.json"

SEEN_PATH = Path("data/seen_jobs.json")
Path("data").mkdir(exist_ok=True)

WORK_DIR = Path("work_items")
WORK_DIR.mkdir(exist_ok=True)

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# --- AUTO DECISION THRESHOLDS (tune later) ---
HYBRID_AUTO_THRESHOLD = 0.60
MAX_DOMAIN_DISTANCE = 2


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


async def capture_work_item(url: str) -> Dict[str, Any]:
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
            "selected_resume_profile_id": "resume_product_owner_ai_pm",
            "notes": "batch_capture_auto",
        },
    }


def run_extract_profile(work_item_path: Path, profile_out_path: Path) -> Dict[str, Any]:
    extract_job_profile.IN_PATH = work_item_path
    extract_job_profile.OUT_PATH = profile_out_path
    extract_job_profile.main()
    return json.loads(profile_out_path.read_text(encoding="utf-8"))


def run_score(profile_path: Path, resume_path: Path, search_path: Path) -> Dict[str, Any]:
    score_job.JOB_PROFILE = profile_path
    score_job.RESUME = resume_path
    score_job.SEARCH = search_path

    job = score_job.load(profile_path)
    resume = score_job.load(resume_path)
    search = score_job.load(search_path)

    rs = score_job.rule_score(job, search)

    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")

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


def auto_decide(scoring: Dict[str, Any]) -> str:
    if scoring["domain_distance"] > MAX_DOMAIN_DISTANCE:
        return "skip"
    return "pursue" if scoring["hybrid_score"] >= HYBRID_AUTO_THRESHOLD else "skip"


def run_draft(work_item_path: Path, resume_path: Path, draft_out_path: Path) -> Dict[str, Any]:
    prepare_application_draft.WORK_ITEM = work_item_path
    prepare_application_draft.RESUME_PROFILE = resume_path
    prepare_application_draft.OUT_PATH = draft_out_path
    prepare_application_draft.main()
    return json.loads(draft_out_path.read_text(encoding="utf-8"))


async def main():
    seen = load_seen()

    resume_path = Path("profiles/resumes/resume_product_owner_ai_pm.json")
    search_path = Path("profiles/resumes/search_profile.json")

    jobs = await fetch_jobs.fetch_jobs_async(max_jobs=60)
    urls = [normalize_url(j["href"]) for j in jobs if "href" in j]

    new_urls = [u for u in urls if u not in seen["seen_urls"]]
    print(f"Found {len(urls)} jobs, {len(new_urls)} new/unseen.")

    for url in new_urls:
        print("\n---")
        print(f"Processing: {url}")

        work_item = await capture_work_item(url)

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

        scoring = run_score(profile_path, resume_path, search_path)
        decision = auto_decide(scoring)

        print(f"Hybrid: {scoring['hybrid_score']} | DomainDist: {scoring['domain_distance']} | Auto: {decision}")

        draft_path = None
        if decision == "pursue":
            draft_path = WORK_DIR / f"{job_id}.application_draft.json"
            run_draft(work_item_path, resume_path, draft_path)
            print(f"Draft saved: {draft_path}")

        run_log = {
            "job_id": job_id,
            "job_url": url,
            "captured_at": work_item["meta"]["created_at"],
            "auto_threshold": HYBRID_AUTO_THRESHOLD,
            "max_domain_distance": MAX_DOMAIN_DISTANCE,
            "scoring": scoring,
            "decision": decision,
            "draft_path": str(draft_path) if draft_path else None,
        }
        run_path = RUNS_DIR / f"run_{job_id}.json"
        run_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Run log saved: {run_path}")

        seen["seen_urls"][url] = {
            "job_id": job_id,
            "first_seen_at": datetime.now().astimezone().isoformat(),
            "decision": decision,
            "hybrid_score": scoring["hybrid_score"],
        }
        save_seen(seen)

    print("\nOK: batch run complete.")


if __name__ == "__main__":
    asyncio.run(main())
