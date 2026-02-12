from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sentence_transformers import SentenceTransformer

import score_job  # your local score_job.py


QUEUE_INJECTED = Path("queue/queue_injected.json")
WORK_ITEMS_DIR = Path("work_items")
RESUMES_DIR = Path("profiles/resumes")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_resume_paths() -> List[Path]:
    # Same behavior as batch_run.py: *.json + filename contains "resume"
    return sorted(
        p for p in RESUMES_DIR.glob("*.json")
        if p.is_file() and "resume" in p.name.lower()
    )


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{ts}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    return backup_path


def find_job_ids(queue_obj: Dict[str, Any]) -> List[str]:
    """
    Supports your typical queue schema where jobs live under items_by_job_id.
    Falls back to scanning other plausible shapes without crashing.
    """
    if isinstance(queue_obj, dict) and "items_by_job_id" in queue_obj:
        iby = queue_obj.get("items_by_job_id") or {}
        if isinstance(iby, dict):
            return list(iby.keys())

    # fallback: try "jobs" list
    if isinstance(queue_obj, dict) and isinstance(queue_obj.get("jobs"), list):
        job_ids = []
        for it in queue_obj["jobs"]:
            if isinstance(it, dict) and it.get("job_id"):
                job_ids.append(str(it["job_id"]))
        return job_ids

    return []


def get_item_ref(queue_obj: Dict[str, Any], job_id: str) -> Dict[str, Any] | None:
    iby = queue_obj.get("items_by_job_id")
    if isinstance(iby, dict) and job_id in iby and isinstance(iby[job_id], dict):
        return iby[job_id]
    return None


def load_job_profile(job_id: str) -> Dict[str, Any] | None:
    prof = WORK_ITEMS_DIR / f"{job_id}.profile.json"
    if not prof.exists():
        return None
    obj = load_json(prof)
    return obj if isinstance(obj, dict) else None


def apply_scoring_to_item(item: Dict[str, Any], result: Dict[str, Any]) -> None:
    # Top-level outputs
    item["decision"] = result.get("decision")
    item["selected_resume_id"] = result.get("selected_resume_id")
    item["top_resume_scores"] = result.get("top_resume_scores", [])
    item["reasons"] = result.get("reasons", [])

    # Normalized score fields used by UI/logic
    item["scores"] = item.get("scores") or {}
    if not isinstance(item["scores"], dict):
        item["scores"] = {}

    item["scores"]["hybrid_score"] = float(result.get("hybrid_score", 0.0))
    item["scores"]["semantic_similarity"] = float(result.get("semantic_similarity", 0.0))
    item["scores"]["reward"] = float(result.get("reward", 0.0))
    item["scores"]["penalty"] = float(result.get("penalty", 0.0))

    # Full scoring blob (diff-friendly, keeps debug/version)
    item["scoring"] = result
    item["scoring"]["scored_at"] = datetime.now().isoformat(timespec="seconds")


def main(write_back: bool = True) -> None:
    if not QUEUE_INJECTED.exists():
        raise FileNotFoundError(f"Missing {QUEUE_INJECTED}")

    queue_obj = load_json(QUEUE_INJECTED)
    if not isinstance(queue_obj, dict):
        raise ValueError("queue_injected.json must be a JSON object")

    resume_paths = discover_resume_paths()
    if not resume_paths:
        raise RuntimeError(f"No resumes found in {RESUMES_DIR} (need *.json with 'resume' in filename)")

    # Build model once for speed/consistency
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    job_ids = find_job_ids(queue_obj)
    if not job_ids:
        raise RuntimeError("Could not find job IDs in queue_injected.json (expected items_by_job_id)")

    updated: List[Tuple[str, str, float]] = []
    skipped_no_profile: List[str] = []

    for job_id in job_ids:
        item = get_item_ref(queue_obj, job_id)
        if item is None:
            continue

        job_profile = load_job_profile(job_id)
        if job_profile is None:
            skipped_no_profile.append(job_id)
            continue

        result = score_job.score_job(
            job_profile=job_profile,
            resume_paths=resume_paths,
            model=model,
        )

        apply_scoring_to_item(item, result)
        updated.append((job_id, str(result.get("selected_resume_id")), float(result.get("hybrid_score", 0.0))))

    # Print a compact summary (top score implied by selected)
    updated.sort(key=lambda t: t[2], reverse=True)
    print("\n=== UPDATED (sorted by hybrid_score) ===")
    for job_id, resume_id, h in updated:
        print(f"{job_id:>10}  {h:.3f}  {resume_id}")

    if skipped_no_profile:
        print("\n=== SKIPPED (missing work_items/<job_id>.profile.json) ===")
        for jid in skipped_no_profile:
            print(jid)

    if write_back:
        backup = backup_file(QUEUE_INJECTED)
        print(f"\nBackup written: {backup}")
        save_json(QUEUE_INJECTED, queue_obj)
        print(f"Updated queue written: {QUEUE_INJECTED}")
    else:
        print("\nDry-run only (no file written).")


if __name__ == "__main__":
    main(write_back=True)
