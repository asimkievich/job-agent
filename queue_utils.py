from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
import queue_db

QUEUE_DIR = Path("queue")
ALLOWED_DECISIONS = {"pursue", "review", "skip"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()

def load_queue(*, db_path: Path) -> Dict[str, Any]:
    # Return same structure as before so callers don’t break.
    items = queue_db.list_items(db_path)
    return {
        "meta": {"created_at": _now_iso(), "updated_at": _now_iso(), "version": 1},
        "items_by_job_id": items,
    }


def save_queue(queue: Dict[str, Any], *, db_path: Path) -> None:
    # Persist all items currently in queue dict (used rarely; prefer enqueue_item/upsert)
    items = queue.get("items_by_job_id") or {}
    if not isinstance(items, dict):
        return
    for _, item in items.items():
        if isinstance(item, dict):
            queue_db.upsert_item(db_path, item)

def patch_queue_item(*, db_path: Path, job_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    updated = queue_db.patch_item(db_path, job_id, patch)
    if not updated:
        raise KeyError(f"patch_queue_item: job_id not found: {job_id}")
    return updated



def set_lifecycle(*, db_path: Path, job_id: str, lifecycle: str) -> Dict[str, Any]:
    return patch_queue_item(
        db_path=db_path,
        job_id=job_id,
        patch={"lifecycle": lifecycle, "lifecycle_updated_at": _now_iso()},
    )



def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _coerce_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _normalize_decision(scoring: Dict[str, Any]) -> str:
    d = (scoring or {}).get("decision")
    if isinstance(d, str):
        d = d.strip().lower()
    if d in ALLOWED_DECISIONS:
        return d
    return "skip"


def _status_from_decision(decision: str) -> str:
    return {
        "pursue": "pending",
        "review": "pending",
        "skip": "skipped",
    }.get(decision, "pending")



def upsert_queue_item(
    *,
    queue: Dict[str, Any],
    job_id: str,
    source: str,
    url: str,
    title: str,
    captured_at: str,
    published_at: str,
    scoring: Dict[str, Any],
    reasons: List[str],
    top_resume_scores: List[Dict[str, Any]],
    work_item_path: str,
    profile_path: str,
    draft_path: Optional[str],
    run_log_path: str,
    status_override: Optional[str] = None,
    lifecycle_override: Optional[str] = None, 
) -> Dict[str, Any]:
    """
    Insert/update a queue item keyed by job_id.

    Single source of truth:
      - decision comes ONLY from scoring["decision"] (score_job.py)
      - displayed scores come from top_resume_scores[0] if present; else scoring

    status:
      - if status_override is provided, use it
      - else mirror decision: pursue->pending, review->hold, skip->rejected
    """
    items = queue.setdefault("items_by_job_id", {})
    existing = items.get(job_id, {})
    lifecycle = (lifecycle_override or existing.get("lifecycle") or "").strip().lower()
    if not lifecycle:
        lifecycle = "discovered"
    scoring = scoring or {}
    top_resume_scores = top_resume_scores or []

    decision = _normalize_decision(scoring)

    # Choose "best" score bundle (prefer explicit best resume row)
    best = (top_resume_scores[0] if top_resume_scores else {}) or {}

    hybrid = _coerce_float(best.get("hybrid_score"), default=None)
    if hybrid is None:
        hybrid = _coerce_float(scoring.get("hybrid_score", 0.0))

    sem = _coerce_float(best.get("semantic_similarity"), default=None)
    if sem is None:
        sem = _coerce_float(scoring.get("semantic_similarity", 0.0))

    # penalty/reward are job-level in your current design
    penalty = _coerce_float(scoring.get("penalty", 0.0))
    reward = _coerce_float(scoring.get("reward", 0.0))

    status = (status_override or "").strip().lower()
    if not status:
        status = _status_from_decision(decision)

    item = {
        "job_id": job_id,
        "source": source,
        "url": url,
        "title": title,
        "captured_at": captured_at,
        "published_at": published_at,
        "status": status,
        "decision": decision,
        "lifecycle": lifecycle,

        # Prefer scorer values; do NOT accept top-level copies as truth
        "selected_resume_id": scoring.get("selected_resume_id") or "default",

        "scores": {
            "hybrid_score": hybrid,
            "semantic_similarity": sem,

            # v2 fields
            "penalty": penalty,
            "reward": reward,

            # legacy fields kept for UI/backward compatibility (remove later)
            "rule_score": _coerce_float(scoring.get("rule_score", 0.0)),
            "domain_distance": _coerce_int(scoring.get("domain_distance", 0)),
        },

        "reasons": reasons or (scoring.get("reasons") or []),

        # Keep per-resume breakdown if present
        "top_resume_scores": top_resume_scores or (scoring.get("top_resume_scores") or []),

        "artifacts": {
            "work_item_path": work_item_path,
            "profile_path": profile_path,
            "draft_path": draft_path,
            "run_log_path": run_log_path,
        },

        # Keep raw scoring for provenance/debugging
        "scoring": scoring,

        "updated_at": _now_iso(),
    }

    # Preserve human actions/notes if already present
    for k in ("human_notes", "human_actions", "approved_at", "rejected_at"):
        if k in existing:
            item[k] = existing[k]

    items[job_id] = item
    return item


def enqueue_item(*, db_path: Path, queue: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter used by batch_run.py.

    Converts the generic item dict produced by batch_run into the canonical queue schema,
    then persists it to SQLite (queue.db / queue_injected.db).
    """
    # Robust job_id extraction
    job_id = (
        item.get("project_id")
        or item.get("Projekt-ID")
        or item.get("Projekt_ID")
        or item.get("projekt_id")
        or item.get("ProjektId")
        or item.get("projectId")
        or item.get("job_id")
        or item.get("normalized_href")
        or item.get("href")
        or "unknown"
    )

    scoring = item.get("scoring") or {}

    # Prefer scoring keys; allow batch payload fallbacks for convenience
    top_resume_scores = (
        item.get("top_resume_scores")
        or scoring.get("top_resume_scores")
        or []
    )

    reasons = (
        item.get("reasons")
        or scoring.get("reasons")
        or []
    )

    item_dict = upsert_queue_item(
        queue=queue,
        job_id=str(job_id),
        source=item.get("source") or "freelancermap",
        url=item.get("href") or item.get("url") or "",
        title=item.get("title") or "",
        captured_at=item.get("captured_at") or item.get("created_at") or _now_iso(),
        published_at=item.get("published_at") or "",
        scoring=scoring,
        reasons=reasons,
        top_resume_scores=top_resume_scores,
        work_item_path=item.get("work_item_path") or "",
        profile_path=item.get("profile_path") or "",
        draft_path=item.get("draft_path"),
        run_log_path=item.get("run_log_path") or "",
        status_override=item.get("status"),
        lifecycle_override=item.get("lifecycle"),
    )

    # NEW: persist to SQLite
    queue_db.upsert_item(db_path, item_dict)
    return item_dict
