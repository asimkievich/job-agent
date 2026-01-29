# queue_utils.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

QUEUE_DIR = Path("queue")
QUEUE_PATH = QUEUE_DIR / "queue.json"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _ensure_queue_dir() -> None:
    QUEUE_DIR.mkdir(exist_ok=True)


def load_queue() -> Dict[str, Any]:
    _ensure_queue_dir()
    if not QUEUE_PATH.exists():
        return {
            "meta": {
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "version": 1,
            },
            "items_by_job_id": {},
        }
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def save_queue(queue: Dict[str, Any]) -> None:
    _ensure_queue_dir()
    queue.setdefault("meta", {})
    queue["meta"]["updated_at"] = _now_iso()
    QUEUE_PATH.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_queue_item(
    *,
    queue: Dict[str, Any],
    job_id: str,
    source: str,
    url: str,
    title: str,
    captured_at: str,
    decision: str,
    selected_resume_id: str,
    scoring: Dict[str, Any],
    reasons: list[str],
    top_resume_scores: list[Dict[str, Any]],
    work_item_path: str,
    profile_path: str,
    draft_path: Optional[str],
    run_log_path: str,
    status_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Insert/update a queue item keyed by job_id.
    Status policy:
      - pursue -> pending
      - skip   -> rejected
    Caller can override via status_override (e.g., "hold", "pending").
    """
    items = queue.setdefault("items_by_job_id", {})
    existing = items.get(job_id, {})

    # Default status mapping
    status = status_override
    if not status:
        status = "pending" if decision == "pursue" else "rejected"

    item = {
        "job_id": job_id,
        "source": source,
        "url": url,
        "title": title,
        "captured_at": captured_at,
        "status": status,
        "decision": decision,
        "selected_resume_id": selected_resume_id,
        "scores": {
            "hybrid_score": float(scoring.get("hybrid_score", 0.0)),
            "domain_distance": int(scoring.get("domain_distance", 99)),
            "rule_score": float(scoring.get("rule_score", 0.0)),
            "semantic_similarity": float(scoring.get("semantic_similarity", 0.0)),
        },
        "reasons": reasons or [],
        "top_resume_scores": top_resume_scores or [],
        "artifacts": {
            "work_item_path": work_item_path,
            "profile_path": profile_path,
            "draft_path": draft_path,
            "run_log_path": run_log_path,
        },
        "updated_at": _now_iso(),
    }

    # Preserve human actions/notes if already present
    # (UI can add these later; batch should not clobber them.)
    for k in ("human_notes", "human_actions", "approved_at", "rejected_at"):
        if k in existing and k not in item:
            item[k] = existing[k]
        elif k in existing:
            item[k] = existing[k]

    items[job_id] = item
    return item
