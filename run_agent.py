import asyncio
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import batch_run

QUEUE_PATH = Path("queue/queue.json")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

TERMINAL_STATES = {"skipped", "rejected", "submitted"}
HUMAN_GATE_STATES = {"awaiting_human_decision", "draft_generated", "awaiting_final_review"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_queue() -> Dict[str, Any]:
    if QUEUE_PATH.exists():
        return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    return {"meta": {"created_at": now_iso(), "updated_at": now_iso(), "version": 1}, "items_by_job_id": {}}


def save_queue(doc: Dict[str, Any]) -> None:
    meta = doc.setdefault("meta", {})
    meta["updated_at"] = now_iso()
    QUEUE_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_lifecycle(doc: Dict[str, Any]) -> int:
    """
    Apply agent-driven lifecycle transitions idempotently:
    - Do NOT override lifecycle if already set (unless blank).
    - Do NOT touch status.
    - Do NOT change terminal or human-gate states.
    - Map decisions:
        skip -> skipped
        review/pursue -> awaiting_human_decision
      (legacy 'candidate' handled too)
    """
    items_by_job_id = doc.get("items_by_job_id")
    if not isinstance(items_by_job_id, dict):
        return 0

    updated = 0
    for job_id, item in items_by_job_id.items():
        if not isinstance(item, dict):
            continue

        lifecycle = (item.get("lifecycle") or "").strip()
        if lifecycle in TERMINAL_STATES or lifecycle in HUMAN_GATE_STATES:
            continue
        if lifecycle and lifecycle not in {"discovered", "profile_extracted", "scored"}:
            # Unknown/non-empty lifecycle -> don't override
            continue

        decision = (item.get("decision") or (item.get("scoring") or {}).get("decision") or "").strip().lower()

        new_lifecycle = None
        if not lifecycle:
            # missing lifecycle
            if decision == "skip":
                new_lifecycle = "skipped"
            elif decision in {"review", "pursue", "candidate"}:
                new_lifecycle = "awaiting_human_decision"
            else:
                new_lifecycle = "discovered"

        if new_lifecycle and new_lifecycle != lifecycle:
            item["lifecycle"] = new_lifecycle
            item["lifecycle_updated_at"] = now_iso()
            updated += 1

    return updated


def write_run_log(payload: Dict[str, Any]) -> Path:
    p = RUNS_DIR / f"run_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def main() -> None:
    run_info: Dict[str, Any] = {
        "started_at": now_iso(),
        "cwd": os.getcwd(),
        "queue_path": str(QUEUE_PATH),
        "ok": False,
    }

    try:
        q_before = load_queue()
        run_info["items_before"] = len((q_before.get("items_by_job_id") or {}))

        # 1) Run the existing ingestion pipeline
        asyncio.run(batch_run.main())

        q_after_ingest = load_queue()
        run_info["items_after_ingest"] = len((q_after_ingest.get("items_by_job_id") or {}))

        # 2) Normalize lifecycle according to state machine (agent transitions only)
        updated = normalize_lifecycle(q_after_ingest)
        run_info["lifecycle_updated"] = updated

        if updated:
            save_queue(q_after_ingest)

        run_info["ok"] = True
        run_info["finished_at"] = now_iso()
        run_info["run_log_path"] = str(write_run_log(run_info))

    except Exception:
        run_info["error"] = traceback.format_exc()
        run_info["finished_at"] = now_iso()
        err_path = RUNS_DIR / f"run_agent_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        err_path.write_text(run_info["error"], encoding="utf-8")
        run_info["error_path"] = str(err_path)
        write_run_log(run_info)
        raise


if __name__ == "__main__":
    main()
