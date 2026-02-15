# queue_db.py
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path) -> Iterable[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def _extract_index_fields(item: Dict[str, Any]) -> Dict[str, str]:
    # decision may live in item["decision"] or item["scoring"]["decision"]
    scoring = item.get("scoring") or {}
    decision = (item.get("decision") or scoring.get("decision") or "").strip()

    return {
        "job_id": str(item.get("job_id") or ""),
        "lifecycle": str(item.get("lifecycle") or ""),
        "status": str(item.get("status") or ""),
        "decision": str(decision),
        "source": str(item.get("source") or ""),
        "captured_at": str(item.get("captured_at") or ""),
        "updated_at": str(item.get("updated_at") or ""),
        "published_at": str(item.get("published_at") or ""),
    }


def upsert_item(db_path: Path, item: Dict[str, Any]) -> None:
    idx = _extract_index_fields(item)
    if not idx["job_id"]:
        raise ValueError("upsert_item: missing job_id")

    # Always ensure updated_at is set
    if not idx["updated_at"]:
        idx["updated_at"] = now_iso()
        item = dict(item)
        item["updated_at"] = idx["updated_at"]

    payload = json.dumps(item, ensure_ascii=False)

    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO queue_items (
              job_id, lifecycle, status, decision, source,
              captured_at, updated_at, published_at, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
              lifecycle=excluded.lifecycle,
              status=excluded.status,
              decision=excluded.decision,
              source=excluded.source,
              captured_at=excluded.captured_at,
              updated_at=excluded.updated_at,
              published_at=excluded.published_at,
              payload_json=excluded.payload_json
            """,
            (
                idx["job_id"],
                idx["lifecycle"],
                idx["status"],
                idx["decision"],
                idx["source"],
                idx["captured_at"],
                idx["updated_at"],
                idx["published_at"],
                payload,
            ),
        )
        conn.commit()


def get_item(db_path: Path, job_id: str) -> Optional[Dict[str, Any]]:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM queue_items WHERE job_id=?",
            (str(job_id),),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["payload_json"])


def patch_item(db_path: Path, job_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    existing = get_item(db_path, job_id)
    if not existing:
        return None

    patch = dict(patch)
    # Merge artifacts dict if provided
    if "artifacts" in patch and isinstance(patch["artifacts"], dict):
        merged = dict(existing.get("artifacts") or {})
        merged.update(patch["artifacts"])
        patch["artifacts"] = merged

    updated = dict(existing)
    updated.update(patch)
    updated["job_id"] = str(updated.get("job_id") or job_id)
    updated["updated_at"] = now_iso()

    upsert_item(db_path, updated)
    return updated


def list_items(db_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with connect(db_path) as conn:
        rows = conn.execute("SELECT job_id, payload_json FROM queue_items").fetchall()
        for r in rows:
            out[str(r["job_id"])] = json.loads(r["payload_json"])
    return out


def get_seen_urls(db_path: Path) -> Set[str]:
    # url is inside payload_json; query + parse (OK once per run)
    urls: Set[str] = set()
    with connect(db_path) as conn:
        rows = conn.execute("SELECT payload_json FROM queue_items").fetchall()
    for r in rows:
        try:
            it = json.loads(r["payload_json"])
            u = (it.get("url") or "").strip()
            if u:
                urls.add(u)
        except Exception:
            continue
    return urls
