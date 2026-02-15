import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import sys
from pathlib import Path


# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROD_JSON = Path("queue/queue.json")
INJ_JSON = Path("queue/queue_injected.json")

from queue_repo import initialize_db, get_connection

def utc_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def backup_file(path: Path) -> Path:
    if not path.exists():
        return path
    bak = path.with_suffix(path.suffix + f".bak_{utc_ts()}")
    shutil.copy2(path, bak)
    return bak


def load_queue_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[migrate] JSON not found: {path} (skipping)")
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Your schema: {"meta": {...}, "items_by_job_id": {...}}
    if isinstance(data, dict) and "items_by_job_id" in data and isinstance(data["items_by_job_id"], dict):
        inner = data["items_by_job_id"]
        items = []
        for job_id, item in inner.items():
            if not isinstance(item, dict):
                continue
            if "job_id" not in item:
                item = dict(item)
                item["job_id"] = str(job_id)
            items.append(item)
        return items

    # Fallbacks (just in case other queue files differ)
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("items", "queue", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(
        f"Unsupported JSON structure in {path}: expected items_by_job_id or list/wrapper-list"
    )


def upsert_many(items: List[Dict[str, Any]], injected: bool) -> int:
    if not items:
        return 0

    initialize_db(injected=injected)
    conn = get_connection(injected=injected)
    cur = conn.cursor()

    sql = """
    INSERT INTO queue_items (
        job_id, lifecycle, status, decision, source,
        captured_at, updated_at, published_at, payload_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(job_id) DO UPDATE SET
        lifecycle=excluded.lifecycle,
        status=excluded.status,
        decision=excluded.decision,
        source=excluded.source,
        captured_at=excluded.captured_at,
        updated_at=excluded.updated_at,
        published_at=excluded.published_at,
        payload_json=excluded.payload_json
    ;
    """

    rows = 0
    for item in items:
        job_id = str(item.get("job_id", "")).strip()
        if not job_id:
            # Skip malformed entries
            continue

        lifecycle = item.get("lifecycle", "") or ""
        status = item.get("status", "") or ""
        decision = item.get("decision", "") or ""
        source = item.get("source", "") or ""
        captured_at = item.get("captured_at", "") or ""
        updated_at = item.get("updated_at", "") or ""
        published_at = item.get("published_at", "") or ""

        payload_json = json.dumps(item, ensure_ascii=False)

        cur.execute(
            sql,
            (
                job_id, lifecycle, status, decision, source,
                captured_at, updated_at, published_at, payload_json
            ),
        )
        rows += 1

    conn.commit()
    conn.close()
    return rows


def count_rows(injected: bool) -> int:
    conn = get_connection(injected=injected)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM queue_items;")
    n = int(cur.fetchone()[0])
    conn.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prod", "injected"], required=True)
    args = ap.parse_args()

    injected = args.mode == "injected"
    json_path = INJ_JSON if injected else PROD_JSON

    print(f"[migrate] mode={args.mode} json={json_path}")

    # Backup JSON (safety)
    bak = backup_file(json_path)
    if json_path.exists():
        print(f"[migrate] backup created: {bak}")

    items = load_queue_json(json_path)
    print(f"[migrate] loaded_items={len(items)}")

    inserted = upsert_many(items, injected=injected)
    print(f"[migrate] upserted_rows={inserted}")

    total = count_rows(injected=injected)
    print(f"[migrate] db_total_rows={total}")

    print("[migrate] done")


if __name__ == "__main__":
    main()
