import json
from pathlib import Path
from datetime import datetime, timezone

QUEUE_PATH = Path("queue/queue.json")

TERMINAL = {"skipped", "rejected", "submitted"}
HUMAN_GATES = {"awaiting_human_decision", "awaiting_final_review", "draft_generated"}
EARLY = {"", "discovered", "profile_extracted", "scored"}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def main():
    if not QUEUE_PATH.exists():
        print("Queue not found.")
        return

    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    items = queue.get("items_by_job_id", {})

    updated = 0

    for job_id, item in items.items():
        lifecycle = (item.get("lifecycle") or "").strip()

        if lifecycle in TERMINAL or lifecycle in HUMAN_GATES:
            continue

        decision = (item.get("decision") or "").strip().lower()
        scores = item.get("scores") or {}

        new_lifecycle = lifecycle

        if lifecycle in EARLY:
            if decision == "skip":
                new_lifecycle = "skipped"
            elif decision in {"review", "pursue"}:
                new_lifecycle = "awaiting_human_decision"
            elif scores:
                new_lifecycle = "scored"
            else:
                new_lifecycle = "discovered"

        if new_lifecycle != lifecycle:
            item["lifecycle"] = new_lifecycle
            item["lifecycle_updated_at"] = now_iso()
            updated += 1

    if updated:
        queue["meta"]["updated_at"] = now_iso()
        QUEUE_PATH.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Updated {updated} items.")


if __name__ == "__main__":
    main()
