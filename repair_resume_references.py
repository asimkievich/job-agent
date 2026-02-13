import json
import os
from pathlib import Path
from difflib import get_close_matches

QUEUE_PATH = Path("queue/queue.json")
RESUME_DIR = Path("profiles/resumes")

def load_queue():
    with open(QUEUE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_queue(doc):
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

def get_existing_resumes():
    return [p.name for p in RESUME_DIR.glob("*.json")]

def repair():
    queue_doc = load_queue()
    items_by_job_id = queue_doc.get("items_by_job_id", {})
    existing_resumes = get_existing_resumes()

    print("Existing resumes:")
    for r in existing_resumes:
        print("  ", r)

    changes = 0

    for job_id, item in items_by_job_id.items():
        selected = item.get("selected_resume_id")

        if not selected:
            continue

        if selected in existing_resumes:
            continue  # already valid

        print(f"\nJob {job_id}: missing resume '{selected}'")

        # Try to find closest filename match
        matches = get_close_matches(selected, existing_resumes, n=1, cutoff=0.4)

        if matches:
            new_resume = matches[0]
            print(f"  → Auto-remapping to '{new_resume}'")
            item["selected_resume_id"] = new_resume
            changes += 1
        else:
            print("  → No close match found. Leaving unchanged.")

    if changes > 0:
        save_queue(queue_doc)
        print(f"\nUpdated {changes} job(s).")
    else:
        print("\nNo changes made.")

if __name__ == "__main__":
    repair()
