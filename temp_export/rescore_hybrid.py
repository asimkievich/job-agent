# rescore_hybrid.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def new_hybrid(rule_score: float, semantic_similarity: float, domain_distance: int) -> float:
    # Keep the same normalization convention you used before
    rule_norm = clamp01(rule_score / 0.8)
    sim01 = clamp01(semantic_similarity)
    domain_bonus = clamp01(1.0 - (float(domain_distance) / 3.0))  # 1 best .. 0 worst
    hybrid = (0.65 * sim01) + (0.25 * rule_norm) + (0.10 * domain_bonus)
    return clamp01(hybrid)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def update_run_file(obj: Dict[str, Any]) -> Tuple[float, float]:
    """
    Returns: (old_hybrid, new_hybrid)
    """
    scoring = obj.get("scoring", {}) or {}
    rule = float(scoring.get("rule_score", 0.0) or 0.0)
    sim = float(scoring.get("semantic_similarity", 0.0) or 0.0)
    dom = int(scoring.get("domain_distance", 3) or 3)

    old = float(scoring.get("hybrid_score", 0.0) or 0.0)
    new = new_hybrid(rule, sim, dom)

    # Update scoring.hybrid_score
    scoring["hybrid_score"] = round(new, 3)
    obj["scoring"] = scoring

    # Update top_resume_scores[*].hybrid_score
    trs = obj.get("top_resume_scores", [])
    if isinstance(trs, list):
        for r in trs:
            if not isinstance(r, dict):
                continue
            rr = float(r.get("rule_score", 0.0) or 0.0)
            ss = float(r.get("semantic_similarity", 0.0) or 0.0)
            dd = int(r.get("domain_distance", 3) or 3)
            r["hybrid_score"] = round(new_hybrid(rr, ss, dd), 3)

    # Update decision using the run file's stored parameters
    thr = float(obj.get("auto_threshold", 0.55) or 0.55)
    max_dom = int(obj.get("max_domain_distance", 3) or 3)
    obj["decision"] = "pursue" if (new >= thr and dom < max_dom) else "skip"

    return old, new


def update_queue_file(obj: Dict[str, Any]) -> Tuple[int, int]:
    """
    Returns: (items_updated, items_total)
    """
    items = obj.get("items_by_job_id", {}) or {}
    if not isinstance(items, dict):
        return (0, 0)

    updated = 0
    total = 0

    for _, it in items.items():
        if not isinstance(it, dict):
            continue
        total += 1

        scores = it.get("scores", {}) or {}
        rule = float(scores.get("rule_score", 0.0) or 0.0)
        sim = float(scores.get("semantic_similarity", 0.0) or 0.0)
        dom = int(scores.get("domain_distance", 3) or 3)

        new = new_hybrid(rule, sim, dom)
        scores["hybrid_score"] = round(new, 3)
        it["scores"] = scores

        # Update top_resume_scores
        trs = it.get("top_resume_scores", [])
        if isinstance(trs, list):
            for r in trs:
                if not isinstance(r, dict):
                    continue
                rr = float(r.get("rule_score", 0.0) or 0.0)
                ss = float(r.get("semantic_similarity", 0.0) or 0.0)
                dd = int(r.get("domain_distance", 3) or 3)
                r["hybrid_score"] = round(new_hybrid(rr, ss, dd), 3)

        # Update decision only (never status)
        # Queue items don't store threshold; keep default consistent with your system for now.
        it["decision"] = "pursue" if (new >= 0.55 and dom < 3) else "skip"

        updated += 1

    # Touch meta.updated_at if present
    meta = obj.get("meta", {}) or {}
    if isinstance(meta, dict):
        # don't require time_utils; keep it simple & non-invasive
        meta["rescored_hybrid"] = True
        obj["meta"] = meta

    return (updated, total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs", help="Directory containing run_*.json files")
    ap.add_argument("--queue-path", default="queue/queue.json", help="Path to queue.json")
    ap.add_argument("--apply", action="store_true", help="Actually write changes (otherwise dry-run)")
    ap.add_argument("--backup", action="store_true", help="Write .bak backups before overwriting")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    queue_path = Path(args.queue_path)

    run_files = sorted(runs_dir.glob("run_*.json"))
    print(f"[rescore] found {len(run_files)} run files in {runs_dir}")

    changed = 0
    for p in run_files:
        obj = read_json(p)
        old, new = update_run_file(obj)
        if round(old, 3) != round(new, 3):
            changed += 1
        if args.apply:
            if args.backup:
                write_json(p.with_suffix(p.suffix + ".bak"), read_json(p))
            write_json(p, obj)

    print(f"[rescore] runs changed: {changed}/{len(run_files)}")

    if queue_path.exists():
        q = read_json(queue_path)
        upd, total = update_queue_file(q)
        print(f"[rescore] queue items updated: {upd}/{total}")
        if args.apply:
            if args.backup:
                write_json(queue_path.with_suffix(queue_path.suffix + ".bak"), read_json(queue_path))
            write_json(queue_path, q)
    else:
        print(f"[rescore] queue not found at {queue_path} (skipped)")

    if not args.apply:
        print("[rescore] DRY-RUN only. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
