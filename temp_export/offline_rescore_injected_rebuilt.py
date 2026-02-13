from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sentence_transformers import SentenceTransformer

import score_job  # your local score_job.py


QUEUE_INJECTED = Path("queue/queue_injected.json")
WORK_ITEMS_DIR = Path("work_items")
RESUMES_DIR = Path("profiles/resumes")
SEARCH_PROFILE = Path("profiles/resumes/search_profile.json")


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


def _marker_hit(text_lower: str, marker: str) -> bool:
    ml = (marker or "").lower().strip()
    if not ml:
        return False
    if len(ml) <= 3:
        return re.search(rf"\b{re.escape(ml)}\b", text_lower) is not None
    return ml in text_lower


def _is_product_title_strict(title: str) -> bool:
    t = (title or "").lower()
    return bool(
        re.search(r"\bproduct\s+(owner|manager|lead|leader|head)\b", t)
        or re.search(r"\bprodukt\s+(owner|manager|leitung|lead)\b", t)
    )


def _title_only_adjust(item: Dict[str, Any], search_profile: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    If work_items/<job_id>.profile.json is missing, we MUST NOT re-run semantic embeddings.
    We only recompute title-based reward/penalty and then recompute hybrid/decision using
    existing semantic_similarity values already stored in item['top_resume_scores'].
    """
    trs = item.get("top_resume_scores")
    if not isinstance(trs, list) or not trs:
        return None

    cfg = (search_profile or {}).get("simple_scoring_v2", {}) or {}
    max_reward = float(cfg.get("max_reward", 0.10))
    max_penalty = float(cfg.get("max_penalty", 0.10))

    thresholds = cfg.get("thresholds", {}) or {}
    t_review = float(thresholds.get("review", 0.60))
    t_pursue = float(thresholds.get("pursue", 0.75))

    rewards_cfg = cfg.get("rewards", {}) or {}
    penalties_cfg = cfg.get("penalties", {}) or {}

    title = (item.get("title") or "").strip()
    title_lower = title.lower()

    # AI signal (TITLE ONLY) using the same marker criteria (short tokens => word boundary)
    ai_cfg = rewards_cfg.get("ai_ml_markers", {}) or {}
    ai_markers = ai_cfg.get("markers", []) or []
    ai_signal = any(isinstance(m, str) and _marker_hit(title_lower, m) for m in ai_markers)

    # Title-only reward
    reward = 0.0
    reward_reasons: List[str] = []

    pm_cfg = rewards_cfg.get("pm_title_markers", {}) or {}
    pm_bonus = float(pm_cfg.get("bonus", 0.00))
    pm_markers = pm_cfg.get("markers", []) or []

    pm_hit = None
    for m in pm_markers:
        if isinstance(m, str) and m.strip() and m.lower() in title_lower:
            pm_hit = m
            break
    if pm_hit and pm_bonus > 0:
        applied = pm_bonus if pm_bonus <= (max_reward - reward) else (max_reward - reward)
        if applied > 0:
            reward += applied
            reward_reasons.append(f"Reward: PM marker in title (+{applied:.3f})")

    ai_bonus = float(ai_cfg.get("bonus", 0.00))
    if ai_signal and ai_bonus > 0 and reward < max_reward:
        applied = ai_bonus if ai_bonus <= (max_reward - reward) else (max_reward - reward)
        if applied > 0:
            reward += applied
            reward_reasons.append(f"Reward: AI/ML signal (+{applied:.3f})")

    # Title-only penalty: non-product title, but ONLY if NOT an AI role
    penalty = 0.0
    penalty_reasons: List[str] = []

    nonprod_cfg = penalties_cfg.get("non_product_title", {}) or {}
    nonprod_pen = float(nonprod_cfg.get("penalty", 0.00))

    title_is_product_role = _is_product_title_strict(title)

    if (not title_is_product_role) and (not ai_signal) and nonprod_pen > 0:
        applied = nonprod_pen if nonprod_pen <= (max_penalty - penalty) else (max_penalty - penalty)
        if applied > 0:
            penalty += applied
            penalty_reasons.append(f"Penalty: title not a product role (−{applied:.3f})")

    denom = 1.0 + max_reward + max_penalty

    new_trs: List[Dict[str, Any]] = []
    for d in trs:
        if not isinstance(d, dict):
            continue
        sem = float(d.get("semantic_similarity", 0.0) or 0.0)
        raw = sem + reward - penalty
        hybrid = (raw + max_penalty) / denom

        nd = dict(d)
        nd["reward"] = round(reward, 3)
        nd["penalty"] = round(penalty, 3)
        nd["raw_score"] = round(raw, 3)
        nd["rule_score"] = round(reward, 3)
        nd["domain_distance"] = 0
        nd["hybrid_score"] = round(hybrid, 3)
        new_trs.append(nd)

    if not new_trs:
        return None

    new_trs.sort(key=lambda x: float(x.get("hybrid_score", 0.0) or 0.0), reverse=True)
    best = new_trs[0]
    best_h = float(best.get("hybrid_score", 0.0) or 0.0)

    if best_h >= t_pursue:
        decision = "pursue"
    elif best_h >= t_review:
        decision = "review"
    else:
        decision = "skip"

    reasons = reward_reasons + penalty_reasons

    return {
        "decision": decision,
        "selected_resume_id": best.get("resume_id", item.get("selected_resume_id", "default")),
        "top_resume_scores": new_trs,
        "hybrid_score": best_h,
        "semantic_similarity": float(best.get("semantic_similarity", 0.0) or 0.0),
        "rule_score": round(reward, 3),
        "domain_distance": 0,
        "reasons": reasons,
        "reward": round(reward, 3),
        "penalty": round(penalty, 3),
        "scoring_version": "title_only_adjust_v1",
        "debug": {
            "mode": "title_only_adjust",
            "title_used": title,
            "title_is_product_role": title_is_product_role,
            "ai_signal_title_only": ai_signal,
            "nonprod_pen": nonprod_pen,
            "thresholds": {"review": t_review, "pursue": t_pursue},
            "normalization": {"raw_min": -max_penalty, "raw_max": 1.0 + max_reward, "denom": denom},
        },
    }


def main(write_back: bool = True) -> None:
    if not QUEUE_INJECTED.exists():
        raise FileNotFoundError(f"Missing {QUEUE_INJECTED}")

    queue_obj = load_json(QUEUE_INJECTED)
    if not isinstance(queue_obj, dict):
        raise ValueError("queue_injected.json must be a JSON object")

    search_profile = load_json(SEARCH_PROFILE)
    if not isinstance(search_profile, dict):
        raise ValueError(f"search_profile.json must be a JSON object: {SEARCH_PROFILE}")

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
            # Title-only adjustment: do NOT destroy semantic similarity by rescoring with empty profile.
            result = _title_only_adjust(item, search_profile)
            if result is None:
                skipped_no_profile.append(job_id)
                continue

            apply_scoring_to_item(item, result)
            updated.append((job_id, str(result.get("selected_resume_id")), float(result.get("hybrid_score", 0.0))))
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
