# ui_replay.py
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer
from openai import OpenAI

import score_job
from config import load_secrets, require_env, build_application_draft_prompt



WORK_ITEMS_DIR = Path("work_items")
RESUMES_DIR = Path("profiles/resumes")
APPLICATIONS_DIR = Path("applications")


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _discover_resume_paths() -> List[Path]:
    if not RESUMES_DIR.exists():
        return []
    return sorted(
        p for p in RESUMES_DIR.glob("*.json")
        if p.is_file() and "resume" in p.name.lower()
    )


def _load_job_profile(job_id: str, artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # 1) artifacts.profile_path (if present)
    if artifacts and isinstance(artifacts, dict):
        pp = artifacts.get("profile_path")
        if isinstance(pp, str) and pp.strip():
            p = Path(pp)
            if not p.exists():
                p = Path(pp.replace("\\", "/"))
            if p.exists():
                obj = _read_json(p)
                if isinstance(obj, dict):
                    return obj

    # 2) work_items/<job_id>.profile.json
    p2 = WORK_ITEMS_DIR / f"{job_id}.profile.json"
    if not p2.exists():
        raise FileNotFoundError(f"Missing job profile: {p2}")
    obj = _read_json(p2)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid JSON in {p2}")
    return obj


def rescore_queue_item(
    queue_doc: Dict[str, Any],
    job_id: str,
    model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    items = queue_doc.setdefault("items_by_job_id", {})
    raw = items.get(job_id) or {}
    artifacts = raw.get("artifacts") if isinstance(raw.get("artifacts"), dict) else {}

    job_profile = _load_job_profile(job_id, artifacts=artifacts)
    resume_paths = _discover_resume_paths()

    scoring = score_job.score_job(job_profile, resume_paths, model=model)

    # Apply scoring fields (UI expects these)
    raw["decision"] = scoring.get("decision", raw.get("decision", "skip"))
    raw["selected_resume_id"] = scoring.get("selected_resume_id", raw.get("selected_resume_id", ""))
    raw["top_resume_scores"] = scoring.get("top_resume_scores", []) or []
    raw["reasons"] = scoring.get("reasons", []) or []
    raw["scores"] = {
        "hybrid_score": scoring.get("hybrid_score", 0.0),
        "semantic_similarity": scoring.get("semantic_similarity", 0.0),
        "reward": scoring.get("reward", 0.0),
        "penalty": scoring.get("penalty", 0.0),
        "rule_score": scoring.get("rule_score", 0.0),
        "domain_distance": scoring.get("domain_distance", 0),
    }
    raw["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    items[job_id] = raw
    return scoring


def generate_and_attach_draft(
    queue_doc: Dict[str, Any],
    job_id: str,
    *,
    force: bool = False,
) -> Optional[Path]:
    items = queue_doc.setdefault("items_by_job_id", {})
    raw = items.get(job_id) or {}
    decision = (raw.get("decision") or "").strip().lower()

    if (decision not in {"review", "pursue"}) and not force:
        return None

    artifacts = raw.get("artifacts") if isinstance(raw.get("artifacts"), dict) else {}
    job_profile = _load_job_profile(job_id, artifacts=artifacts)

    resume_id = (raw.get("selected_resume_id") or "").strip()
    if not resume_id:
        return None

    resume_path = RESUMES_DIR / f"{resume_id}.json"
    if not resume_path.exists():
        return None
    resume_json = _read_json(resume_path)

    body_text = (job_profile.get("body_text") or "")
    created_at = datetime.now(timezone.utc).isoformat()
    
    prompt = build_application_draft_prompt(
    job_id=job_id,
    href=str(raw.get("url", "") or ""),
    title=str(raw.get("title", "") or ""),
    published_at=str(raw.get("published_at", "") or ""),
    selected_resume_id=resume_id,
    job_profile=job_profile,
    body_text=body_text,
    resume_json=resume_json,
    created_at=created_at,
)
    load_secrets()
    require_env("OPENAI_API_KEY")

    client = OpenAI()
    model_name = os.environ.get("OPENAI_DRAFT_MODEL", "gpt-4.1-mini")

    resp = client.responses.create(
        model=model_name,
        input=prompt,
    )

    text = resp.output_text
    draft_obj = json.loads(text)

    out_dir = APPLICATIONS_DIR / job_id
    out_path = out_dir / "draft.json"
    _write_json(out_path, draft_obj)


    return out_path