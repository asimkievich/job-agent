import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from config import load_secrets, require_env
from openai import OpenAI

import queue_utils


WORK_ITEM = Path("work_items/job_001.json")
RESUME_PROFILE = Path("profiles/resumes/resume_product_owner_ai_pm.json")
OUT_PATH = Path("work_items/job_001.application_draft.json")


GERMAN_MARKERS = [
    "beschreibung", "aufgaben", "anforderungen", "bewerben", "freiberuflich",
    "projekt", "deutsch", "kenntnisse", "erfahrung", "ansprechpartner",
    "willkommen", "veröffentlicht", "mehrsprachigkeit"
]
ENGLISH_MARKERS = [
    "responsibilities", "requirements", "apply", "contract", "remote",
    "experience", "skills", "english", "we are looking for", "role", "must have"
]


def detect_language(text: str) -> str:
    t = (text or "").lower()

    de = sum(1 for w in GERMAN_MARKERS if w in t)
    en = sum(1 for w in ENGLISH_MARKERS if w in t)

    # Extra heuristic: presence of German umlauts/ß
    if re.search(r"[äöüß]", t):
        de += 2

    if de >= en + 2:
        return "de"
    if en >= de + 2:
        return "en"

    # neither strong: default to German (your preference for DE market)
    return "de"


def _safe_filename(s: str, max_len: int = 140) -> str:
    s = (s or "").strip()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return (s[:max_len] if len(s) > max_len else s) or "draft"


def _choose_draft_dir(mode: str) -> Path:
    # Keep prod and injected drafts separated
    d = Path("drafts_injected") if mode == "injected" else Path("drafts")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_resume_for_item(item: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer resume path provided by scoring, fall back to default RESUME_PROFILE.
    scoring = item.get("scoring") or {}
    resume_path = scoring.get("resume_path")

    if resume_path:
        p = Path(resume_path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))

    # fallback
    return json.loads(RESUME_PROFILE.read_text(encoding="utf-8"))


def _openai_client() -> OpenAI:
    load_secrets()
    api_key = require_env("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def _build_prompt(job_url: str, job_title: str, job_text: str, resume: Dict[str, Any], forced_lang: str) -> Dict[str, str]:
    system = (
        "You are an expert product/consulting career assistant. "
        "You write concise, credible application text. "
        "Never invent experience. If something is unknown, omit it."
    )

    user = f"""
JOB URL: {job_url}
JOB TITLE: {job_title}

JOB TEXT:
{job_text}

CANDIDATE RESUME PROFILE (JSON):
{json.dumps(resume, ensure_ascii=False, indent=2)}

LANGUAGE RULE:
- Write the output in this language ONLY: {forced_lang}
- If {forced_lang} == "de": write natural, professional German.
- If {forced_lang} == "en": write natural, professional English.

TASK:
Generate application content.
Output STRICT JSON with this shape:
{{
  "language": "{forced_lang}",
  "tailored_summary": "2-4 lines, first person, no hype",
  "cover_note": "6-10 lines, first person, professional, specific to the job, no fluff",
  "fit_bullets": ["bullet1", "bullet2", "bullet3"],
  "risks_or_gaps": ["any real gaps to be aware of (optional)"]
}}

Constraints:
- Do NOT claim hands-on experience with tools/tech unless present in the resume JSON.
- If this job is engineering-heavy, state that clearly in risks_or_gaps.
- Keep tone direct and credible.
"""
    return {"system": system, "user": user}


def _call_llm_for_draft(client: OpenAI, system: str, user: str) -> Dict[str, Any]:
    model = (Path("profiles") / "openai_model.txt")
    chosen_model = model.read_text(encoding="utf-8").strip() if model.exists() else "gpt-4.1-mini"

    resp = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        # Keep raw output for debugging if model didn't return strict JSON
        return {"raw": content}


def _should_generate_draft(item: Dict[str, Any]) -> bool:
    # Conservative: only draft if decision says pursue OR human status indicates active.
    decision = (item.get("decision") or (item.get("scoring") or {}).get("decision") or "").lower()
    status = (item.get("status") or "").lower()

    if decision == "pursue":
        return True
    if status in {"pending", "approved"}:
        return True
    return False


def maybe_prepare_application_draft(item: Dict[str, Any]) -> Optional[str]:
    """
    Called by batch_run.py.

    - Works for injected mode and prod mode.
    - Writes a draft JSON file.
    - Updates queue item draft_path.
    - Returns the draft_path (str) or None if skipped.
    """
    if not _should_generate_draft(item):
        return None

    mode = (item.get("mode") or "prod").lower()
    job_url = item.get("href") or item.get("url") or ""
    job_title = item.get("title") or ""
    profile = item.get("profile") or {}

    # Use extracted text if available, otherwise fallback to any raw body.
    job_text = profile.get("body_text") or profile.get("description") or profile.get("tasks") or ""

    forced_lang = detect_language(job_text)
    resume = _load_resume_for_item(item)

    # Create output path
    run_id = item.get("run_id") or "run"
    job_key = item.get("normalized_href") or job_url or job_title
    fname = f"{run_id}__{_safe_filename(job_key)}.application_draft.json"
    out_dir = _choose_draft_dir(mode)
    out_path = out_dir / fname

    client = _openai_client()
    prompt = _build_prompt(job_url=job_url, job_title=job_title, job_text=job_text, resume=resume, forced_lang=forced_lang)
    data = _call_llm_for_draft(client, system=prompt["system"], user=prompt["user"])

    # Ensure minimal metadata is always present
    data.setdefault("language", forced_lang)
    data.setdefault("job_url", job_url)
    data.setdefault("job_title", job_title)
    data.setdefault("generated_at", __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"))
    data.setdefault("mode", mode)

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update queue with draft_path so UI/pipeline can find it
    try:
        queue_path = Path("queue/queue_injected.json") if mode == "injected" else Path("queue/queue.json")
        q = queue_utils.load_queue(path=queue_path)
        item["draft_path"] = str(out_path)
        queue_utils.enqueue_item(q, item)
        queue_utils.save_queue(q, path=queue_path)
    except Exception:
        # Don't fail the batch if queue update fails
        pass

    return str(out_path)


def main() -> None:
    """
    CLI mode for your original one-off testing:
    Reads work_items/job_001.json and writes work_items/job_001.application_draft.json
    """
    client = _openai_client()

    item = json.loads(WORK_ITEM.read_text(encoding="utf-8"))
    resume = json.loads(RESUME_PROFILE.read_text(encoding="utf-8"))

    job_title = item["job"]["title"]
    job_url = item["job"]["url"]
    job_text = item["job"]["body_text"]

    forced_lang = detect_language(job_text)
    prompt = _build_prompt(job_url=job_url, job_title=job_title, job_text=job_text, resume=resume, forced_lang=forced_lang)
    data = _call_llm_for_draft(client, system=prompt["system"], user=prompt["user"])

    data.setdefault("language", forced_lang)
    data.setdefault("job_url", job_url)
    data.setdefault("job_title", job_title)
    data.setdefault("generated_at", __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"))
    data.setdefault("mode", "cli")

    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
