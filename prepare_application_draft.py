import json
import re
from pathlib import Path

from config import load_secrets, require_env
from openai import OpenAI


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
    t = text.lower()

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


def main() -> None:
    load_secrets()
    api_key = require_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    item = json.loads(WORK_ITEM.read_text(encoding="utf-8"))
    resume = json.loads(RESUME_PROFILE.read_text(encoding="utf-8"))

    job_title = item["job"]["title"]
    job_url = item["job"]["url"]
    job_text = item["job"]["body_text"]

    forced_lang = detect_language(job_text)

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
- Do NOT claim hands-on experience with Strapi/Astro/Medusa/Mollie unless present in the resume JSON.
- If this job is engineering-heavy, state that clearly in risks_or_gaps.
- Keep tone direct and credible.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {"raw": content, "forced_language": forced_lang}

    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
