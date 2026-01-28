import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


IN_PATH = Path("work_items/job_001.json")
OUT_PATH = Path("work_items/job_001.profile.json")


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def find_first(patterns: list[str], text: str) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return clean(m.group(1))
    return None


def extract_section(text: str, start_label: str, end_labels: list[str]) -> Optional[str]:
    # crude but works for these page dumps
    start = re.search(rf"{re.escape(start_label)}\s*", text, flags=re.IGNORECASE)
    if not start:
        return None
    rest = text[start.end():]
    end_positions = []
    for lab in end_labels:
        m = re.search(rf"\n{re.escape(lab)}\b", rest, flags=re.IGNORECASE)
        if m:
            end_positions.append(m.start())
    if end_positions:
        rest = rest[: min(end_positions)]
    return rest.strip() or None


def extract_keywords(text: str) -> list[str]:
    # On freelancermap pages, tags often appear as single words in a block.
    # We'll grab a limited set of plausible tags from the area above "Beschreibung".
    upper = text.split("Beschreibung")[0] if "Beschreibung" in text else text[:2000]
    raw = re.findall(r"\b[A-Za-zÄÖÜäöü0-9][A-Za-zÄÖÜäöü0-9+.\-]{1,29}\b", upper)

    blacklist = {
        "Dashboard", "Projekte", "finden", "veröffentlicht", "Deutschland", "Remote",
        "Freiberuflich", "ab", "sofort", "Bewerben", "Merken", "Eingestellt", "von",
        "Ansprechpartner", "Projekt", "ID", "Top", "Endkundenprojekt"
    }

    kws = []
    for w in raw:
        if w in blacklist:
            continue
        if w.isdigit():
            continue

        wl = w.lower()
        tech_allow = {
            "docker", "nginx", "redis", "postgresql", "strapi", "astro",
            "medusa", "mollie", "hetzner", "cicd", "ci", "cd"
        }

        if wl in tech_allow:
            kws.append(w)
        elif any(ch.isdigit() for ch in w):
            kws.append(w)
        elif w[0].isupper() and len(w) >= 3:
            # Allow some tags like "Nachhaltigkeit"
            kws.append(w)

    out = []
    seen = set()
    for k in kws:
        kk = k.lower()
        if kk in seen:
            continue
        seen.add(kk)
        out.append(k)

    return out[:30]


def make_fallback_job_id(url: str) -> str:
    # stable fallback if Projekt-ID cannot be found
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"\W+", "_", slug).strip("_")
    return slug or "unknown_job"


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}")

    item = json.loads(IN_PATH.read_text(encoding="utf-8"))
    text = item["job"]["body_text"]

    title = item["job"]["title"]
    url = item["job"]["url"]

    published = find_first([r"veröffentlicht am\s+([0-9]{2}\.[0-9]{2}\.[0-9]{4}.*?Uhr)"], text)
    location = find_first([r"\n([A-Za-zÄÖÜäöüß\- ]+)\n,\s*\nDeutschland"], text)
    remote = "remote" if re.search(r"\bRemote\b", text, re.IGNORECASE) else None
    contract = "freelance" if re.search(r"\bFreiberuflich\b", text, re.IGNORECASE) else None

    # Often "ab sofort" appears on its own line
    start = find_first([r"\bFreiberuflich\b\s*\n([^\n]+)\n"], text)

    company = find_first([r"Eingestellt von\s*\n([^\n]+)"], text)
    project_id = find_first([r"Projekt-ID\s*\n([0-9]+)"], text)

    job_id = project_id or make_fallback_job_id(url)

    description = extract_section(
        text,
        "Beschreibung",
        end_labels=["Bewerben", "Ähnliche Projekte", "Für Freelancer", "Datenschutz"]
    )
    tasks = extract_section(
        text,
        "Aufgaben:",
        end_labels=["Wir suchen", "Bewerben", "Ähnliche Projekte", "Für Freelancer", "Datenschutz"]
    )

    # classify role family (very rough heuristic; improve later)
    role_family = "engineering" if re.search(
        r"\bDocker\b|\bCI/CD\b|\bMigration\b|\bFrontend\b|\bBackend\b",
        text,
        re.IGNORECASE
    ) else "product"

    profile: Dict[str, Any] = {
        "job_id": job_id,
        "source": "freelancermap",
        "url": url,
        "title": clean(title.replace(" auf www.freelancermap.de", "")),
        "company": clean(company) if company else None,
        "published_at": published,
        "location": clean(location) if location else None,
        "work_mode": remote,
        "contract_type": contract,
        "start_hint": clean(start) if start else None,
        "role_family_guess": role_family,
        "keywords": extract_keywords(text),
        "description": clean(description) if description else None,
        "tasks": clean(tasks) if tasks else None
    }

    OUT_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
