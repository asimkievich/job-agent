import json
from pathlib import Path
from typing import Dict, Any

from sentence_transformers import SentenceTransformer, util


JOB_PROFILE = Path("work_items/job_001.profile.json")
RESUME = Path("profiles/resumes/resume_product_owner_ai_pm.json")
SEARCH = Path("profiles/resumes/search_profile.json")


def load(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p.resolve()}")
    return json.loads(p.read_text(encoding="utf-8"))


def build_resume_text(resume: Dict[str, Any]) -> str:
    cand = resume.get("candidate", {})
    roles = resume.get("roles", {})
    skills = resume.get("skills", {})
    domains = resume.get("domains", [])
    industries = resume.get("industries", [])

    parts = [
        cand.get("headline", ""),
        cand.get("summary", ""),
        "Primary roles: " + ", ".join(roles.get("primary", []) or []),
        "Secondary roles: " + ", ".join(roles.get("secondary", []) or []),
        "Domains: " + ", ".join(domains or []),
        "Industries: " + ", ".join(industries or []),
        "Hard skills: " + ", ".join((skills.get("hard", []) or [])),
        "Tools: " + ", ".join((skills.get("tools", []) or [])),
        "Methods: " + ", ".join((skills.get("methods", []) or [])),
    ]
    return "\n".join([p for p in parts if p.strip()])


def build_job_text(job: Dict[str, Any]) -> str:
    parts = [
        job.get("title", ""),
        job.get("description", "") or "",
        job.get("tasks", "") or "",
        "Keywords: " + ", ".join(job.get("keywords") or []),
        f"Contract: {job.get('contract_type')}, Work mode: {job.get('work_mode')}, Location: {job.get('location')}",
    ]
    return "\n".join([p for p in parts if p.strip()])


def rule_score(job: Dict[str, Any], search: Dict[str, Any]) -> Dict[str, Any]:
    reasons = []
    score = 0.0

    title = (job.get("title") or "").lower()
    description = (job.get("description") or "").lower()

    # Hard gates (quick rejects)
    hard_reject_terms = ["werkstudent", "junior", "call center", "help desk", "service desk"]
    if any(t in title or t in description for t in hard_reject_terms):
        return {
            "rule_score": 0.0,
            "domain_distance": 3,
            "rule_decision": "skip",
            "reasons": ["Hard reject: clearly not target seniority/role type."]
        }

    # Penalties
    for bad_kw in search["anti_patterns"]["keywords_to_penalize"]:
        if bad_kw.lower() in description or bad_kw.lower() in title:
            score -= 0.2
            reasons.append(f"Penalized keyword: {bad_kw}")

    # Role title match (simple)
    target_roles = [r.lower() for r in search["target_roles"]["highest_priority"]]
    if any(r in title for r in target_roles):
        score += 0.35
        reasons.append("Role title matches target product roles.")
    else:
        score += 0.10
        reasons.append("Role title not explicitly PM/PO/HoP; relying on domain + signals.")

    # Domain fit (simple heuristic)
    preferred_domains = search["domain_fit"]["preferred_domains"]
    domain_hits = sum(1 for d in preferred_domains if d.split("/")[0].lower() in description)

    if domain_hits >= 1:
        score += 0.30
        domain_distance = 0
        reasons.append("Strong domain adjacency (preferred domain signals found).")
    else:
        score += 0.10
        domain_distance = 2
        reasons.append("Weak/unclear domain adjacency (no preferred domain signals found).")

    # AI bonus (optional)
    if any(x in description for x in ["genai", "llm", "künstliche intelligenz", "ai "]):
        score += 0.10
        reasons.append("AI-related signals detected (bonus).")

    # Remote preference
    if (job.get("work_mode") or "").lower() == "remote":
        score += 0.05
        reasons.append("Remote-friendly role.")

    rule_decision = "pursue" if score >= 0.45 and domain_distance < 3 else "skip"
    return {
        "rule_score": round(score, 2),
        "domain_distance": domain_distance,
        "rule_decision": rule_decision,
        "reasons": reasons
    }


def main():
    job = load(JOB_PROFILE)
    resume = load(RESUME)
    search = load(SEARCH)

    # 1) Rule score
    rs = rule_score(job, search)

    # 2) Semantic similarity (Sentence-BERT embeddings + cosine similarity)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    resume_text = build_resume_text(resume)
    job_text = build_job_text(job)

    emb_resume = model.encode(resume_text, convert_to_tensor=True, normalize_embeddings=True)
    emb_job = model.encode(job_text, convert_to_tensor=True, normalize_embeddings=True)
    sim = float(util.cos_sim(emb_resume, emb_job).item())  # -1..1, usually 0..1

    # 3) Hybrid score: combine rule score (0..~1) with similarity (0..1)
    # rule_score here is roughly 0..0.8; normalize it conservatively by dividing by 0.8
    rule_norm = max(0.0, min(1.0, (rs["rule_score"] / 0.8)))
    hybrid = 0.6 * rule_norm + 0.4 * max(0.0, min(1.0, sim))

    decision = "pursue" if hybrid >= 0.55 and rs["domain_distance"] < 3 else "skip"

    out = {
        "rule_score": rs["rule_score"],
        "semantic_similarity": round(sim, 3),
        "hybrid_score": round(hybrid, 3),
        "domain_distance": rs["domain_distance"],
        "decision": decision,
        "reasons": rs["reasons"]
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
