import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from sentence_transformers import SentenceTransformer, util


JOB_PROFILE = Path("work_items/job_001.profile.json")
RESUME = Path("profiles/resumes/resume_product_owner_ai_pm.json")
SEARCH = Path("profiles/resumes/search_profile.json")


# -----------------------
# Generic helpers
# -----------------------
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

    # --- Add experience/positions (your schema uses "experience") ---
    exp = resume.get("experience", []) or resume.get("positions", []) or []
    if isinstance(exp, list) and exp:
        exp_lines: List[str] = []
        MAX_ROLES = 6
        MAX_CHARS_PER_ROLE = 800

        for item in exp[:MAX_ROLES]:
            if not isinstance(item, dict):
                continue

            role = (item.get("role") or item.get("title") or "").strip()
            company = (item.get("company") or "").strip()
            header = " | ".join([x for x in [role, company] if x]).strip()

            # Many of your entries use "highlights": [ ... ]
            highlights = item.get("highlights") or item.get("bullets") or item.get("achievements") or []
            text_bits: List[str] = []

            # Optional free-text fields if present
            for k in ("summary", "description", "details"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    text_bits.append(v.strip())

            if isinstance(highlights, list) and highlights:
                hl = " ".join([str(h).strip() for h in highlights if str(h).strip()])
                if hl:
                    text_bits.append(hl)

            text = " ".join(text_bits).strip()
            if text:
                text = text[:MAX_CHARS_PER_ROLE].strip()

            if header and text:
                exp_lines.append(f"- {header}: {text}")
            elif header:
                exp_lines.append(f"- {header}")
            elif text:
                exp_lines.append(f"- {text}")

        if exp_lines:
            parts.append("Experience:\n" + "\n".join(exp_lines))

    return "\n".join([p for p in parts if p.strip()])



def build_job_text(job: Dict[str, Any]) -> str:
    raw = "\n".join([
        job.get("title", "") or "",
        job.get("description", "") or "",
        job.get("tasks", "") or "",
        job.get("requirements", "") or "",
        "Keywords: " + ", ".join(job.get("keywords") or []),
        f"Contract: {job.get('contract_type')}, Work mode: {job.get('work_mode')}, Location: {job.get('location')}",
    ])
    return _clean_job_text_for_embedding(raw)

def _clean_job_text_for_embedding(text: str) -> str:
    """
    Conservative, deterministic noise removal for scraped job text.
    - Drops typical UI/navigation/footer/cookie lines
    - Deduplicates repeated lines
    - Limits length to avoid boilerplate dominating embeddings
    """
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines()]
    drop_patterns = [
        r"^\s*(anmelden|registrieren|login|passwort|konto)\s*$",
        r"^\s*(projekte finden|jobs|freelancer|profil)\s*$",
        r"^\s*(impressum|datenschutz|privacy|cookies?)\s*$",
        r"^\s*(kontakt|agb|hilfe)\s*$",
        r"freelancermap",
        r"cookie",
        r"datenschutz",
        r"impressum",
    ]

    out = []
    seen = set()
    for ln in lines:
        if not ln:
            continue
        lnl = ln.lower()
        if any(re.search(p, lnl) for p in drop_patterns):
            continue
        # de-dupe exact repeated lines
        if lnl in seen:
            continue
        seen.add(lnl)
        out.append(ln)

    cleaned = "\n".join(out).strip()

    # hard cap to reduce boilerplate domination
    return cleaned[:8000]


# -----------------------
# Simple scoring v2: penalties (config-driven)
# -----------------------

# Tiny in-memory cache to avoid re-embedding the same strings repeatedly.
# Keyed by (id(model), text).

_EMBED_CACHE: Dict[tuple, Any] = {}


def _sanitize_title_for_embedding(title: str) -> str:
    """Keep title embedding stable by removing known boilerplate suffixes."""
    t = (title or "").strip()
    # batch_run path doesn't strip this suffix, so do it here defensively
    t = re.sub(r"\s+auf\s+www\.freelancermap\.de\s*$", "", t, flags=re.IGNORECASE).strip()
    return t

def _is_product_title(title: str) -> bool:
    t = (title or "").lower()

    # Strong signals (keep it strict so we don't accidentally whitelist non-product titles)
    patterns = [
        r"\bproduct\s*(owner|manager|management|lead|leader|head)\b",
        r"\bprodukt\s*(owner|manager|management|lead|leitung)\b",
        r"\bpo\b",  # optional; only keep if you trust PO in titles
        r"\bpm\b",  # optional; risky (can be project manager). I'd usually NOT include.
    ]
    return any(re.search(p, t) for p in patterns)

def _has_exact_term(text_lc: str, term_lc: str) -> bool:
    """
    True if term_lc occurs in text_lc as a whole term (not inside a bigger word).
    Works for single words and multi-word phrases.
    """
    if not term_lc:
        return False
    return re.search(rf"(?<!\w){re.escape(term_lc)}(?!\w)", text_lc) is not None

def _embed_text(model: SentenceTransformer, text: str):
    """Embed text with caching; returns a tensor suitable for util.cos_sim."""
    key = (id(model), text)
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    _EMBED_CACHE[key] = emb
    return emb


def compute_penalty_v2(
    job_profile: Dict[str, Any],
    search_profile: Dict[str, Any],
    model: SentenceTransformer,
) -> Dict[str, Any]:
    """
    Compute a bounded penalty (0..max_penalty) using search_profile['simple_scoring_v2'].

    Penalty categories (each applied at most once, total capped):
      - junior_markers (keyword)
      - support_admin_markers (keyword)
      - non_pm_title_intent (embedding-based title intent margin)

    Returns:
      {
        "penalty": float,
        "reasons": List[str],
        "debug": { ... optional sims/margins ... }
      }
    """
    cfg = (search_profile or {}).get("simple_scoring_v2", {}) or {}
    max_penalty = float(cfg.get("max_penalty", 0.10))
    penalties_cfg = cfg.get("penalties", {}) or {}

    title = _sanitize_title_for_embedding((job_profile or {}).get("title", "") or "")
    description = ((job_profile or {}).get("description", "") or "")
    tasks = ((job_profile or {}).get("tasks", "") or "")
    requirements = ((job_profile or {}).get("requirements", "") or "")
    full_text = " ".join([title, description, tasks, requirements]).lower()
    title_is_product_role = _is_product_title(title)


    reasons: List[str] = []
    debug: Dict[str, Any] = {}
    penalty_total = 0.0

    def add_pen(amount: float, reason: str) -> None:
        nonlocal penalty_total
        if amount <= 0:
            return
        remaining = max_penalty - penalty_total
        if remaining <= 0:
            return
        applied = amount if amount <= remaining else remaining
        penalty_total += applied
        reasons.append(f"{reason} (−{applied:.3f})")

    # --- P0: non-product title (soft penalty) ---

    rewards_cfg = cfg.get("rewards", {}) or {}
    ai_cfg = rewards_cfg.get("ai_ml_markers", {}) or {}
    ai_markers = ai_cfg.get("markers", []) or []
    ai_signal = False
    for m in ai_markers:
        if isinstance(m, str) and m.strip():
            ml = m.lower().strip()
            if _has_exact_term(full_text, ml):
                ai_signal = True
                break

    nonprod_cfg = penalties_cfg.get("non_product_title", {}) or {}
    nonprod_pen = float(nonprod_cfg.get("penalty", 0.00))
    if (not title_is_product_role) and (not ai_signal) and nonprod_pen > 0:
        add_pen(nonprod_pen, "Penalty: title not a product role")

    # --- P1: junior markers (strong penalty) ---
    junior_cfg = penalties_cfg.get("junior_markers", {}) or {}
    junior_pen = float(junior_cfg.get("penalty", 0.10))
    junior_markers = junior_cfg.get("markers", []) or []
    if any(isinstance(m, str) and m.strip() and _has_exact_term(full_text, m.lower().strip()) for m in junior_markers):
        add_pen(junior_pen, "Penalty: junior/intern/werkstudent marker")

    # --- P3: support/admin markers (soft penalty) ---
    supp_cfg = penalties_cfg.get("support_admin_markers", {}) or {}
    supp_pen = float(supp_cfg.get("penalty", 0.04))
    supp_markers = supp_cfg.get("markers", []) or []
    if (not title_is_product_role) and any(
        isinstance(m, str) and m.strip() and _has_exact_term(full_text, m.lower().strip())
        for m in supp_markers
    ):

            add_pen(supp_pen, "Penalty: support/admin/ops marker")


    # --- P2: non-PM title intent (embedding-based) ---
    intent_cfg = penalties_cfg.get("non_pm_title_intent", {}) or {}
    intent_pen = float(intent_cfg.get("penalty", 0.06))
    min_margin = float(intent_cfg.get("min_margin", 0.10))
    pm_protos = intent_cfg.get("pm_prototypes", []) or []
    nonpm_protos = intent_cfg.get("non_pm_prototypes", []) or []

    # Only compute if we have a usable title and prototypes and some remaining budget.
    if title and pm_protos and nonpm_protos and (max_penalty - penalty_total) > 0:
        title_emb = _embed_text(model, title)

        pm_sims = []
        for p in pm_protos:
            if p and isinstance(p, str):
                pm_sims.append(float(util.cos_sim(title_emb, _embed_text(model, p))[0][0]))
        nonpm_sims = []
        for p in nonpm_protos:
            if p and isinstance(p, str):
                nonpm_sims.append(float(util.cos_sim(title_emb, _embed_text(model, p))[0][0]))

        pm_sim = max(pm_sims) if pm_sims else -1.0
        nonpm_sim = max(nonpm_sims) if nonpm_sims else -1.0
        margin = nonpm_sim - pm_sim

        debug.update({"pm_sim": pm_sim, "nonpm_sim": nonpm_sim, "margin": margin, "title_used": title})

        if margin >= min_margin:
            add_pen(intent_pen, f"Penalty: title intent looks non-PM (margin={margin:.3f})")

    penalty_total = min(max(penalty_total, 0.0), max_penalty)

    return {
        "penalty": penalty_total,
        "reasons": reasons,
        "debug": debug,
    }


def compute_reward_v2(
    job_profile: Dict[str, Any],
    search_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute a bounded reward (0..max_reward) using search_profile['simple_scoring_v2'].

    Reward categories (each applied at most once, total capped):
      - pm_title_markers  (title-based)
      - ai_ml_markers     (full-text)

    Returns:
      {
        "reward": float,
        "reasons": List[str],
        "debug": { ... matched markers ... }
      }
    """
    cfg = (search_profile or {}).get("simple_scoring_v2", {}) or {}
    max_reward = float(cfg.get("max_reward", 0.10))
    rewards_cfg = cfg.get("rewards", {}) or {}

    title = _sanitize_title_for_embedding((job_profile or {}).get("title", "") or "")
    description = ((job_profile or {}).get("description", "") or "")
    tasks = ((job_profile or {}).get("tasks", "") or "")
    requirements = ((job_profile or {}).get("requirements", "") or "")
    full_text = " ".join([title, description, tasks, requirements]).lower()

    reasons: List[str] = []
    debug: Dict[str, Any] = {"matched": {}}
    reward_total = 0.0

    def add_rew(amount: float, reason: str, key: str, matched_value: Optional[str] = None) -> None:
        nonlocal reward_total
        if amount <= 0:
            return
        remaining = max_reward - reward_total
        if remaining <= 0:
            return
        applied = amount if amount <= remaining else remaining
        reward_total += applied
        reasons.append(f"{reason} (+{applied:.3f})")
        if matched_value:
            debug["matched"][key] = matched_value

    # --- R1: PM title markers ---
    pm_cfg = rewards_cfg.get("pm_title_markers", {}) or {}
    pm_bonus = float(pm_cfg.get("bonus", 0.05))
    pm_markers = pm_cfg.get("markers", []) or []

    title_lc = title.lower()
    pm_hit = None
    for m in pm_markers:
        if isinstance(m, str) and m.strip() and _has_exact_term(title_lc, m.lower().strip()):
            pm_hit = m
            break
    if pm_hit:
        add_rew(pm_bonus, "Reward: PM marker in title", "pm_title_markers", pm_hit)

    # --- R2: AI/ML markers in full text ---
    ai_cfg = rewards_cfg.get("ai_ml_markers", {}) or {}
    ai_bonus = float(ai_cfg.get("bonus", 0.05))
    ai_markers = ai_cfg.get("markers", []) or []

    ai_hit = None
    for m in ai_markers:
        if isinstance(m, str) and m.strip():
            ml = m.lower().strip()
            if _has_exact_term(full_text, ml):
                ai_hit = m
                break

    if ai_hit:
        add_rew(ai_bonus, "Reward: AI/ML signal", "ai_ml_markers", ai_hit)

    reward_total = min(max(reward_total, 0.0), max_reward)

    return {
        "reward": reward_total,
        "reasons": reasons,
        "debug": debug,
    }

def score_job(
    job_profile: Dict[str, Any],
    resume_paths: List[Path],
    model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    """
    Simple Scoring v2 (backward compatible output):

      sem  = best resume cosine similarity (0..1)
      rew  = compute_reward_v2(...)  (0..max_reward)
      pen  = compute_penalty_v2(..., model) (0..max_penalty)

      raw    = sem + rew - pen              ∈ [-max_penalty, 1+max_reward]
      hybrid = (raw + max_penalty) / (1 + max_reward + max_penalty)  ∈ [0,1]

    Decision via thresholds in search_profile.json:
      pursue if hybrid >= thresholds.pursue
      review if hybrid >= thresholds.review
      else   skip

    Backward compatibility:
      - rule_score      := reward (not penalty)
      - domain_distance := 0 (deprecated placeholder)
      - reasons         := reward reasons + penalty reasons
    """
    search = load(SEARCH)

    cfg = (search or {}).get("simple_scoring_v2", {}) or {}
    max_reward = float(cfg.get("max_reward", 0.10))
    max_penalty = float(cfg.get("max_penalty", 0.10))

    thresholds = cfg.get("thresholds", {}) or {}
    t_review = float(thresholds.get("review", 0.63))
    t_pursue = float(thresholds.get("pursue", 0.75))

    if model is None:
        # Match your new multilingual setup (batch_run passes a model anyway).
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Compute reward/penalty ONCE per job
    rew_obj = compute_reward_v2(job_profile, search)
    pen_obj = compute_penalty_v2(job_profile, search, model)

    reward = float(rew_obj.get("reward", 0.0))
    penalty = float(pen_obj.get("penalty", 0.0))

    reward_reasons = rew_obj.get("reasons", []) or []
    penalty_reasons = pen_obj.get("reasons", []) or []
    reasons = reward_reasons + penalty_reasons

    # Prepare embeddings
    job_text = build_job_text(job_profile)
    
    emb_job = model.encode(job_text, convert_to_tensor=True, normalize_embeddings=True)

    if not resume_paths:
        resume_paths = [RESUME]

    denom = 1.0 + max_reward + max_penalty

    top_resume_scores: List[Dict[str, Any]] = []
    for rp in resume_paths:
        try:
            resume = load(rp)
        except Exception:
            continue

        resume_id = rp.stem
        resume_text = build_resume_text(resume)

        emb_resume = model.encode(resume_text, convert_to_tensor=True, normalize_embeddings=True)
        sem = float(util.cos_sim(emb_resume, emb_job).item())

        raw = sem + reward - penalty
        hybrid = (raw + max_penalty) / denom

        top_resume_scores.append({
            "resume_id": resume_id,

            # Fields UI already expects / displays
            "hybrid_score": round(hybrid, 3),
            "semantic_similarity": round(sem, 3),
            "rule_score": round(reward, 3),     # repurposed
            "domain_distance": 0,               # deprecated

            # Extra v2 fields (safe additions)
            "reward": round(reward, 3),
            "penalty": round(penalty, 3),
            "raw_score": round(raw, 3),
        })

    top_resume_scores.sort(key=lambda d: d.get("hybrid_score", 0.0), reverse=True)

    if not top_resume_scores:
        return {
            "decision": "skip",
            "selected_resume_id": "default",
            "top_resume_scores": [],
            "hybrid_score": 0.0,
            "semantic_similarity": 0.0,
            "rule_score": round(reward, 3),
            "domain_distance": 0,
            "reasons": reasons,
            "reward": round(reward, 3),
            "penalty": round(penalty, 3),
            "scoring_version": "simple_scoring_v2",
            "debug": {
                "reward_debug": rew_obj.get("debug", {}),
                "penalty_debug": pen_obj.get("debug", {}),
                "thresholds": {"review": t_review, "pursue": t_pursue},
                "normalization": {"raw_min": -max_penalty, "raw_max": 1.0 + max_reward, "denom": denom},
            },
        }

    best = top_resume_scores[0]
    best_h = float(best["hybrid_score"])

    if best_h >= t_pursue:
        decision = "pursue"
    elif best_h >= t_review:
        decision = "review"
    else:
        decision = "skip"

    return {
        # Backward-compatible top-level keys
        "decision": decision,
        "selected_resume_id": best["resume_id"],
        "top_resume_scores": top_resume_scores,

        "hybrid_score": best["hybrid_score"],
        "semantic_similarity": best["semantic_similarity"],
        "rule_score": best["rule_score"],
        "domain_distance": 0,
        "reasons": reasons,

        # Helpful v2 fields (safe additions)
        "reward": round(reward, 3),
        "penalty": round(penalty, 3),
        "scoring_version": "simple_scoring_v2",
        "debug": {
            "reward_reasons": reward_reasons,
            "penalty_reasons": penalty_reasons,
            "reward_debug": rew_obj.get("debug", {}),
            "penalty_debug": pen_obj.get("debug", {}),
            "thresholds": {"review": t_review, "pursue": t_pursue},
            "normalization": {"raw_min": -max_penalty, "raw_max": 1.0 + max_reward, "denom": denom},
        },
    }
