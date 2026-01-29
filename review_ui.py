# review_ui.py
"""
Job Agent — Review UI (Streamlit)

This version:
- Removes "Human-in-the-loop" subtitle to save vertical space.
- Removes Streamlit header anchor UI (often shows as a “mystery rounded bar”).
- Keeps ONE bottom nav only, sticky, but with invisible wrapper (no extra rounded bar above buttons).
- Sidebar item picker: TITLE ONLY.
- No URL in overview (redundant with "Open job posting").
- Captured/Updated renamed to First seen / Last modified.
- Expanders for reasons + top resume scores.
- Index-based radio to avoid Streamlit ValueError.
- Atomic writes to queue.json.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


# -----------------------------
# Config
# -----------------------------

DEFAULT_QUEUE_PATH = Path("queue") / "queue.json"
ALLOWED_STATUSES = ["pending", "hold", "rejected", "approved"]

ACCENT = "#2563EB"  # blue-600
ACCENT_SOFT = "rgba(37, 99, 235, 0.10)"


# -----------------------------
# Helpers
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truncate(s: str, n: int = 85) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def extract_domain(url: str) -> str:
    if not url:
        return ""
    m = re.search(r"https?://([^/]+)/?", url)
    return m.group(1) if m else ""


def clean_title(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r"\s+auf\s+www\.[\w\.-]+\.\w+\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+auf\s+[\w\.-]+\.\w+\s*$", "", t, flags=re.IGNORECASE)
    return t.strip()


def pick_best_text_field(draft: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if not isinstance(draft, dict):
        return None

    preferred_keys = [
        "cover_letter",
        "coverLetter",
        "message",
        "pitch",
        "application_message",
        "applicationMessage",
        "email",
        "body",
        "text",
    ]
    for k in preferred_keys:
        v = draft.get(k)
        if isinstance(v, str) and v.strip():
            return (k, v.strip())

    best = None
    for k, v in draft.items():
        if isinstance(v, str) and len(v.strip()) >= 200:
            if best is None or len(v.strip()) > len(best[1]):
                best = (k, v.strip())
    return best


# -----------------------------
# Data model
# -----------------------------

@dataclass
class QueueItem:
    job_id: str
    title: str
    url: str
    source: str
    captured_at: str
    updated_at: str
    status: str
    decision: str
    scores: Dict[str, Any]
    reasons: List[str]
    selected_resume_id: str
    top_resume_scores: List[Dict[str, Any]]
    artifacts: Dict[str, Any]
    human_notes: str = ""
    approved_at: str = ""
    rejected_at: str = ""


def parse_queue_item(job_id: str, raw: Dict[str, Any]) -> QueueItem:
    return QueueItem(
        job_id=job_id,
        title=raw.get("title", job_id),
        url=raw.get("url", ""),
        source=raw.get("source", ""),
        captured_at=raw.get("captured_at", ""),
        updated_at=raw.get("updated_at", ""),
        status=raw.get("status", "pending"),
        decision=raw.get("decision", ""),
        scores=raw.get("scores", {}) if isinstance(raw.get("scores", {}), dict) else {},
        reasons=raw.get("reasons", []) if isinstance(raw.get("reasons", []), list) else [],
        selected_resume_id=raw.get("selected_resume_id", ""),
        top_resume_scores=raw.get("top_resume_scores", []) if isinstance(raw.get("top_resume_scores", []), list) else [],
        artifacts=raw.get("artifacts", {}) if isinstance(raw.get("artifacts", {}), dict) else {},
        human_notes=raw.get("human_notes", "") or "",
        approved_at=raw.get("approved_at", "") or "",
        rejected_at=raw.get("rejected_at", "") or "",
    )


# -----------------------------
# Page + CSS
# -----------------------------

st.set_page_config(page_title="Job Agent — Review UI", layout="wide")

st.markdown(
    f"""
<style>
/* Tighten vertical spacing */
.block-container {{
  padding-top: 0.55rem;
  padding-bottom: 0.9rem;
  max-width: 1320px;
}}

a {{ color: {ACCENT}; }}

/* Hide Streamlit header anchor/action elements (often renders as “mystery bar”) */
a[data-testid="stHeaderActionElements"],
div[data-testid="stHeaderActionElements"],
button[data-testid="stHeaderActionElements"] {{
  display: none !important;
}}

.header-card {{
  border: 1px solid rgba(49, 51, 63, 0.16);
  border-radius: 16px;
  padding: 12px 14px;
  margin-top: 6px;
  margin-bottom: 8px;
  background: rgba(255,255,255,0.02);
}}
.header-title {{
  margin: 0;
  line-height: 1.14;
  font-weight: 770;
  font-size: 1.65rem;
}}
.header-meta {{
  margin-top: 5px;
  opacity: 0.78;
  font-size: 0.92rem;
}}
.header-link {{
  margin-top: 6px;
}}
.header-link a {{
  color: {ACCENT};
  font-weight: 700;
  text-decoration: none;
}}
.header-link a:hover {{ text-decoration: underline; }}

.header-badges {{
  margin-top: 8px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}}
.badge {{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.16);
  font-size: 0.84rem;
  opacity: 0.95;
}}
.badge-strong {{
  border-color: rgba(37, 99, 235, 0.38);
  background: {ACCENT_SOFT};
}}

.kv-label {{ opacity: .75; font-size: .86rem; margin-bottom: 2px; }}
.kv-value {{ font-size: 1.0rem; margin-bottom: 8px; }}

.stButton>button {{
  border-radius: 12px;
  height: 44px;
  border: 1px solid rgba(37, 99, 235, 0.25);
}}

/* Sticky bottom nav — invisible wrapper (prevents extra rounded bar above buttons) */
.sticky-nav {{
  position: sticky;
  bottom: 10px;
  z-index: 50;
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
  box-shadow: none;
  backdrop-filter: none;
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Queue load
# -----------------------------

def load_queue(queue_path: Path) -> tuple[Dict[str, Any], Dict[str, QueueItem]]:
    if not queue_path.exists():
        stub = {
            "meta": {"created_at": now_iso(), "updated_at": now_iso(), "version": 1},
            "items_by_job_id": {},
        }
        atomic_write_json(queue_path, stub)

    q = load_json(queue_path)
    items_raw = q.get("items_by_job_id", {}) if isinstance(q, dict) else {}
    items: Dict[str, QueueItem] = {}
    if isinstance(items_raw, dict):
        for job_id, raw in items_raw.items():
            if isinstance(raw, dict):
                items[job_id] = parse_queue_item(job_id, raw)
    return q, items


# -----------------------------
# Navigation callbacks
# -----------------------------

def nav_prev() -> None:
    st.session_state["radio_idx"] = max(0, int(st.session_state.get("radio_idx", 0)) - 1)


def nav_next(max_idx: int) -> None:
    st.session_state["radio_idx"] = min(max_idx, int(st.session_state.get("radio_idx", 0)) + 1)


# -----------------------------
# Mutations
# -----------------------------

def update_item_status(queue_doc: Dict[str, Any], job_id: str, new_status: str) -> None:
    if new_status not in ALLOWED_STATUSES:
        return
    items_by_job_id = queue_doc.setdefault("items_by_job_id", {})
    it = items_by_job_id.setdefault(job_id, {})
    it["status"] = new_status
    it["updated_at"] = now_iso()
    if new_status == "approved":
        it["approved_at"] = now_iso()
    if new_status == "rejected":
        it["rejected_at"] = now_iso()
    queue_doc.setdefault("meta", {})
    queue_doc["meta"]["updated_at"] = now_iso()


def update_item_notes(queue_doc: Dict[str, Any], job_id: str, notes: str) -> None:
    items_by_job_id = queue_doc.setdefault("items_by_job_id", {})
    it = items_by_job_id.setdefault(job_id, {})
    it["human_notes"] = notes
    it["updated_at"] = now_iso()
    queue_doc.setdefault("meta", {})
    queue_doc["meta"]["updated_at"] = now_iso()


# -----------------------------
# Sidebar (filters)
# -----------------------------

with st.sidebar:
    st.markdown("### Filters")

    queue_path_str = st.text_input("Queue path", str(DEFAULT_QUEUE_PATH))
    queue_path = Path(queue_path_str)

    status_filter = st.selectbox("Status", ["all"] + ALLOWED_STATUSES, index=0)
    sort_mode = st.selectbox(
        "Sort by",
        ["hybrid score (desc)", "updated_at (desc)", "captured_at (desc)", "title (A→Z)"],
        index=0,
    )
    search = st.text_input("Search (title/url/job_id/resume)", "")

    st.divider()
    st.caption("Tip: Keep this local. queue.json and artifacts are private.")


queue_doc, items_map = load_queue(queue_path)
items = list(items_map.values())


def matches_search(it: QueueItem, s: str) -> bool:
    s = normalize_text(s).lower()
    if not s:
        return True
    hay = " ".join(
        [
            it.job_id,
            it.title,
            it.url,
            it.selected_resume_id,
            json.dumps(it.reasons, ensure_ascii=False),
        ]
    ).lower()
    return s in hay


def hybrid(it: QueueItem) -> float:
    return safe_float(it.scores.get("hybrid_score", 0.0))


filtered: List[QueueItem] = []
for it in items:
    if status_filter != "all" and it.status != status_filter:
        continue
    if not matches_search(it, search):
        continue
    filtered.append(it)

if sort_mode == "hybrid score (desc)":
    filtered.sort(key=lambda x: hybrid(x), reverse=True)
elif sort_mode == "updated_at (desc)":
    filtered.sort(key=lambda x: (x.updated_at or ""), reverse=True)
elif sort_mode == "captured_at (desc)":
    filtered.sort(key=lambda x: (x.captured_at or ""), reverse=True)
elif sort_mode == "title (A→Z)":
    filtered.sort(key=lambda x: (x.title or "").lower())


# -----------------------------
# Sidebar (items picker) — TITLE ONLY
# -----------------------------

with st.sidebar:
    st.markdown("### Items")
    if not filtered:
        st.info("No items match filter/search.")
        st.stop()

    labels: List[str] = [truncate(clean_title(it.title), 85) for it in filtered]

    if "radio_idx" not in st.session_state:
        st.session_state["radio_idx"] = 0
    st.session_state["radio_idx"] = max(0, min(int(st.session_state["radio_idx"]), len(filtered) - 1))

    st.radio(
        "Select job",
        options=list(range(len(filtered))),
        index=int(st.session_state["radio_idx"]),
        key="radio_idx",
        format_func=lambda i: labels[i],
    )


selected_idx = int(st.session_state["radio_idx"])
item = filtered[selected_idx]


# -----------------------------
# Top title (no subtitle)
# -----------------------------

top_l, top_r = st.columns([0.72, 0.28], vertical_alignment="center")
with top_l:
    st.title("Job Agent — Review UI")
with top_r:
    st.caption(f"Queue: {truncate(str(queue_path), 60)}")


# -----------------------------
# Header
# -----------------------------

def render_header(it: QueueItem, idx: int, total: int) -> None:
    url = it.url
    source = it.source or extract_domain(url) or "unknown"

    hyb = safe_float(it.scores.get("hybrid_score", 0.0))
    sem = safe_float(it.scores.get("semantic_similarity", 0.0))
    rule = safe_float(it.scores.get("rule_score", 0.0))
    dom = safe_int(it.scores.get("domain_distance", 0))
    decision = it.decision or "—"
    resume_id = it.selected_resume_id or "—"

    st.markdown('<div class="header-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='header-title'>{clean_title(it.title)}</div>", unsafe_allow_html=True)

    meta = f"Source: <b>{source}</b> • Item <b>{idx+1}</b> / <b>{total}</b> • Status: <b>{it.status}</b>"
    st.markdown(f"<div class='header-meta'>{meta}</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='header-link'>🔗 <a href='{url}' target='_blank'>Open job posting</a></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='header-badges'>"
        f"<span class='badge badge-strong'>Hybrid <b>{hyb:0.2f}</b></span>"
        f"<span class='badge'>Semantic <b>{sem:0.2f}</b></span>"
        f"<span class='badge'>Rule <b>{rule:0.2f}</b></span>"
        f"<span class='badge'>Dom <b>{dom}</b></span>"
        f"<span class='badge'>Decision <b>{decision}</b></span>"
        f"<span class='badge'>Resume <b>{resume_id}</b></span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


render_header(item, selected_idx, len(filtered))


# -----------------------------
# Tabs
# -----------------------------

tab_overview, tab_draft, tab_artifacts, tab_actions = st.tabs(["Overview", "Draft", "Artifacts", "Notes & Actions"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='kv-label'>Job ID</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv-value'><b>{item.job_id}</b></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='kv-label'>First seen</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv-value'>{item.captured_at or '—'}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='kv-label'>Last modified</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv-value'>{item.updated_at or '—'}</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='kv-label'>Selected resume</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv-value'>{item.selected_resume_id or '—'}</div>", unsafe_allow_html=True)

    c5, c6, c7 = st.columns(3)
    with c5:
        st.markdown("<div class='kv-label'>Decision</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv-value'>{item.decision or '—'}</div>", unsafe_allow_html=True)
    with c6:
        st.markdown("<div class='kv-label'>Source</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='kv-value'>{item.source or extract_domain(item.url) or 'unknown'}</div>",
            unsafe_allow_html=True,
        )
    with c7:
        st.markdown("<div class='kv-label'>Status</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv-value'>{item.status}</div>", unsafe_allow_html=True)

    with st.expander("Show reasons", expanded=False):
        if item.reasons:
            for r in item.reasons:
                st.write(f"- {r}")
        else:
            st.caption("No reasons recorded.")

    with st.expander("Show top resume scores", expanded=False):
        trs = item.top_resume_scores or []
        if trs:
            for row in trs[:12]:
                rid = row.get("resume_id", "—")
                hyb = safe_float(row.get("hybrid_score", 0.0))
                sem = safe_float(row.get("semantic_similarity", 0.0))
                rule = safe_float(row.get("rule_score", 0.0))
                dom = safe_int(row.get("domain_distance", 0))
                st.write(f"- **{rid}** — hybrid {hyb:0.2f}, semantic {sem:0.2f}, rule {rule:0.2f}, dom {dom}")
        else:
            st.caption("No multi-resume scoring data.")

with tab_draft:
    draft_path = item.artifacts.get("draft_path") if isinstance(item.artifacts, dict) else None
    if not draft_path:
        st.info("No draft path recorded for this item.")
    else:
        p = Path(draft_path)
        if not p.exists():
            st.warning(f"Draft file not found: {draft_path}")
        else:
            try:
                draft = load_json(p)
            except Exception as e:
                st.error(f"Failed to load draft JSON: {e}")
                draft = None

            if isinstance(draft, dict):
                best = pick_best_text_field(draft)
                if best:
                    k, text = best
                    st.caption(f"Field: {k}")
                    st.text_area("Draft", value=text, height=420)
                else:
                    st.info("Draft JSON loaded, but no obvious text field found.")
                with st.expander("Raw draft JSON"):
                    st.json(draft)
            else:
                st.write(draft)

with tab_artifacts:
    art = item.artifacts if isinstance(item.artifacts, dict) else {}
    if not art:
        st.info("No artifacts recorded.")
    else:
        for k, v in art.items():
            st.write(f"**{k}**")
            st.code(str(v) if v is not None else "null")

with tab_actions:
    notes_key = f"notes_{item.job_id}"
    if notes_key not in st.session_state:
        st.session_state[notes_key] = item.human_notes or ""

    st.text_area("Human notes", key=notes_key, height=140)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Save notes", use_container_width=True):
            update_item_notes(queue_doc, item.job_id, st.session_state[notes_key])
            atomic_write_json(queue_path, queue_doc)
            st.success("Notes saved.")
    with b2:
        if st.button("Approve", use_container_width=True):
            update_item_status(queue_doc, item.job_id, "approved")
            atomic_write_json(queue_path, queue_doc)
            st.success("Approved.")
    with b3:
        if st.button("Hold", use_container_width=True):
            update_item_status(queue_doc, item.job_id, "hold")
            atomic_write_json(queue_path, queue_doc)
            st.success("On hold.")
    with b4:
        if st.button("Reject", use_container_width=True):
            update_item_status(queue_doc, item.job_id, "rejected")
            atomic_write_json(queue_path, queue_doc)
            st.success("Rejected.")


# -----------------------------
# Bottom nav (sticky, single)
# -----------------------------
st.markdown("<div class='sticky-nav'>", unsafe_allow_html=True)
nav_l, nav_mid, nav_r = st.columns([1, 0.35, 1], vertical_alignment="center")
with nav_l:
    st.button("← Prev", use_container_width=True, on_click=nav_prev)
with nav_mid:
    st.markdown(
        f"<div style='text-align:center; opacity:.78; font-weight:650;'>"
        f"{selected_idx+1}/{len(filtered)}"
        f"</div>",
        unsafe_allow_html=True,
    )
with nav_r:
    st.button("Next →", use_container_width=True, on_click=nav_next, args=(len(filtered) - 1,))
st.markdown("</div>", unsafe_allow_html=True)
