import json
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any

from langgraph.graph import StateGraph, END

import score_job
import prepare_application_draft


class AgentState(TypedDict, total=False):
    job_work_item_path: str
    job_profile_path: str
    resume_path: str
    search_path: str

    job_id: str

    scoring_result: Dict[str, Any]
    approved: Optional[bool]

    draft_path: Optional[str]
    draft: Optional[Dict[str, Any]]


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def node_init(state: AgentState) -> AgentState:
    profile = load_json(Path(state["job_profile_path"]))
    job_id = profile.get("job_id") or "unknown_job"
    state["job_id"] = str(job_id)
    return state


def node_score(state: AgentState) -> AgentState:
    score_job.JOB_PROFILE = Path(state["job_profile_path"])
    score_job.RESUME = Path(state["resume_path"])
    score_job.SEARCH = Path(state["search_path"])

    job = score_job.load(score_job.JOB_PROFILE)
    resume = score_job.load(score_job.RESUME)
    search = score_job.load(score_job.SEARCH)

    rs = score_job.rule_score(job, search)

    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")

    resume_text = score_job.build_resume_text(resume)
    job_text = score_job.build_job_text(job)

    emb_resume = model.encode(resume_text, convert_to_tensor=True, normalize_embeddings=True)
    emb_job = model.encode(job_text, convert_to_tensor=True, normalize_embeddings=True)
    sim = float(util.cos_sim(emb_resume, emb_job).item())

    rule_norm = max(0.0, min(1.0, (rs["rule_score"] / 0.8)))
    
    sim01 = max(0.0, min(1.0, sim))
    domain_bonus = max(0.0, 1.0 - (rs["domain_distance"] / 3.0))

    hybrid = min(1.0, (0.65 * sim01) + (0.35 * rule_norm) + (0.10 * domain_bonus))

    decision = "pursue" if hybrid >= 0.55 and rs["domain_distance"] < 3 else "skip"

    state["scoring_result"] = {
        "rule_score": rs["rule_score"],
        "semantic_similarity": round(sim, 3),
        "hybrid_score": round(hybrid, 3),
        "domain_distance": rs["domain_distance"],
        "decision": decision,
        "reasons": rs["reasons"],
        "job_title": job.get("title"),
        "job_url": job.get("url"),
        "job_id": state.get("job_id"),
    }
    return state


def node_human_gate(state: AgentState) -> AgentState:
    r = state["scoring_result"]

    print("\n=== JOB REVIEW ===")
    print(f"Job ID: {r.get('job_id')}")
    print(f"Title:  {r.get('job_title')}")
    print(f"URL:    {r.get('job_url')}")
    print("")
    print(f"Rule score:          {r['rule_score']}")
    print(f"Semantic similarity: {r['semantic_similarity']}")
    print(f"Hybrid score:        {r['hybrid_score']}")
    print(f"Domain distance:     {r['domain_distance']}")
    print(f"Suggested decision:  {r['decision']}")
    print("\nReasons:")
    for x in r["reasons"]:
        print(f" - {x}")

    ans = input("\nApprove pursuing this job? (y/n) ").strip().lower()
    state["approved"] = ans == "y"
    return state


def node_prepare_draft(state: AgentState) -> AgentState:
    job_id = state.get("job_id", "unknown_job")

    prepare_application_draft.WORK_ITEM = Path(state["job_work_item_path"])
    prepare_application_draft.RESUME_PROFILE = Path(state["resume_path"])

    out_path = Path(f"work_items/{job_id}.application_draft.json")
    prepare_application_draft.OUT_PATH = out_path

    prepare_application_draft.main()

    state["draft_path"] = str(out_path)
    state["draft"] = json.loads(out_path.read_text(encoding="utf-8"))
    return state


def route_after_gate(state: AgentState) -> str:
    return "prepare_draft" if state.get("approved") else END


def main():
    state: AgentState = {
        "job_work_item_path": str(Path("work_items/job_001.json")),
        "job_profile_path": str(Path("work_items/job_001.profile.json")),
        "resume_path": str(Path("profiles/resumes/resume_product_owner_ai_pm.json")),
        "search_path": str(Path("profiles/resumes/search_profile.json")),
    }

    g = StateGraph(AgentState)
    g.add_node("init", node_init)
    g.add_node("score", node_score)
    g.add_node("human_gate", node_human_gate)
    g.add_node("prepare_draft", node_prepare_draft)

    g.set_entry_point("init")
    g.add_edge("init", "score")
    g.add_edge("score", "human_gate")
    g.add_conditional_edges("human_gate", route_after_gate)
    g.add_edge("prepare_draft", END)

    app = g.compile()
    final = app.invoke(state)

    Path("runs").mkdir(exist_ok=True)
    run_path = Path(f"runs/run_{final.get('job_id', 'unknown_job')}.json")
    run_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOK: saved run state to {run_path.resolve()}")


if __name__ == "__main__":
    main()
