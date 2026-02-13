Scheduler
  ↓
run_agent.py  (single orchestrator)
  ↓
┌───────────────────────────────────────────────┐
│ Tools (existing modules; no duplication)       │
│  - fetch_jobs.py / fetch_job_details.py        │
│  - extract_job_profile.py                      │
│  - score_job.py                                │
│  - prepare_application_draft.py (later)         │
│  - queue_utils.py (single write path)          │
└───────────────────────────────────────────────┘
  ↓
queue.json  (single source of truth: job records + status)
  ↓
review_ui.py  (human gate: approve / reject / regenerate)
