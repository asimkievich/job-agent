# Job Agent

An experimental AI-powered job application assistant that automates the repetitive parts of professional job searching while keeping humans in full control.

This project explores how agentic workflows, LLMs, and browser automation can be combined into a disciplined, privacy-first job search pipeline.

---

## Vision

The goal of Job Agent is to build a local, human-supervised AI assistant that:

- discovers relevant job or freelance opportunities
- evaluates relevance using structured scoring
- adapts resumes to each posting
- generates tailored cover letters or pitches
- prepares application payloads
- presents everything in a review UI
- submits applications only after approval

This is **not** a spam bot.  
It is a precision assistant designed to increase quality, consistency, and speed.

The architecture is intentionally modular so the system can expand from a single platform prototype into a multi-platform job ecosystem assistant.

---

## Core Workflow

### 1. Job Discovery
The agent searches supported job platforms.

Current target:
- Freelancermap.de

Future expansion:
- additional job boards
- recruiter portals
- pitch-based marketplaces
- direct company application sites

Each platform is implemented as a connector module that normalizes job data into a shared schema.

---

### 2. Relevance Evaluation

Jobs are scored against one or more professional profiles.

The system evaluates:
- skill overlap
- domain adjacency
- seniority alignment
- role fit
- risk / gap analysis

Outputs include:
- relevance score
- explainable reasoning
- pursue vs skip decision

All decisions are stored for auditability.

---

### 3. Resume Adaptation

For relevant jobs, the agent generates a tailored resume version.

This includes:
- emphasis shifts
- reordered achievements
- skill highlighting
- keyword alignment

The system never fabricates experience.  
It acts as an editor, not an inventor.

Resume changes are stored as structured diffs so the user can review exactly what changed.

---

### 4. Cover Letter / Pitch Generation

Depending on platform requirements:

- cover letter
- pitch message
- custom introduction
- form text

Drafts are generated in the language of the job posting and aligned with the adapted resume.

Engineering-heavy roles include explicit risk disclosures when appropriate.

All drafts remain editable before submission.

---

### 5. Human Review Interface

A local web interface acts as the control center.

For each candidate application, the user sees:

- job summary
- relevance explanation
- tailored resume diff
- generated cover letter
- risks and gaps
- extracted requirements

Available actions:

- approve & send
- approve but hold
- regenerate draft
- request modification
- reject

No application is sent automatically.

The UI also provides:

- pending application queue
- rejected project archive
- statistics dashboard

This keeps the human in command while eliminating repetitive labor.

---

### 6. Application Execution

After approval, the agent:

- fills platform forms
- uploads tailored documents
- submits the application
- confirms success
- stores a receipt

Errors are logged and surfaced in the UI.

---

## Architecture Principles

- **Modular platform connectors**  
  Each job website is a plugin, not hardcoded logic.

- **Strict structured data schema**  
  Jobs, resumes, and drafts use JSON models.

- **Human-in-the-loop design**  
  Automation assists — humans decide.

- **Privacy-first**  
  Local processing, secrets outside repo, no sensitive data committed.

- **Auditability**  
  Every decision is explainable and reproducible.

---

## Future Extensions

- multi-profile career strategy
- feedback learning from outcomes
- A/B testing resume variants
- analytics-driven improvement
- recruiter response tracking
- interview pipeline integration
- calendar integration
- CRM-style opportunity tracking

The system is designed to evolve into a personal job search operating system.

---

## Current Status

- ✅ Playwright smoke test working
- ✅ Project scaffolding complete
- 🚧 Active development


## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.\.venv\Scripts\Activate.ps1   # Windows

pip install -r requirements.txt
playwright install
python smoke_test.py
```


## Disclaimer

This project is experimental and intended for research, education, and personal automation use.

Users are solely responsible for ensuring that their use of this software complies with the terms of service, policies, and legal requirements of any platform they interact with.

The authors assume no responsibility for misuse, platform violations, or consequences arising from automated interactions.

This tool is designed to assist human decision-making, not to perform unsupervised mass automation or spam.

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.
